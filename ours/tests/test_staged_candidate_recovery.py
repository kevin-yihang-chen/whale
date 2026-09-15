"""Real native round continuation with synthetic scores; no provider/GPU calls."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.native_search import tree_hashes, write_json
from ours.scoped_proposer import isolated_native_proposal
from ours.staged_search import StagedSearch
from ours import staged_search_recovery as first
from ours import staged_metadata_recovery as second
from ours.staged_candidate_recovery import (FAILURE, MetadataRecoveredSearch, check_amendment,
    check_metadata_proposal, compatible_proposal, normalize_metadata, prepare)
from ours.visual_task import file_sha256


class MetadataTests(unittest.TestCase):
    def test_lossless_alias_mapping_and_ambiguity_rejection(self):
        payload=[{'candidate':'h4','path':'harnesses/h4/harness.py','hypothesis':'unverified prose'}]
        normalized=normalize_metadata(payload,['h4'],{'h1'})
        self.assertEqual(normalized,{'candidates':[dict(payload[0],name='h4')]})
        self.assertNotIn('name',payload[0])
        for entry in ({'candidate':'h4','slot':'h5'}, {'candidate':'h4','name':'h5'},{'id':'h6'},{'name':'h4','parent':'h99'},
                      {'name':'h4','path':'../outside.py'},{'harness':4},{}):
            with self.subTest(entry=entry),self.assertRaises(ValueError):
                normalize_metadata([entry],['h4','h5'],{'h1'})
        for payload in ([],[{'name':'h4'},{'name':'h4'}],{'candidates':'h4'}):
            with self.assertRaises(ValueError):normalize_metadata(payload,['h4'],{'h1'})

    def test_format_compatibility_does_not_allow_protected_or_extra_files(self):
        for case in ('valid','protected','extra','symlink'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);view=root/'view';(view/'harnesses/h1').mkdir(parents=True)
                (view/'harnesses/h1/harness.py').write_text('baseline')
                before=tree_hashes(view)
                (view/'harnesses/h4').mkdir();(view/'harnesses/h4/harness.py').write_text('candidate')
                write_json(view/'pending_eval.json',[{'candidate':'h4'}])
                report=view/'logs/iteration_2/report.md';report.parent.mkdir(parents=True);report.write_text('prose')
                if case=='protected':(view/'harnesses/h1/harness.py').write_text('tampered')
                elif case=='extra':(view/'extra').write_text('extra')
                elif case=='symlink':(view/'extra').symlink_to(view/'pending_eval.json')
                if case=='valid':
                    allowed=check_metadata_proposal(view,before,['h4'],2,root/'archive')
                    self.assertIn('logs/iteration_002/report.md',allowed)
                    self.assertEqual(json.loads((view/'pending_eval.json').read_text()),[{'candidate':'h4'}])
                else:
                    with self.assertRaises(ValueError):check_metadata_proposal(view,before,['h4'],2,root/'archive')
                    self.assertFalse((root/'archive').exists())


@unittest.skipUnless(importlib.util.find_spec('chess'),'Requires native Chess runtime')
class NativeRecoveryTests(unittest.TestCase):
    def test_three_interruptions_preserve_original_rounds_and_perfect_stop(self):
        from autoharness_chess_puzzle import runner
        from meta_harness import meta_harness_chess_puzzle as native
        from ours.tests.test_prompt_subspace import BASELINE,replace_user,write_harness
        terminal='JobId=123 JobState=COMPLETED ExitCode=0:0 RunTime=00:02:00 AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h800=2\n'
        @contextmanager
        def fake_relay(*args,**kwargs):yield {'base_url':'http://127.0.0.1:1','token':'offline-fixture'}
        for perfect in (False,True):
            outcomes=[]
            for recover in (False,True):
                with self.subTest(perfect=perfect,recover=recover),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);trial=root/'trial';plan_path=root/'plan.json'
                    plan={'phase_root':str(trial),'condition':'harness_only','seed':42,'dataset':'fixture',
                        'target_config':{'model':'fixture'},'incoming_harness':str(BASELINE.resolve()),
                        'data_role':'mh_val','max_cli_turns':12,'max_api_requests':12,
                        'limitations':['Synthetic CPU fixture; no model or API calls.']}
                    write_json(plan_path,plan);proposed=[];evaluated=[]
                    def proposer(**kw):
                        names=[f'h{n}' for n in range(3*(kw['iteration']-1)+1,3*kw['iteration']+1)]
                        proposed.append(names)
                        for name in names:write_harness(kw['run_dir'],name,replace_user(BASELINE.read_text(),repr(name+' {observation}')))
                        payload=({'candidates':[{'slot':n} for n in names]} if kw['iteration']==1 else [{('harness' if kw['iteration']==2 else 'candidate'):n} for n in names])
                        write_json(kw['run_dir']/'pending_eval.json',payload)
                        report=kw['run_dir']/f"logs/iteration_{kw['iteration']}/report.md"
                        report.parent.mkdir(parents=True,exist_ok=True);report.write_text('Fixture prose, no evidence.\n')
                        return SimpleNamespace(exit_code=0,duration_seconds=0.,show=lambda:None)
                    def command(argv,**kw):
                        if argv[0]=='squeue':return ''
                        if argv[0]=='scontrol':return terminal
                        self.assertEqual(argv[:3],['sbatch','--parsable','ours/run_staged_evaluation.sh'])
                        private=Path(argv[-1]);request=json.loads((private/'request.json').read_text())
                        name=Path(request['kwargs']['harness_path']).parent.name;evaluated.append(name)
                        solved=32 if perfect and name=='h8' else int(name[1:])
                        outputs=[{'example_id':'redacted','reward':float(i<solved),'turn_count':1,
                                  'metrics':{'assistant_tokens_used':1},'prompt':[],'harness_trace':[]} for i in range(32)]
                        summary=runner.summarize_outputs(outputs,{'seed':42,'fixture_only':True})
                        merged=private/'merged';merged.mkdir();write_json(merged/'val.json',summary);runner.write_trajectories(outputs,merged)
                        write_json(private/'audit.json',{'status':'PASS','examples':32,'seed':42,'plan_sha256':file_sha256(plan_path),
                            'audit_source_sha256':file_sha256(Path('ours/audit_staged_evaluation.py')),
                            'harness_sha256':request['harness_sha256'],'solved':solved,'calls':32,'generated_tokens':32})
                        write_json(private/'result.json',{'status':'PASS','job_id':'123','plan_sha256':file_sha256(plan_path),
                            'request_sha256':file_sha256(private/'request.json'),'artifact_sha256':tree_hashes(private),
                            'solved':solved,'new_model_calls':32,'generated_tokens':32})
                        return '123\n'
                    journal=SimpleNamespace(summary=lambda:{'unresolved_calls':0,'fixture_only':True})
                    with patch('ours.staged_search.subprocess.check_output',command), \
                         patch('ours.staged_search.BudgetJournal',return_value=journal), \
                         patch.object(first,'BudgetJournal',return_value=journal), \
                         patch('ours.staged_metadata_recovery.BudgetJournal',return_value=journal), \
                         patch('ours.staged_candidate_recovery.BudgetJournal',return_value=journal), \
                         patch('ours.scoped_proposer.relay',fake_relay),patch.object(native,'propose_claude',proposer), \
                         patch('ours.staged_search.isolated_native_proposal',isolated_native_proposal if recover else compatible_proposal):
                        if not recover:
                            StagedSearch(plan,plan_path,submit=True).run()
                        else:
                            with self.assertRaisesRegex(ValueError,first.FAILURE):StagedSearch(plan,plan_path,submit=True).run()
                            path1=root/'amendment1.json'
                            with patch.object(first,'check_plan',return_value=plan),patch.object(first.os,'kill',side_effect=ProcessLookupError):
                                first.prepare(plan_path,path1)
                                amendment,checked,original=first.check_amendment(path1)
                            with self.assertRaisesRegex(ValueError,second.FAILURE):
                                first.RecoveredSearch(checked,original,submit=True).run_recovered(amendment,path1)
                            self.assertEqual(evaluated,['h0','h1','h2','h3']);self.assertEqual(len(proposed),2)
                            evidence={str(p):tree_hashes(p) for p in (trial/'proposal-1',trial/'proposal-2',trial/'recovery-1')}
                            completed=(trial/'search/logs/iteration_001/comparison.json').read_bytes()
                            path2=root/'amendment2.json'
                            with patch('ours.staged_metadata_recovery.check_plan',return_value=plan), \
                                 patch('ours.staged_metadata_recovery.os.kill',side_effect=ProcessLookupError):
                                second.prepare(plan_path,path2)
                                amendment,checked,original=second.check_amendment(path2)
                            with self.assertRaisesRegex(ValueError,FAILURE):
                                second.MetadataRecoveredSearch(checked,original,submit=True).run_recovered(amendment,path2)
                            self.assertEqual(evaluated,[f'h{n}' for n in range(7)])
                            self.assertEqual(len(proposed),3)
                            evidence={str(p):tree_hashes(p) for p in (trial/'proposal-1',trial/'proposal-2',
                                trial/'proposal-3',trial/'recovery-1',trial/'recovery-2')}
                            completed={str(p):p.read_bytes() for p in (trial/'search/logs/iteration_001/comparison.json',
                                trial/'search/logs/iteration_002/comparison.json',trial/'search/logs/evolution_summary.jsonl')}
                            path3=root/'amendment3.json'
                            with patch('ours.staged_candidate_recovery.check_plan',return_value=plan), \
                                 patch('ours.staged_candidate_recovery.os.kill',side_effect=ProcessLookupError):
                                prepare(plan_path,path3)
                                candidate=trial/'proposal-3/proposer-workspace/harnesses/h7/harness.py'
                                saved=candidate.read_bytes();candidate.write_bytes(saved+b'\n# tampered\n')
                                with self.assertRaisesRegex(ValueError,'Changed interrupted trial evidence'):check_amendment(path3)
                                candidate.write_bytes(saved)
                                amendment,checked,original=check_amendment(path3)
                            MetadataRecoveredSearch(checked,original,submit=True).run_recovered(amendment,path3)
                            self.assertEqual(evidence,{p:tree_hashes(Path(p)) for p in evidence})
                            for path,value in completed.items():
                                if path.endswith('jsonl'):self.assertTrue(Path(path).read_bytes().startswith(value))
                                else:self.assertEqual(Path(path).read_bytes(),value)
                            self.assertEqual(json.loads((trial/'recovery-3/proposal-replay.json').read_text())['new_api_calls'],0)
                            with self.assertRaises(FileExistsError):MetadataRecoveredSearch(checked,original).run_recovered(amendment,path3)
                    self.assertEqual(json.loads((trial/'result.json').read_text())['status'],'COMPLETED_NATIVE_SEARCH')
                    history=[json.loads(p.read_text())['accepted_harness'] for p in sorted((trial/'search/logs').glob('iteration_*/comparison.json'))]
                    evolution=[json.loads(line) for line in (trial/'search/logs/evolution_summary.jsonl').read_text().splitlines()]
                    outcomes.append((history,proposed,evaluated,evolution))
            self.assertEqual(outcomes[0],outcomes[1])
            self.assertEqual(len(outcomes[1][1]),3 if perfect else 5)
            self.assertEqual(len(outcomes[1][2]),10 if perfect else 16)
            print(json.dumps({'kind':'native_third_metadata_recovery_fixture','status':'PASS',
                'perfect_stop':perfect,'accepted_history':outcomes[1][0],
                'evaluations':outcomes[1][2],'new_api_calls':0,'scores_are_synthetic':True}),flush=True)


if __name__=='__main__':unittest.main()
