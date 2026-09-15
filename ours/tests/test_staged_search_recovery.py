"""Report compatibility and original-loop recovery, with no paid inference."""
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
from ours.staged_search_recovery import (FAILURE, RecoveredSearch, check_amendment,
    compatible_proposal, normalize_report, prepare)
from ours.visual_task import file_sha256


class ReportTests(unittest.TestCase):
    def fixture(self,root):
        view=root/'view';(view/'harnesses/h0').mkdir(parents=True)
        (view/'harnesses/h0/harness.py').write_text('baseline')
        before=tree_hashes(view)
        (view/'harnesses/h1').mkdir();(view/'harnesses/h1/harness.py').write_text('candidate')
        (view/'logs/iteration_1').mkdir(parents=True)
        (view/'logs/iteration_1/report.md').write_text('Original report prose\n')
        return view,before

    def test_normalization_preserves_bytes_and_accepts_only_current_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);view,before=self.fixture(root)
            hashes=tree_hashes(view)
            allowed=normalize_report(view,before,['h1'],1,root/'archive')
            self.assertEqual(allowed,['harnesses/h1/harness.py','logs/iteration_001/report.md'])
            self.assertEqual((root/'archive/report-original.md').read_bytes(),(view/'logs/iteration_001/report.md').read_bytes())
            self.assertEqual(json.loads((root/'archive/normalization.json').read_text())['before_normalization_sha256'],hashes)
            self.assertEqual(normalize_report(view,before,['h1'],1,root/'unused'),allowed)
            self.assertFalse((root/'unused').exists())

    def test_protected_edits_extra_slots_other_iterations_and_duplicate_reports_still_fail(self):
        for case in ('protected','slot','iteration','duplicate','symlink'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);view,before=self.fixture(root)
                if case=='protected':(view/'harnesses/h0/harness.py').write_text('altered')
                elif case=='symlink':(view/'extra').symlink_to(view/'harnesses/h0/harness.py')
                else:
                    path=view/({'slot':'harnesses/h2/harness.py','iteration':'logs/iteration_2/report.md',
                                'duplicate':'logs/iteration_001/report.md'}[case])
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('extra')
                with self.assertRaises(ValueError):normalize_report(view,before,['h1'],1,root/'archive')
                self.assertFalse((root/'archive').exists())
                self.assertTrue((view/'logs/iteration_1/report.md').exists())


@unittest.skipUnless(importlib.util.find_spec('chess'),'Requires the native Chess runtime')
class NativeRecoveryTests(unittest.TestCase):
    def test_first_proposal_recovery_matches_uninterrupted_five_rounds_and_perfect_stop(self):
        from autoharness_chess_puzzle import runner
        from meta_harness import meta_harness_chess_puzzle as native
        from ours.tests.test_prompt_subspace import BASELINE,replace_user,write_harness
        terminal='JobId=123 JobState=COMPLETED ExitCode=0:0 RunTime=00:02:00 AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h800=2\n'
        @contextmanager
        def fake_relay(*args,**kwargs):yield {'base_url':'http://127.0.0.1:1','token':'offline-fixture'}
        for perfect in (False,True):
            histories=[];allocations=[];evaluations=[]
            for recover in (False,True):
                with self.subTest(perfect=perfect,recover=recover),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);trial=root/'trial';plan_path=root/'plan.json'
                    plan={'phase_root':str(trial),'condition':'harness_only','seed':42,'dataset':'fixture',
                        'target_config':{'model':'fixture'},'incoming_harness':str(BASELINE.resolve()),
                        'data_role':'mh_val','max_cli_turns':12,'max_api_requests':12,
                        'limitations':['Synthetic CPU fixture; no model or API calls.']}
                    write_json(plan_path,plan)
                    proposed=[];evaluated=[]
                    def proposer(**kw):
                        names=[f'h{n}' for n in range(3*(kw['iteration']-1)+1,3*kw['iteration']+1)]
                        proposed.append(names)
                        for name in names:
                            write_harness(kw['run_dir'],name,replace_user(BASELINE.read_text(),repr(name+' {observation}')))
                        write_json(kw['run_dir']/'pending_eval.json',{'candidates':[{'slot':n} for n in names]})
                        report=kw['run_dir']/f"logs/iteration_{kw['iteration']}/report.md"
                        report.parent.mkdir(parents=True,exist_ok=True);report.write_text('Fixture prose, no evidence.\n')
                        return SimpleNamespace(exit_code=0,duration_seconds=0.,show=lambda:None)
                    def command(argv,**kw):
                        if argv[0]=='squeue':return ''
                        if argv[0]=='scontrol':return terminal
                        self.assertEqual(argv[:3],['sbatch','--parsable','ours/run_staged_evaluation.sh'])
                        private=Path(argv[-1]);request=json.loads((private/'request.json').read_text())
                        name=Path(request['kwargs']['harness_path']).parent.name;evaluated.append(name)
                        solved=32 if perfect and name=='h2' else int(name[1:])
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
                         patch('ours.staged_search_recovery.BudgetJournal',return_value=journal), \
                         patch('ours.scoped_proposer.relay',fake_relay),patch.object(native,'propose_claude',proposer), \
                         patch('ours.staged_search.isolated_native_proposal',isolated_native_proposal if recover else compatible_proposal):
                        if not recover:
                            StagedSearch(plan,plan_path,submit=True).run()
                        else:
                            with self.assertRaisesRegex(ValueError,FAILURE):StagedSearch(plan,plan_path,submit=True).run()
                            self.assertEqual((evaluated,proposed),(['h0'],[['h1','h2','h3']]))
                            raw=tree_hashes(trial/'proposal-1',ignore_cli_state=False)
                            amendment_path=root/'amendment.json'
                            with patch('ours.staged_search_recovery.check_plan',return_value=plan), \
                                 patch('ours.staged_search_recovery.os.kill',side_effect=ProcessLookupError):
                                prepare(plan_path,amendment_path)
                                # A modified candidate must invalidate recovery before any paid call.
                                candidate=trial/'proposal-1/proposer-workspace/harnesses/h1/harness.py'
                                saved=candidate.read_bytes();candidate.write_bytes(saved+b'\n# tampered\n')
                                with self.assertRaisesRegex(ValueError,'Changed interrupted trial evidence'):check_amendment(amendment_path)
                                candidate.write_bytes(saved)
                                amendment,checked,original=check_amendment(amendment_path)
                            RecoveredSearch(checked,original,submit=True).run_recovered(amendment,amendment_path)
                            self.assertEqual(tree_hashes(trial/'proposal-1',ignore_cli_state=False),raw)
                            with self.assertRaises(FileExistsError):
                                RecoveredSearch(checked,original).run_recovered(amendment,amendment_path)
                    result=json.loads((trial/'result.json').read_text())
                    self.assertEqual(result['status'],'COMPLETED_NATIVE_SEARCH')
                    histories.append([json.loads(p.read_text())['accepted_harness'] for p in sorted((trial/'search/logs').glob('iteration_*/comparison.json'))])
                    allocations.append(proposed);evaluations.append(evaluated)
            self.assertEqual(histories[0],histories[1]);self.assertEqual(allocations[0],allocations[1]);self.assertEqual(evaluations[0],evaluations[1])
            self.assertEqual(len(allocations[1]),1 if perfect else 5)
            self.assertEqual(evaluations[1].count('h0'),1)
            print(json.dumps({'kind':'native_first_proposal_recovery_fixture','status':'PASS',
                'perfect_stop':perfect,'accepted_history':histories[1],'evaluations':evaluations[1],
                'proposals_generated':len(allocations[1]),'new_api_calls':0,'scores_are_synthetic':True}),flush=True)


if __name__=='__main__':unittest.main()
