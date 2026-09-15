"""Original search and final selection on synthetic future-seed fixtures."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.native_search import write_json,tree_hashes
from ours.visual_task import file_sha256


@unittest.skipUnless(importlib.util.find_spec('chess') and importlib.util.find_spec('transformers'),'Requires native Chess runtime')
class IndependentHarnessSearchTests(unittest.TestCase):
    def test_native_five_rounds_metadata_shapes_seed_propagation_and_perfect_stop(self):
        from autoharness_chess_puzzle import runner
        from meta_harness import meta_harness_chess_puzzle as native
        from ours.independent_harness_search import IndependentHarnessSearch
        from ours.tests.test_prompt_subspace import BASELINE,replace_user,write_harness
        from ours.search_completion import audit_archive
        @contextmanager
        def relay(*args,**kwargs):yield {'base_url':'http://127.0.0.1:1','token':'offline-fixture'}
        for seed,perfect in ((43,False),(44,False),(43,True)):
            with self.subTest(seed=seed,perfect=perfect),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);trial=root/'trial';path=root/'plan.json'
                plan={'phase_root':str(trial),'condition':'harness_only','seed':seed,'dataset':'fixture',
                    'target_config':{'model':'fixture','seed':seed},'incoming_harness':str(BASELINE.resolve()),
                    'data_role':'mh_val','max_cli_turns':12,'max_api_requests':12,'iterations':5,'proposals_per_iter':3,
                    'limitations':['Synthetic score/proposer/Slurm fixture; no actual independent trial.']}
                write_json(path,plan);evaluated=[];proposed=[]
                def proposer(**kw):
                    names=native.next_harness_names(kw['run_dir'],3);proposed.append(names)
                    for name in names:write_harness(kw['run_dir'],name,replace_user(BASELINE.read_text(),repr(name+' {observation}')))
                    alias=('slot','harness','candidate')[min(kw['iteration']-1,2)]
                    candidates=[{alias:n} for n in names]
                    write_json(kw['run_dir']/'pending_eval.json',{'candidates':candidates} if alias=='slot' else candidates)
                    report=kw['run_dir']/f"logs/iteration_{kw['iteration']}/report.md"
                    report.parent.mkdir(parents=True,exist_ok=True);report.write_text('Synthetic proposer fixture.\n')
                    return SimpleNamespace(exit_code=0,duration_seconds=0.,show=lambda:None)
                def command(argv,**kwargs):
                    if argv[0]=='squeue':return ''
                    if argv[0]=='scontrol':return 'JobId=123 JobState=COMPLETED ExitCode=0:0 RunTime=00:02:00 AllocTRES=gres/gpu=2,gres/gpu:h800=2\n'
                    self.assertEqual(argv[:3],['sbatch','--parsable','ours/run_staged_evaluation.sh'])
                    private=Path(argv[-1]);request=json.loads((private/'request.json').read_text())
                    self.assertEqual(request['kwargs']['seed'],seed);self.assertEqual(request['kwargs']['llm_config']['seed'],seed)
                    name=Path(request['kwargs']['harness_path']).parent.name;evaluated.append(name)
                    solved=32 if perfect and name=='h2' else int(name[1:])
                    rows=[{'example_id':'redacted','reward':float(i<solved),'turn_count':1,
                        'metrics':{'assistant_tokens_used':1},'prompt':[],'harness_trace':[]} for i in range(32)]
                    merged=private/'merged';merged.mkdir();write_json(merged/'val.json',runner.summarize_outputs(rows,{'seed':seed,'fixture_only':True}));runner.write_trajectories(rows,merged)
                    write_json(private/'audit.json',{'status':'PASS','examples':32,'seed':seed,'plan_sha256':file_sha256(path),
                        'audit_source_sha256':file_sha256(Path('ours/audit_staged_evaluation.py')),
                        'harness_sha256':request['harness_sha256'],'solved':solved,'calls':32,'generated_tokens':32})
                    write_json(private/'result.json',{'status':'PASS','job_id':'123','plan_sha256':file_sha256(path),
                        'request_sha256':file_sha256(private/'request.json'),'artifact_sha256':tree_hashes(private),
                        'solved':solved,'new_model_calls':32,'generated_tokens':32});return '123\n'
                journal=SimpleNamespace(summary=lambda:{'unresolved_calls':0,'fixture_only':True})
                with patch('ours.staged_search.subprocess.check_output',command),patch('ours.staged_search.BudgetJournal',return_value=journal), \
                     patch('ours.scoped_proposer.relay',relay),patch.object(native,'propose_claude',proposer):
                    IndependentHarnessSearch(plan,path,submit=True).run()
                proof=audit_archive(path,plan)
                self.assertEqual(proof['accepted'],'h2' if perfect else 'h15');self.assertEqual(proof['completed_rounds'],1 if perfect else 5)
                self.assertEqual(len(evaluated),4 if perfect else 16);self.assertEqual(len(proposed),1 if perfect else 5)
                self.assertEqual(proof['seed'],seed);self.assertEqual(proof['condition'],'harness_only')
                print(json.dumps({'kind':'independent_harness_native_loop_fixture','seed':seed,'perfect':perfect,
                    'status':'PASS','accepted':proof['accepted'],'synthetic_scores_and_jobs':True,'new_model_calls':0,'new_api_calls':0}),flush=True)

    def test_frozen_contract_is_required_beyond_the_shared_plan(self):
        from ours.independent_harness_search import check_plan,ENTRYPOINT,CONTRACT,SOURCES
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'plan.json'
            plan={'seed':43,'coordinator_entrypoint':ENTRYPOINT,'metadata_contract':CONTRACT,'source_sha256':{n:file_sha256(Path(n)) for n in SOURCES}}
            write_json(path,plan)
            with patch('ours.staged_search.check_plan',side_effect=lambda p,**kw:json.loads(p.read_text())):
                self.assertEqual(check_plan(path),plan)
                for key,value in (('seed',42),('metadata_contract','old'),('coordinator_entrypoint','ours.staged_search'),('source_sha256',{})):
                    changed=deepcopy(plan);changed[key]=value;write_json(path,changed)
                    with self.assertRaisesRegex(ValueError,'compatibility contract'):check_plan(path)


if __name__=='__main__':unittest.main()
