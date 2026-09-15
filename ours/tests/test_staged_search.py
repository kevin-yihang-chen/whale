"""Native five-round equivalence and fail-closed boundaries; no paid calls."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.native_search import tree_hashes, write_json
from ours.staged_search import StagedSearch, native_arguments, terminal_job, verify_evaluation
from ours.visual_task import file_sha256

TERMINAL = 'JobId=123 JobState=COMPLETED ExitCode=0:0 RunTime=00:02:00 AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h800=2\n'


class TerminalTests(unittest.TestCase):
    def test_running_failed_wrong_identity_and_wrong_allocation_are_not_completed_evaluations(self):
        self.assertEqual(terminal_job(TERMINAL,'123')['gpu_hours'],4/60)
        self.assertIsNone(terminal_job(TERMINAL.replace('COMPLETED','RUNNING'),'123'))
        for raw in (TERMINAL.replace('COMPLETED','TIMEOUT'),TERMINAL.replace('0:0','1:0'),
                    TERMINAL.replace('JobId=123','JobId=456'),TERMINAL.replace('gres/gpu=2','gres/gpu=1')):
            with self.subTest(raw=raw),self.assertRaises(ValueError):terminal_job(raw,'123')

    def test_submission_slot_wait_does_not_submit_or_retry_an_occupied_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json';path.write_text('{}')
            controller=StagedSearch({'phase_root':tmp},path,submit=True)
            with patch('ours.staged_search.subprocess.check_output',side_effect=['222801\n','']) as command, \
                 patch('ours.staged_search.time.sleep') as sleep:
                controller.wait_for_submission_slot('h0',Path(tmp)/'evaluations/h0')
            self.assertEqual(command.call_count,2)
            self.assertTrue(all(c.args[0][0]=='squeue' for c in command.call_args_list))
            sleep.assert_called_once_with(30)
            self.assertEqual(json.loads((Path(tmp)/'state.json').read_text())['existing_jobs'],['222801'])


@unittest.skipUnless(importlib.util.find_spec('chess'),'Requires native Chess runtime')
class StagedNativeTests(unittest.TestCase):
    def test_five_rounds_perfect_stop_and_fst_rejection_match_the_original_loop(self):
        from autoharness_chess_puzzle import runner
        from ours.controlled_conditions import native_search_context
        from ours.tests.test_prompt_subspace import BASELINE,replace_user,write_harness
        for condition,perfect,reject in [('harness_only',False,False),('whale',False,False),
                                        ('whale_fst',False,True),('harness_only',True,False)]:
            with self.subTest(condition=condition,perfect=perfect),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                plans=[]; histories=[]; called=[]; allocations=[]
                for staged in (False,True):
                    directory=root/('staged' if staged else 'direct')
                    plan={'phase_root':str(directory),'condition':condition,'seed':44,'dataset':'fixture',
                        'target_config':{'model':'fixture'},'incoming_harness':str(BASELINE.resolve()),
                        'limitations':['Synthetic CPU fixture, zero real model or API calls']}
                    plan_path=root/f'plan-{staged}.json';write_json(plan_path,plan);plans.append(plan)
                    evaluated=[];slots=[]
                    def proposer(**kwargs):
                        slots.append(kwargs['next_names'])
                        for name in kwargs['next_names']:
                            source=replace_user(BASELINE.read_text(),repr(name+' {observation}'))
                            if reject and name=='h2':source=source.replace('MAX_TURNS = 9','MAX_TURNS = 18')
                            write_harness(kwargs['run_dir'],name,source)
                        write_json(kwargs['run_dir']/'pending_eval.json',{'candidates':[{'name':n} for n in kwargs['next_names']]})
                        return SimpleNamespace(show=lambda:None)
                    def outputs(kwargs,destination):
                        name=Path(kwargs['harness_path']).parent.name;evaluated.append(name)
                        successes=32 if perfect and name=='h2' else int(name[1:])
                        rows=[{'example_id':'redacted','reward':float(i<successes),'turn_count':1,
                            'metrics':{'assistant_tokens_used':1},'prompt':[],'harness_trace':[]} for i in range(32)]
                        summary=runner.summarize_outputs(rows,{'seed':44,'fixture_only':True})
                        destination.mkdir(parents=True,exist_ok=True)
                        write_json(destination/'val.json',summary);runner.write_trajectories(rows,destination)
                        return summary
                    if not staged:
                        directory.mkdir()
                        with native_search_context(condition,directory,BASELINE) as (native,benchmark,_), \
                             patch.object(native,'propose_claude_with_retries',proposer), \
                             patch.object(benchmark,'evaluate_harness',lambda **kw:outputs(kw,Path(kw['output_dir']))):
                            native.run_evolve(native_arguments(plan,directory))
                    else:
                        def command(argv,**kwargs):
                            if argv[0]=='squeue':return ''
                            if argv[0]=='scontrol':return TERMINAL
                            self.assertEqual(argv[:3],['sbatch','--parsable','ours/run_staged_evaluation.sh'])
                            private=Path(argv[-1]);request=json.loads((private/'request.json').read_text())
                            summary=outputs(request['kwargs'],private/'merged')
                            audit={'status':'PASS','examples':32,'seed':44,'plan_sha256':file_sha256(plan_path),
                                'audit_source_sha256':file_sha256(Path('ours/audit_staged_evaluation.py')),
                                'harness_sha256':request['harness_sha256'],'solved':summary['solved_examples'],
                                'calls':32,'generated_tokens':32}
                            write_json(private/'audit.json',audit)
                            write_json(private/'result.json',{'status':'PASS','job_id':'123','plan_sha256':file_sha256(plan_path),
                                'request_sha256':file_sha256(private/'request.json'),'artifact_sha256':tree_hashes(private),
                                'solved':summary['solved_examples'],'new_model_calls':32,'generated_tokens':32})
                            return '123\n'
                        fake_journal=SimpleNamespace(summary=lambda:{'fixture_only':True})
                        with patch('ours.staged_search.subprocess.check_output',command), \
                             patch('ours.staged_search.BudgetJournal',return_value=fake_journal), \
                             patch('ours.staged_search.isolated_native_proposal',lambda native,plan,out,journal,**kw:proposer(**kw)):
                            StagedSearch(plan,plan_path,submit=True).run()
                        receipt=json.loads((directory/'result.json').read_text())
                        self.assertEqual(receipt['status'],'COMPLETED_NATIVE_SEARCH')
                        with self.assertRaises(FileExistsError):StagedSearch(plan,plan_path).run()
                        value=receipt['evaluations'][0];private=Path(value['directory'])
                        (private/'merged/val.json').write_text('{}')
                        with self.assertRaisesRegex(ValueError,'Changed evaluation evidence'):
                            verify_evaluation(private,plan,plan_path)
                    histories.append([json.loads(p.read_text())['accepted_harness'] for p in
                                      sorted((directory/'search/logs').glob('iteration_*/comparison.json'))])
                    called.append(evaluated);allocations.append(slots)
                self.assertEqual(histories[0],histories[1]);self.assertEqual(called[0],called[1]);self.assertEqual(allocations[0],allocations[1])
                self.assertEqual(len(histories[1]),1 if perfect else 5)
                self.assertEqual(len(called[1]),4 if perfect else 15 if reject else 16)
                print(json.dumps({'kind':'staged_native_search_fixture','condition':condition,'perfect_stop':perfect,
                    'fst_rejection':reject,'accepted_history':histories[1],'evaluations':len(called[1]),
                    'status':'PASS','new_api_calls':0,'scores_are_synthetic':True}),flush=True)

    def test_failed_allocation_cannot_trigger_a_paid_proposal(self):
        from ours.tests.test_prompt_subspace import BASELINE
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan_path=root/'plan.json'
            plan={'phase_root':str(root/'trial'),'condition':'harness_only','seed':42,'dataset':'fixture',
                  'target_config':{'model':'fixture'},'incoming_harness':str(BASELINE.resolve())}
            write_json(plan_path,plan)
            def command(argv,**kwargs):
                if argv[0]=='squeue':return ''
                return '123\n' if argv[0]=='sbatch' else TERMINAL.replace('COMPLETED','FAILED')
            with patch('ours.staged_search.subprocess.check_output',command), \
                 patch('ours.staged_search.BudgetJournal',return_value=SimpleNamespace(summary=lambda:{})), \
                 patch('ours.staged_search.isolated_native_proposal') as paid:
                with self.assertRaisesRegex(ValueError,'Native evaluation failed'):
                    StagedSearch(plan,plan_path,submit=True).run()
                paid.assert_not_called()
            self.assertEqual(json.loads((root/'trial/state.json').read_text())['status'],'FAILED')
            self.assertTrue(list((root/'trial/search/logs').glob('*/h0/*/error.txt')))


@unittest.skipUnless(importlib.util.find_spec('chess') and importlib.util.find_spec('transformers'),
                     'Requires native Chess runtime and archived MH evidence')
class RecordedAuditTests(unittest.TestCase):
    def test_all_actual_archived_replies_replay_without_new_inference(self):
        from ours.audit_staged_evaluation import audit_evaluation
        plan_path=Path('results/updated-mh-phase-plan-20260909-v2.json')
        plan=json.loads(plan_path.read_text());root=Path(plan['phase_root'])
        for name,successes,calls,tokens in [('h0',0,35,260128),('h1',6,34,251088)]:
            with self.subTest(harness=name),tempfile.TemporaryDirectory() as tmp:
                private=Path(tmp)/name;shutil.copytree(root/f'evaluations/{name}',private)
                original=json.loads((private/'request.json').read_text())
                write_json(private/'request.json',{'plan_sha256':original['plan_sha256'],
                    'harness_sha256':original['harness_sha256'],
                    'kwargs':{k:v for k,v in original.items() if k not in {'plan_sha256','harness_sha256'}}})
                source,=(root/'search/logs').glob(f'*/{name}/*/val.json')
                shutil.copytree(source.parent,private/'merged')
                result=audit_evaluation(plan,plan_path,private)
                self.assertEqual((result['solved'],result['calls'],result['generated_tokens']),(successes,calls,tokens))
                self.assertEqual(result['new_model_calls'],0)
                altered=dict(plan,seed=43)
                with self.assertRaisesRegex(ValueError,'Wrong shard coverage'):
                    audit_evaluation(altered,plan_path,private)
                print(json.dumps({k:v for k,v in result.items() if k not in {'records','limitations'}}),flush=True)

    def test_new_merge_and_audit_boundary_on_complete_original_shards(self):
        from ours.staged_evaluation import run_evaluation
        plan_path=Path('results/updated-mh-phase-plan-20260909-v2.json')
        plan=json.loads(plan_path.read_text());source=Path(plan['phase_root'])/'evaluations/h1'
        with tempfile.TemporaryDirectory() as tmp:
            private=Path(tmp)
            original=json.loads((source/'request.json').read_text())
            write_json(private/'request.json',{'plan_sha256':original['plan_sha256'],'harness_sha256':original['harness_sha256'],
                'kwargs':{k:v for k,v in original.items() if k not in {'plan_sha256','harness_sha256'}}})
            class Child:
                returncode=0
                def __init__(self,argv,**kwargs):
                    replica=int(argv[-1])
                    shutil.copytree(source/f'replica-{replica}',private/f'replica-{replica}')
                def poll(self):return 0
                def wait(self,**kwargs):return 0
            with patch.dict(os.environ,SLURM_JOB_ID='123',CUDA_VISIBLE_DEVICES='0,1'), \
                 patch('ours.staged_evaluation.subprocess.Popen',Child):
                run_evaluation(SimpleNamespace(plan=plan_path,evaluation=private),plan)
            result=json.loads((private/'result.json').read_text())
            self.assertEqual((result['status'],result['solved'],result['new_model_calls'],result['generated_tokens']),
                             ('PASS',6,34,251088))
            self.assertEqual(json.loads((private/'audit.json').read_text())['new_model_calls'],0)
            for name,digest in result['artifact_sha256'].items():self.assertEqual(file_sha256(private/name),digest)
            with patch.dict(os.environ,SLURM_JOB_ID='123',CUDA_VISIBLE_DEVICES='0,1'), \
                 self.assertRaisesRegex(ValueError,'Preserve previous evaluation attempt'):
                run_evaluation(SimpleNamespace(plan=plan_path,evaluation=private),plan)
            print(json.dumps({'kind':'staged_merge_real_archive_fixture','status':'PASS','examples':32,
                'archived_calls':34,'archived_tokens':251088,'new_model_calls':0,'new_gpu_jobs':0}),flush=True)


if __name__=='__main__':unittest.main()
