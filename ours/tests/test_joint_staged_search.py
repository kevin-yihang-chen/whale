"""Joint routing, native selection equivalence and complete archived-shard replay."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.native_search import tree_hashes,write_json
from ours.visual_task import file_sha256


@unittest.skipUnless(importlib.util.find_spec('chess') and importlib.util.find_spec('transformers'), 'Requires native Chess runtime')
class JointStagedTests(unittest.TestCase):
    def test_joint_launcher_preserves_five_rounds_fst_rejections_and_perfect_stop(self):
        from autoharness_chess_puzzle import runner
        from meta_harness import meta_harness_chess_puzzle as native
        from ours.staged_search import StagedSearch
        from ours.joint_staged_search import JointStagedSearch,WRAPPER
        from ours.staged_candidate_recovery import compatible_proposal
        from ours.tests.test_prompt_subspace import BASELINE,replace_user,write_harness
        terminal='JobId=123 JobState=COMPLETED ExitCode=0:0 RunTime=00:02:00 AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h800=2\n'
        @contextmanager
        def relay(*args,**kwargs):yield {'base_url':'http://127.0.0.1:1','token':'offline-fixture'}
        for condition,perfect in (('whale',False),('whale_fst',False),('whale',True)):
            outcomes=[]
            for joint in (False,True):
                with self.subTest(condition=condition,perfect=perfect,joint=joint),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);trial=root/'trial';plan_path=root/'plan.json'
                    plan={'phase_root':str(trial),'condition':condition,'seed':42,'dataset':'fixture',
                        'target_config':{'model':'fixture'},'incoming_harness':str(BASELINE.resolve()),
                        'data_role':'mh_val','max_cli_turns':12,'max_api_requests':12,'iterations':5,'proposals_per_iter':3,
                        'limitations':['Synthetic CPU routing fixture; not certified joint experimental data.']}
                    write_json(plan_path,plan);proposed=[];evaluated=[];wrappers=[]
                    def proposer(**kw):
                        names=native.next_harness_names(kw['run_dir'],3);proposed.append(names)
                        self.assertTrue(all(name in kw['task_prompt'] for name in names))
                        for name in names:
                            source=replace_user(BASELINE.read_text(),repr(name+' {observation}'))
                            if condition=='whale_fst' and name=='h2':source=source.replace('MAX_TURNS = 9','MAX_TURNS = 18')
                            write_harness(kw['run_dir'],name,source)
                        write_json(kw['run_dir']/'pending_eval.json',[{('candidate' if kw['iteration']==3 else 'harness'):n} for n in names])
                        report=kw['run_dir']/f"logs/iteration_{kw['iteration']}/report.md"
                        report.parent.mkdir(parents=True,exist_ok=True);report.write_text('Synthetic commentary, not evidence.\n')
                        return SimpleNamespace(exit_code=0,duration_seconds=0.,show=lambda:None)
                    def command(argv,**kw):
                        if argv[0]=='squeue':return ''
                        if argv[0]=='scontrol':return terminal
                        self.assertEqual(argv[:2],['sbatch','--parsable']);wrappers.append(argv[2])
                        self.assertEqual(argv[2],WRAPPER if joint else 'ours/run_staged_evaluation.sh')
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
                            'solved':solved,'new_model_calls':32,'generated_tokens':32});return '123\n'
                    journal=SimpleNamespace(summary=lambda:{'unresolved_calls':0,'fixture_only':True})
                    with patch('ours.staged_search.subprocess.check_output',command), \
                         patch('ours.staged_search.BudgetJournal',return_value=journal), \
                         patch('ours.scoped_proposer.relay',relay),patch.object(native,'propose_claude',proposer), \
                         patch('ours.staged_search.isolated_native_proposal',compatible_proposal):
                        (JointStagedSearch if joint else StagedSearch)(plan,plan_path,submit=True).run()
                    result=json.loads((trial/'result.json').read_text())
                    self.assertEqual(result['status'],'COMPLETED_NATIVE_SEARCH');self.assertEqual(result['provenance']['condition'],condition)
                    from ours.search_completion import audit_archive
                    checked=audit_archive(plan_path,plan)
                    self.assertEqual(checked['status'],'PASS_COMPLETED_NATIVE_SEARCH')
                    self.assertEqual(checked['totals']['evaluations'],len(evaluated))
                    self.assertEqual(checked['completed_rounds'],1 if perfect else 5)
                    self.assertEqual(checked['new_model_calls'],0)
                    snapshot=(trial/'state.json').read_bytes()
                    state=json.loads(snapshot);state['status']='FAILED';write_json(trial/'state.json',state)
                    with self.assertRaisesRegex(ValueError,'not completed'):audit_archive(plan_path,plan)
                    (trial/'state.json').write_bytes(snapshot)
                    if not perfect:
                        last=trial/'search/logs/iteration_005/comparison.json';saved_last=last.read_bytes()
                        saved_result=(trial/'result.json').read_bytes();last.unlink()
                        last_request=trial/'proposal-request-5.json';saved_request=last_request.read_bytes();last_request.unlink()
                        truncated=json.loads(saved_result);truncated['search_sha256']=tree_hashes(trial/'search')
                        write_json(trial/'result.json',truncated)
                        with self.assertRaisesRegex(ValueError,'frozen stopping condition'):audit_archive(plan_path,plan)
                        last.write_bytes(saved_last);last_request.write_bytes(saved_request);(trial/'result.json').write_bytes(saved_result)
                    comparison=trial/'search/logs/iteration_001/comparison.json';saved_comparison=comparison.read_bytes()
                    changed=json.loads(saved_comparison);changed['frontier']['_best']['avg_success_rate']=0.999
                    write_json(comparison,changed);saved_result=(trial/'result.json').read_bytes()
                    changed_result=json.loads(saved_result);changed_result['search_sha256']=tree_hashes(trial/'search')
                    write_json(trial/'result.json',changed_result)
                    with self.assertRaisesRegex(ValueError,'frontier differs'):audit_archive(plan_path,plan)
                    comparison.write_bytes(saved_comparison);(trial/'result.json').write_bytes(saved_result)
                    history=[json.loads(p.read_text())['accepted_harness'] for p in sorted((trial/'search/logs').glob('iteration_*/comparison.json'))]
                    outcomes.append((history,proposed,evaluated))
            self.assertEqual(outcomes[0],outcomes[1]);self.assertEqual(len(outcomes[1][0]),1 if perfect else 5)
            self.assertEqual(len(outcomes[1][2]),4 if perfect else 15 if condition=='whale_fst' else 16)
            print(json.dumps({'kind':'joint_native_search_routing_fixture','condition':condition,'perfect_stop':perfect,
                'status':'PASS','accepted_history':outcomes[1][0],'new_api_calls':0,'scores_are_synthetic':True}),flush=True)

    def test_joint_worker_entrypoint_merges_and_replays_every_archived_reply(self):
        from ours.joint_staged_search import run_evaluation,ENTRYPOINT
        plan_path=Path('results/updated-mh-phase-plan-20260909-v2.json')
        plan=json.loads(plan_path.read_text());source=Path(plan['phase_root'])/'evaluations/h1'
        if not source.exists():self.skipTest('Archived complete MH shards are required')
        original_hashes=tree_hashes(source)
        with tempfile.TemporaryDirectory() as tmp:
            private=Path(tmp);original=json.loads((source/'request.json').read_text())
            write_json(private/'request.json',{'plan_sha256':original['plan_sha256'],'harness_sha256':original['harness_sha256'],
                'kwargs':{k:v for k,v in original.items() if k not in {'plan_sha256','harness_sha256'}}})
            launched=[]
            class Child:
                returncode=0
                def __init__(self,argv,**kwargs):
                    launched.append((argv,kwargs));replica=int(argv[-1])
                    shutil.copytree(source/f'replica-{replica}',private/f'replica-{replica}')
                def poll(self):return 0
                def wait(self,**kwargs):return 0
            with patch.dict(os.environ,SLURM_JOB_ID='123',CUDA_VISIBLE_DEVICES='4,7'), \
                 patch('ours.staged_evaluation.subprocess.Popen',Child):
                run_evaluation(SimpleNamespace(plan=plan_path,evaluation=private),plan)
            self.assertEqual([argv[1:3] for argv,_ in launched],[['-m',ENTRYPOINT],['-m',ENTRYPOINT]])
            self.assertEqual([kw['env']['CUDA_VISIBLE_DEVICES'] for _,kw in launched],['4','7'])
            result=json.loads((private/'result.json').read_text())
            self.assertEqual((result['solved'],result['new_model_calls'],result['generated_tokens']),(6,34,251088))
            self.assertEqual(tree_hashes(source),original_hashes)
            print(json.dumps({'kind':'joint_worker_routing_archived_replay_fixture','status':'PASS','archived_examples':32,
                'archived_calls':34,'archived_tokens':251088,'new_model_calls':0,'new_gpu_jobs':0,
                'archive_is_engineering_evidence_not_joint_result':True}),flush=True)

    def test_joint_handoff_lineage_and_worker_plan_reject_mixed_or_changed_evidence(self):
        from ours.joint_staged_search import certify_handoff,check_plan,SOURCES,WRAPPER,ENTRYPOINT
        base=json.loads(Path('results/controlled-harness-only-seed42-plan-20260910-v1.json').read_text())
        with tempfile.TemporaryDirectory(prefix='whale-joint-plan-fixture-') as tmp:
            root=Path(tmp);training=root/'training.json';completed=root/'completed.json';export_path=root/'export-plan.json';result_path=root/'export-result.json'
            selected={'step':1,'directory':str(root/'global_step_1'),'artifact_sha256':{}}
            train={'kind':'controlled_joint_phase1_plan','condition':'whale','seed':42,'initial_manifest':base['target_manifest']}
            write_json(training,train)
            done={'kind':'controlled_joint_phase1_training_result','status':'COMPLETED_NATIVE_PHASE','original_preflight_batch_ids_match':True,
                'phase':1,'plan_sha256':file_sha256(training),'job_id':'synthetic-training','condition':'whale','seed':42,
                'checkpoints':[selected],'optimizer_steps':0}
            write_json(completed,done)
            export={'training_plan':str(training),'training_result':str(completed),'source_job_id':'synthetic-training',
                'condition':'whale','seed':42,'resume_checkpoint':selected,'target':base['target_manifest']['path'],
                'base':base['target_manifest'],'source_sha256':{str(p):file_sha256(p) for p in (training,completed)}}
            write_json(export_path,export)
            result={'kind':'controlled_joint_phase1_export_result','status':'COMPLETED_CANONICAL_EXPORT',
                'slurm_terminal_path':str(root/'synthetic-slurm.txt'),'source_job_id':'synthetic-training',
                'job_id':'synthetic-export','condition':'whale','seed':42,'resume_checkpoint':selected,
                'exported':base['target_manifest'],'artifact_sha256':{str(export_path):file_sha256(export_path)}}
            write_json(result_path,result)
            with patch('ours.joint_staged_search.close_export',return_value=result):
                handoff,sources=certify_handoff(export_path,result_path)
                self.assertEqual(handoff['optimizer_steps'],0)
                changed=deepcopy(done);changed['condition']='whale_fst';write_json(completed,changed)
                with self.assertRaisesRegex(ValueError,'lineage'):certify_handoff(export_path,result_path)
                write_json(completed,done)
            plan=deepcopy(base);plan.update(kind='controlled_joint_staged_mh_plan',condition='whale',joint_phase=1,
                phase_root=str(root/'trial'),joint_handoff=handoff,joint_source_sha256=sources,
                evaluation_entrypoint=ENTRYPOINT,evaluation_wrapper=WRAPPER,
                metadata_contract='bare_list_or_candidates_and_agreeing_name_slot_harness_id_candidate_v2')
            plan['source_sha256'].update({n:file_sha256(Path(n)) for n in SOURCES})
            path=root/'joint-plan.json';write_json(path,plan)
            self.assertEqual(check_plan(path,verify_weights=False)['condition'],'whale')
            for field,value in (('condition','whale_fst'),('seed',43),('evaluation_entrypoint','ours.staged_search'),
                ('iterations',1),('server_arguments',[]),('joint_source_sha256',{}),('target_config',base['target_config']|{'max_tokens':512})):
                changed=deepcopy(plan);changed[field]=value;write_json(path,changed)
                with self.subTest(field=field),self.assertRaises(ValueError):check_plan(path,verify_weights=False)
            write_json(path,plan);training.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Changed completed joint evidence'):check_plan(path,verify_weights=False)
        print(json.dumps({'kind':'joint_provenance_schema_fixture','status':'PASS','handoff_is_synthetic':True,
            'worker_metadata_check_executed':True,'full_weight_or_terminal_certification_is_stubbed':True,'new_model_calls':0}),flush=True)


if __name__=='__main__':unittest.main()
