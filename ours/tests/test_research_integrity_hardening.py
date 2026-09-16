"""CPU adversarial fixtures for E1/E2/E3 evidence and E4 checkpoint boundaries."""
import ast
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ours import fast_chart_search as search
from ours.fast_chart_endpoint_evaluation import endpoint_identity
from ours.evidence import EvaluationIdentity, fingerprint
from ours.local_completion import checkpoint_tensor, checkpoint_manifest
from ours.native_visual_service import embedding_coordinates
from ours.visual_task import file_sha256


def write(path, value):
    Path(path).write_text(json.dumps(value))


def failed_fixture(root, state='FAILED'):
    harness=root/'search/harnesses/h1/harness.py'
    harness.parent.mkdir(parents=True);harness.write_text('# candidate h1\n')
    output=root/'evaluation';output.mkdir()
    write(output/'failure.json', {'status':'INCOMPLETE','error_type':'RuntimeError','error':'fixture failure'})
    identity=asdict(EvaluationIdentity(*(fingerprint(k) for k in ('w','H','C','d','v')),phase='fixture'))
    phase={'identity':identity,'seed':42}
    plan=root/'candidate.json'
    write(plan, {'kind':'compact_chart_search_evaluation','seed':42,'candidate':'h1','identity':identity,
        'output':str(output),'harness_sha256':file_sha256(harness),
        'config':{'data':{'visual_harness_path':str(harness)}},'bounds':{'gpus':1,'time_limit_seconds':3600}})
    allocation_dir=root/'allocation';allocation_dir.mkdir()
    terminal=allocation_dir/'slurm-terminal.txt'
    terminal.write_text(f'JobId=123 JobState={state} RunTime=00:30:00 AllocTRES=cpu=12,gres/gpu=1')
    allocation=allocation_dir/'allocation-result.json'
    write(allocation, {'job_id':'123','state':state,'seconds':1800,'gpu_hours':'0.5','terminal_sha256':file_sha256(terminal)})
    write(allocation_dir/'submission.json', {'plan':str(plan),'plan_sha256':file_sha256(plan),
        'job_id':'123','gpus':1,'time_limit_seconds':3600})
    return phase,plan,allocation


class RuntimeValidationTests(unittest.TestCase):
    def test_critical_modules_do_not_use_removable_asserts(self):
        for name in ('visual_search_bridge','visual_search_evaluation','verify_visual_resumed_parameters',
                     'observe_research_allocation','fast_chart_search','fast_chart_search_report',
                     'fast_chart_endpoint_evaluation','research_budget'):
            tree=ast.parse((search.ROOT/f'ours/{name}.py').read_text())
            self.assertFalse(any(isinstance(node,ast.Assert) for node in ast.walk(tree)),name)

    def test_runtime_rejections_survive_optimized_interpreter(self):
        code='''
import tempfile
from pathlib import Path
import torch
from ours.visual_search_bridge import once
from ours.verify_visual_resumed_parameters import compare_states
def rejects(call):
    try: call()
    except ValueError: return
    raise SystemExit('Invalid input was accepted')
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'receipt.json'
    once(path,{'a':1})
    rejects(lambda:once(path,{'a':2}))
base={'weight':torch.tensor([1.])}
rejects(lambda:compare_states(base,{**base,'extra':torch.tensor([1.])},{},1))
rejects(lambda:compare_states(base,{'weight':torch.tensor([2.],dtype=torch.float64)},{},1))
rejects(lambda:compare_states(base,base,{},2))
print('REJECTIONS_PASSED')
'''
        for flags in ([],['-O']):
            with self.subTest(flags=flags):
                result=subprocess.run([sys.executable,'-B',*flags,'-c',code],cwd=search.ROOT,
                    capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('REJECTIONS_PASSED',result.stdout)


class FailedCandidateTests(unittest.TestCase):
    def test_report_reconstruction_rejects_mismatched_failed_submission(self):
        from ours import fast_chart_search_report as report
        with TemporaryDirectory() as folder:
            root=Path(folder);phase,plan,allocation=failed_fixture(root)
            with patch.object(search,'load',return_value=phase):search.failed_evaluation(root,plan,allocation)
            base=root/'baseline';base.mkdir()
            for sub in ('H','C'):(base/sub).mkdir()
            for sub in ('','H','C'):write(base/sub/'result.json',{})
            phase['baseline']={'plan':str(plan),'allocation':str(allocation),
                'plan_sha256':file_sha256(plan),'allocation_sha256':file_sha256(allocation)}
            write(root/'result.json',{'status':'COMPLETE_COMPACT_SEARCH_AND_SELECTION','schema_version':2})
            submission=allocation.parent/'submission.json';value=search.read(submission);value['job_id']='999';write(submission,value)
            with patch.object(report,'load',return_value=phase),patch.object(report,'certify',return_value={'plan':{'output':str(base)}}):
                with self.assertRaisesRegex(ValueError,'submitted plan and job'):report.reconstruct(root)

    def test_failed_registration_and_reconstruction_share_full_bindings(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);phase,plan,allocation=failed_fixture(root)
            with patch.object(search,'load',return_value=phase):
                search.failed_evaluation(root,plan,allocation)
            record=search.verified_failure(root,'h1',phase)
            self.assertEqual(record['execution']['job_id'],'123')
            self.assertEqual(record['execution']['gpu_hours'],'0.5')
            self.assertFalse(record['scored_as_zero'])
            submission=allocation.parent/'submission.json'
            value=search.read(submission);value['job_id']='999';write(submission,value)
            with self.assertRaisesRegex(ValueError,'submitted plan and job'):
                search.verified_failure(root,'h1',phase)

    def test_active_success_and_unknown_states_are_not_failed_slots(self):
        for state in ('RUNNING','PENDING','COMPLETING','COMPLETED','MYSTERY'):
            with self.subTest(state=state),TemporaryDirectory() as folder:
                root=Path(folder);phase,plan,allocation=failed_fixture(root,state)
                with patch.object(search,'load',return_value=phase),self.assertRaises(ValueError):
                    search.failed_evaluation(root,plan,allocation)
                self.assertFalse((root/'failure-h1.json').exists())

    def test_mismatched_plan_job_time_cost_gpu_or_code_fails_before_registration(self):
        changes=[('submission','plan_sha256',fingerprint('wrong')),
            ('submission','plan','/tmp/unrelated-plan.json'),('submission','job_id','999'),
            ('submission','gpus',2),('submission','time_limit_seconds',1),
            ('allocation','terminal_sha256',fingerprint('wrong')),('allocation','job_id','999'),
            ('allocation','state','TIMEOUT'),('allocation','seconds',1),
            ('allocation','gpu_hours','0.4'),('allocation','gpu_hours','NaN'),
            ('plan','identity',{}),('plan','seed',43),('plan','harness_sha256',fingerprint('other'))]
        for target,key,value in changes:
            with self.subTest(target=target,key=key),TemporaryDirectory() as folder:
                root=Path(folder);phase,plan,allocation=failed_fixture(root)
                path={'submission':allocation.parent/'submission.json','allocation':allocation,'plan':plan}[target]
                data=search.read(path);data[key]=value;write(path,data)
                with patch.object(search,'load',return_value=phase),self.assertRaises(ValueError):
                    search.failed_evaluation(root,plan,allocation)
                self.assertFalse((root/'failure-h1.json').exists())

    def test_complete_slurm_record_cannot_be_relabelled_failed(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);phase,plan,allocation=failed_fixture(root,'COMPLETED')
            value=search.read(allocation);value['state']='FAILED';write(allocation,value)
            with patch.object(search,'load',return_value=phase),self.assertRaisesRegex(ValueError,'failed terminal'):
                search.failed_evaluation(root,plan,allocation)

    def test_slots_bind_absent_generated_and_failed_candidates_before_selection(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);phase,plan,allocation=failed_fixture(root)
            with patch.object(search,'load',return_value=phase):search.failed_evaluation(root,plan,allocation)
            comparison=root/'search/logs/iteration_001/comparison.json';comparison.parent.mkdir(parents=True)
            write(comparison,{})
            # h2 was generated but never evaluated; h3 was not generated.
            h2=root/'search/harnesses/h2/harness.py';h2.parent.mkdir(parents=True);h2.write_text('# h2')
            manifest=search.candidate_slots(root,[],phase,finalize=True)
            self.assertEqual(manifest['counts'],{'candidate_budget':3,'generated_candidates':2,
                'evaluation_attempts':1,'successful_evaluations':0,'failed_slots':3})
            self.assertEqual(manifest,search.candidate_slots(root,[],phase))
            self.assertFalse((root/'selection-veto.json').exists())
            h2.write_text('# changed after failure')
            with self.assertRaisesRegex(ValueError,'candidate code'):
                search.candidate_slots(root,[],phase)


class CheckpointShardTests(unittest.TestCase):
    def test_sharded_tensor_and_coordinates_equal_single_file(self):
        import torch
        from safetensors.torch import save_file
        name='model.language_model.embed_tokens.weight'
        with TemporaryDirectory() as folder:
            root=Path(folder);single=root/'single';split=root/'split';single.mkdir();split.mkdir()
            tensor=torch.arange(9000*4,dtype=torch.float32).reshape(9000,4)
            save_file({name:tensor,'other':torch.zeros(1)},str(single/'model.safetensors'))
            save_file({name:tensor},str(split/'model-00001-of-00002.safetensors'))
            save_file({'other':torch.zeros(1)},str(split/'model-00002-of-00002.safetensors'))
            write(split/'config.json',{})
            index={'weight_map':{name:'model-00001-of-00002.safetensors','other':'model-00002-of-00002.safetensors'}}
            write(split/'model.safetensors.index.json',index)
            self.assertEqual(len(checkpoint_manifest(split)['weights']),2)
            torch.testing.assert_close(checkpoint_tensor(split,name),tensor)
            self.assertEqual(embedding_coordinates(single),embedding_coordinates(split))
            index['weight_map'][name]='model-00002-of-00002.safetensors';write(split/'model.safetensors.index.json',index)
            with self.assertRaisesRegex(ValueError,'index'):checkpoint_tensor(split,name)

    def test_missing_and_duplicate_tensor_fail(self):
        import torch
        from safetensors.torch import save_file
        with TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'found 0'):checkpoint_tensor(root,'embedding')
            for name in ('one','two'):save_file({'embedding':torch.ones(2,2)},str(root/f'{name}.safetensors'))
            with self.assertRaisesRegex(ValueError,'found 2'):checkpoint_tensor(root,'embedding')


class EndpointIdentityTests(unittest.TestCase):
    def test_new_endpoint_plan_freezes_new_sources_without_rewriting_reference(self):
        from ours import fast_chart_endpoint_evaluation as endpoint
        from ours import fast_chart_inference_policy as policy
        for partition in ('T','R','chartqa'):
            with self.subTest(partition=partition),TemporaryDirectory() as folder:
                root=Path(folder);harness=root/'h0.py';harness.write_text('# fixed h0')
                configuration=root/'configuration.json';write(configuration,{'harness':str(harness)})
                reference=root/'reference.json'
                original=self.plan('T')
                original.update(config={'data':{'visual_harness_path':str(harness)},'trainer':{},'reward':{},
                    'actor_rollout_ref':{'rollout':{'trace':{}}}},source_sha256={'ours/native_visual_service.py':'old-runtime'},
                    model={'path':str(root/'weights'),'weights_sha256':fingerprint('weights'),'assets':{}})
                write(reference,original);reference_sha=file_sha256(reference)
                registration=root/'registration.json'
                write(registration,{'kind':'compact_endpoint_registration','condition':'veto','seed':42,
                    'reference':str(reference),'configuration':str(configuration),'evidence_sha256':{str(reference):reference_sha}})
                if partition=='chartqa':
                    data=root/'chartqa512-v1';data.mkdir();write(data/'plan.json',{})
                    write(data/'result.json',{'status':'READY_FIXED_EXTERNAL_SUBSET'})
                else:
                    data=root/f'dataset/{partition}';data.mkdir(parents=True);write(data/'manifest.json',{})
                frozen_policy=root/'policy.json';write(frozen_policy,{'h0_sha256':file_sha256(harness)})
                with patch.object(endpoint,'OUTPUT',root),patch.object(endpoint,'storage_check'),\
                     patch.object(endpoint,'decode_identity',return_value=fingerprint('decode')),\
                     patch.object(endpoint,'checkpoint_manifest',return_value=original['model']),\
                     patch.object(endpoint,'pair_inputs',return_value=({'role':'V' if partition=='R' else 'T',
                         'audit_data_sha256':fingerprint('new-pairs')},[object()],[])),\
                     patch.object(policy,'POLICY',frozen_policy),patch.object(policy,'load',return_value=search.read(frozen_policy)),\
                     patch.object(policy,'bind',return_value=None):
                    path=root/'new-plan.json'
                    plan=endpoint.prepare(path,root/'output',registration,partition=partition)
                    self.assertEqual(endpoint.check(path),plan)
                    self.assertEqual(plan['source_sha256']['ours/native_visual_service.py'],file_sha256(search.ROOT/'ours/native_visual_service.py'))
                    self.assertEqual(file_sha256(reference),reference_sha)
                    self.assertEqual(plan['phase'],'registered-endpoint-'+partition)
                    plan['identity']={'stale':True};write(path,plan)
                    with self.assertRaisesRegex(ValueError,'Endpoint identity'):endpoint.check(path)
                    with self.assertRaisesRegex(ValueError,'Endpoint identity'):asyncio.run(endpoint.run(plan,path))

    def plan(self,partition):
        return {'partition':partition,'role':{'T':'T','R':'V','chartqa':'external'}[partition],
            'model':{'weights_sha256':fingerprint('weights')},'manifest_sha256':fingerprint('manifest'),
            'audit_data_sha256':fingerprint('pairs'),'decode_sha256':fingerprint('decode'),
            'phase':'stale-V-phase','identity':{'stale':True},'verifier_sha256':fingerprint('old')}

    def test_paired_identity_is_same_at_preparation_validation_and_execution(self):
        for partition in ('T','R'):
            plan=self.plan(partition);identity=endpoint_identity(plan,bind=True)
            self.assertEqual(identity,endpoint_identity(plan))
            self.assertEqual(EvaluationIdentity(**identity).phase,'registered-endpoint-'+partition)
            for key in ('phase','manifest_sha256','audit_data_sha256','decode_sha256','verifier_sha256'):
                changed=deepcopy(plan);changed[key]=fingerprint('changed')
                with self.subTest(partition=partition,key=key),self.assertRaises(ValueError):endpoint_identity(changed)
            plan['model']['weights_sha256']=fingerprint('different-weights')
            with self.assertRaises(ValueError):endpoint_identity(plan)

    def test_chartqa_does_not_inherit_paired_identity_or_change_scoring(self):
        from ours.chartqa_open_answer import relaxed_correct,PROTOCOL
        plan=self.plan('chartqa');identity=endpoint_identity(plan,bind=True)
        self.assertNotIn('audit_data_sha256',plan)
        self.assertNotIn('audit_data_sha256',identity)
        self.assertEqual(identity['protocol'],PROTOCOL)
        self.assertEqual(identity,endpoint_identity(plan))
        self.assertNotEqual(identity['verifier_sha256'],endpoint_identity(self.plan('T'),bind=True)['verifier_sha256'])
        self.assertFalse(relaxed_correct('1,000','1000'))
        self.assertFalse(relaxed_correct('1000','1,000'))
        plan['audit_data_sha256']=fingerprint('stale-pairs')
        with self.assertRaisesRegex(ValueError,'inapplicable'):endpoint_identity(plan)


if __name__=='__main__':unittest.main()
