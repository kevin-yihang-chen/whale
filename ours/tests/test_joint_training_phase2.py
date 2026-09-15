"""Native second-phase configuration/data replay; no joint GPU experiment."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from ours.joint_training_phase2 import certify_search,probe_resume,resolve
from ours.native_resume_observation import resume_plan,state_digest
from ours.visual_task import file_sha256


class ResumeInputTests(unittest.TestCase):
    def test_weight_only_or_harness_only_cannot_supply_a_joint_handoff(self):
        for name in ('results/controlled-harness-only-seed42-plan-20260910-v1.json',
                     'results/controlled-weight-only-seed42-plan-20260910-v2.json'):
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'independent joint search'):
                certify_search(Path(name),Path('/nonexistent-joint-proof.json'))

    def test_actual_restore_hook_binds_its_source_and_selected_harness(self):
        from ours import native_resume_observation as observation
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'global_step_1';(directory/'actor/huggingface').mkdir(parents=True)
            for n in ('model_world_size_1_rank_0.pt','extra_state_world_size_1_rank_0.pt'):(directory/'actor'/n).write_bytes(b'CPU metadata fixture only')
            (directory/'data.pt').write_bytes(b'fixture');(directory/'actor/fsdp_config.json').write_text('{"world_size":1}')
            (directory/'actor/huggingface/config.json').write_text('{}');harness=root/'harness.py';harness.write_text('fixture')
            plan={'kind':'controlled_joint_phase2_plan','condition':'whale','phase':2,'native_batch_steps':1,'fresh_trajectories':64,
                'harness':str(harness),'harness_sha256':file_sha256(harness),
                'source_sha256':{'ours/native_resume_observation.py':file_sha256(observation.SOURCE)},
                'resume_checkpoint':{'step':1,'directory':str(directory)}}
            path=root/'plan.json';path.write_text(json.dumps(plan))
            with patch.dict(os.environ,WHALE_JOINT_RESUME_PLAN=str(path)):
                self.assertEqual(resume_plan(),(path,plan,directory))
                harness.write_text('different fixture')
                with self.assertRaisesRegex(ValueError,'selected harness'):resume_plan()
                harness.write_text('fixture')
                changed=deepcopy(plan);changed['source_sha256']['ours/native_resume_observation.py']='wrong';path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'observation source'):resume_plan()


@unittest.skipUnless(importlib.util.find_spec('torch'),'Requires the native Torch runtime')
class NativeSecondPhaseTests(unittest.TestCase):
    def test_resolved_native_resume_and_selected_harness_keep_the_actual_second_batch(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from ours.controlled_data_order import native_data_order
        from ours.native_phase_resume import check_configuration
        plan=deepcopy(json.loads(Path('results/controlled-whale-seed42-phase1-plan-20260910-v1.json').read_text()))
        # Archived weight-only data is a loader fixture, never a joint handoff.
        directory=Path('data/native-rsft/controlled-weight-only-seed42-222801/global_step_1').resolve()
        plan['resume_checkpoint']={'step':1,'directory':str(directory)}
        plan['harness']=str(Path('data/controlled-harness-only-seed42-20260910/search/harnesses/h1/harness.py').resolve())
        plan['kind']='engineering_native_resume_configuration_fixture'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'never-a-formal-plan.json'
            raw,config=resolve(plan,'phase2-native-cpu-fixture',path)
            cfg=OmegaConf.create(config);check_configuration(cfg,directory)
            self.assertEqual(cfg.trainer.total_training_steps,2)
            self.assertEqual(cfg.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook,'ours.native_resume_observation.prepare_worker')
            self.assertEqual(cfg.ray_kwargs.ray_init.runtime_env.env_vars.WHALE_JOINT_RESUME_PLAN,str(path))
            self.assertEqual(cfg.ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_HARNESS_PATH,plan['harness'])
            self.assertEqual(list(cfg.actor_rollout_ref.actor.checkpoint.load_contents),['model','extra'])
            digest=lambda:state_digest((random.getstate(),np.random.get_state(),torch.get_rng_state()))
            before=digest();order=native_data_order(config);self.assertEqual(digest(),before)
            self.assertEqual(order,plan['native_dataset_order'])
            before=digest();result=probe_resume(config,plan);self.assertEqual(digest(),before)
            self.assertEqual(result['next_batch_ids'],order['planned_batch_ids'][1])
            self.assertFalse(result['actor_rpc_executed']);self.assertEqual(result['new_model_calls'],0)
            self.assertFalse(path.exists())
        print(json.dumps({'kind':'joint_phase2_native_configuration_fixture','status':'PASS',
            'actual_hydra_and_task_loader':True,'native_model_extra_resume_configured':True,
            'selected_harness_preserved':True,'next_batch_ids':result['next_batch_ids'],
            'new_model_calls':0,'joint_checkpoint_certified':False,'gpu_restore_executed':False}),flush=True)


if __name__=='__main__':unittest.main()
