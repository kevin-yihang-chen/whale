"""Native phase boundaries and first-stage plan scope, with synthetic rollouts."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.joint_training_phase1 import prepare


class JointPhaseScopeTests(unittest.TestCase):
    def test_non_joint_conditions_and_existing_plans_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'plan.json'
            for condition,seed in [('weight_only',42),('harness_only',42),('whale',45)]:
                with self.subTest(condition=condition,seed=seed),self.assertRaises(ValueError):prepare(p,condition,seed)
            p.write_text('preserve')
            with self.assertRaisesRegex(ValueError,'Preserve existing'):prepare(p,'whale',42)
            self.assertEqual(p.read_text(),'preserve')


@unittest.skipUnless(importlib.util.find_spec('torch'),'Requires the native tensor runtime')
class NativePhaseBoundaryTests(unittest.TestCase):
    def test_original_online_loop_generates_one_batch_and_saves_even_an_empty_update(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import json
        import torch
        from omegaconf import OmegaConf
        from verl import DataProto
        import verl.trainer.ppo.ray_trainer as module
        for before,target,accepted in [(0,1,3),(0,1,0),(1,2,3),(1,2,0)]:
            with self.subTest(before=before,target=target,accepted=accepted):
                calls=[];updates=[];saves=[];synchronizations=[];logged=[]
                config=OmegaConf.create({'trainer':{'online_rsft':{'iterations':0,'sft_epochs':1},
                    'save_freq':1,'test_freq':-1,'rollout_data_dir':None},
                    'actor_rollout_ref':{'actor':{'ppo_mini_batch_size':8,'ppo_micro_batch_size_per_gpu':1},
                        'rollout':{'temperature':1.,'n':8}}})
                trainer=module.RayPPOTrainer.__new__(module.RayPPOTrainer)
                trainer.config=config;trainer.global_steps=before;trainer.total_training_steps=target
                trainer.train_dataloader=[{'input_ids':torch.ones(8,1,dtype=torch.long),
                                          'attention_mask':torch.ones(8,1,dtype=torch.long)} for _ in range(2)]
                trainer.train_dataset=[];trainer.actor_rollout_wg=SimpleNamespace();trainer.use_rm=False;trainer.use_critic=False
                trainer.resource_pool_manager=SimpleNamespace(get_n_gpus=lambda:2)
                trainer._get_gen_batch=lambda batch:batch
                def generate(batch):
                    calls.append(len(batch));self.assertEqual(batch.meta_info['global_steps'],target)
                    rewards=torch.zeros(64,2);rewards[:accepted,-1]=1.
                    return DataProto.from_dict({'responses':torch.ones(64,2,dtype=torch.long),
                        'response_mask':torch.ones(64,2,dtype=torch.long),'rm_scores':rewards})
                trainer.async_rollout_manager=SimpleNamespace(generate_sequences=generate)
                trainer.checkpoint_manager=SimpleNamespace(sleep_replicas=lambda:None,update_weights=synchronizations.append)
                def update(batch):
                    updates.append(len(batch))
                    return DataProto(meta_info={'metrics':{'actor/sft_updates':[1.]}})
                trainer._update_sft_actor=update;trainer._save_checkpoint=lambda:saves.append(trainer.global_steps)
                logger=SimpleNamespace(log=lambda data,step:logged.append((step,data)),finish=lambda:None)
                with patch.object(module,'compute_data_metrics',return_value={}), \
                     patch.object(module,'compute_timing_metrics',return_value={}), \
                     patch.object(module,'compute_throughout_metrics',return_value={}):
                    trainer._fit_online_rsft(logger)
                self.assertEqual(calls,[64]);self.assertEqual(updates,[3] if accepted else [])
                self.assertEqual(saves,[target]);self.assertEqual(synchronizations,[target])
                self.assertEqual(trainer.global_steps,target);self.assertEqual(len(logged),1)
                self.assertEqual(logged[0][1]['online_rsft/accepted'],accepted)
                if not accepted:self.assertEqual(logged[0][1]['online_rsft/skipped_empty_sft'],1.)
                print(json.dumps({'kind':'native_single_phase_stop_fixture','status':'PASS',
                    'global_step_before':before,'target':target,'synthetic_accepted':accepted,
                    'generated_batch_sizes':calls,'saved_steps':saves,'synced_steps':synchronizations,
                    'new_model_calls':0,'native_checkpoint_loader_executed':False}),flush=True)


if __name__=='__main__':unittest.main()
