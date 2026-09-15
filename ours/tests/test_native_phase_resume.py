"""Exercise the native checkpoint loader, including deliberate Adam exclusion."""
from contextlib import nullcontext
import importlib.util
import gzip
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from ours.native_phase_resume import check_layout,phase_overrides


class ResumeBoundaryTests(unittest.TestCase):
    def test_cumulative_stopping_and_native_resume_are_explicit(self):
        first=phase_overrides(42,1)
        self.assertIn('trainer.total_training_steps=1',first)
        self.assertIn('trainer.online_rsft.iterations=0',first)
        second=phase_overrides(44,2,Path('/tmp/fixture/global_step_1'))
        self.assertIn('trainer.resume_mode=resume_path',second)
        self.assertIn('trainer.total_training_steps=2',second)
        self.assertIn('trainer.del_local_ckpt_after_load=false',second)
        for seed,step,path in [(42,2,None),(42,1,'/tmp/global_step_1'),(42,2,'/tmp/global_step_2'),(45,1,None)]:
            with self.subTest(seed=seed,step=step,path=path),self.assertRaises(ValueError):phase_overrides(seed,step,path)

    def test_missing_dataloader_is_an_error_before_native_loader_can_warn_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'global_step_1';(root/'actor/huggingface').mkdir(parents=True)
            for name in ('model_world_size_1_rank_0.pt','extra_state_world_size_1_rank_0.pt'):
                (root/'actor'/name).write_bytes(b'fixture')
            (root/'actor/fsdp_config.json').write_text('{"world_size":1}')
            (root/'actor/huggingface/config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'data.pt'):check_layout(root)
            (root/'data.pt').write_bytes(b'fixture')
            self.assertEqual(check_layout(root),root)
            (root/'actor/optim_world_size_1_rank_0.pt').write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError,'optimizer-state'):check_layout(root)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Requires native Torch runtime')
class NativeCheckpointLoaderTests(unittest.TestCase):
    def test_native_loader_schedule_matches_real_saved_generator_and_real_first_batch(self):
        import torch
        from omegaconf import OmegaConf
        from ours.native_loader_schedule import native_schedule
        plan_path=Path('results/controlled-weight-only-seed42-plan-20260910-v2.json')
        plan=json.loads(plan_path.read_text())
        raw=Path(plan['resolved_config']).read_text();config=OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
        result=native_schedule(config,plan['native_dataset_order']['native_filtered_ids'])
        root=Path('data/native-rsft/controlled-weight-only-seed42-222801')
        with gzip.open(root/'audit/batch-1.jsonl.gz','rt') as stream:
            next(stream)
            ids=[json.loads(line)['metadata']['extra_info']['puzzle_id'] for i,line in enumerate(stream) if i%8==0]
        self.assertEqual(result['batch_ids'][0],ids)
        self.assertNotEqual(result['batch_ids'],plan['native_dataset_order']['planned_batch_ids'])
        state=torch.load(root/'global_step_1/data.pt',map_location='cpu',weights_only=False)
        main=state['_snapshot']['_main_snapshot'];sample=main['_sampler_iter_state']['sampler_iter_state']
        self.assertEqual(result['checkpoint_sampler_snapshots'][0],{'sampler_yielded':sample['yielded'],
            'generator_sha256':hashlib.sha256(sample['generator'].numpy().tobytes()).hexdigest(),'base_seed':main['_base_seed']})
        constructions=result['sampler_iterator_constructions']
        self.assertEqual(len(constructions),2)
        self.assertEqual(constructions[0]['after_sha256'],constructions[1]['before_sha256'])
        self.assertEqual(constructions[1]['before_sha256'],result['checkpoint_sampler_snapshots'][0]['generator_sha256'])
        print(json.dumps({'kind':'native_loader_real_state_regression','status':'PASS','first_batch_ids':ids,
            'next_batch_ids':result['batch_ids'][1],'preflight_deviation_preserved':True,'new_model_calls':0}),flush=True)

    def test_original_loader_restores_model_scheduler_and_rng_but_does_not_load_adam(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import numpy as np
        import torch
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        import verl.utils.checkpoint.fsdp_checkpoint_manager as module
        model=torch.nn.Linear(2,1,bias=False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-7)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        model(torch.ones(1,2)).sum().backward();optimizer.step();scheduler.step()
        expected={k:v.detach().clone() for k,v in model.state_dict().items()}
        extra={'rng':FSDPCheckpointManager.get_rng_state(),'lr_scheduler':scheduler.state_dict()}
        expected_draw=(random.random(),float(np.random.random()),torch.rand(2))
        incoming=torch.nn.Linear(2,1,bias=False)
        fresh=torch.optim.AdamW(incoming.parameters(),lr=1e-7)
        resumed_scheduler=torch.optim.lr_scheduler.LambdaLR(fresh,lambda _:1.)
        manager=FSDPCheckpointManager.__new__(FSDPCheckpointManager)
        manager.rank=0;manager.world_size=1;manager.model=incoming;manager.optimizer=fresh
        manager.lr_scheduler=resumed_scheduler;manager.checkpoint_load_contents=['model','extra']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            torch.save(expected,root/'model_world_size_1_rank_0.pt')
            torch.save(extra,root/'extra_state_world_size_1_rank_0.pt')
            # Poisoned optimizer bytes prove that this path is never opened.
            (root/'optim_world_size_1_rank_0.pt').write_bytes(b'must not be loaded')
            with patch.object(module,'get_fsdp_state_ctx',return_value=nullcontext()), \
                 patch.object(torch.distributed,'barrier'),patch.object(fresh,'load_state_dict',wraps=fresh.load_state_dict) as adam:
                manager.load_checkpoint(str(root),del_local_after_load=False)
                adam.assert_not_called()
            for key,value in incoming.state_dict().items():self.assertTrue(torch.equal(value,expected[key]))
            self.assertEqual(resumed_scheduler.state_dict(),scheduler.state_dict())
            self.assertFalse(fresh.state)
            self.assertEqual(random.random(),expected_draw[0]);self.assertEqual(float(np.random.random()),expected_draw[1])
            self.assertTrue(torch.equal(torch.rand(2),expected_draw[2]))
        print(json.dumps({'kind':'native_phase_checkpoint_loader_fixture','status':'PASS',
            'model_restored':True,'scheduler_restored':True,'cpu_numpy_python_rng_restored':True,
            'adam_moments_loaded':False,'gpu_rpc_executed':False,'task_inference_calls':0}),flush=True)


if __name__=='__main__':unittest.main()
