"""Native restoration with passive observation; CPU fixtures, no task inference."""
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours import native_resume_observation as observation
from ours.visual_task import file_sha256


@unittest.skipUnless(importlib.util.find_spec('torch'),'Requires the native Torch runtime')
class NativeObservationTests(unittest.TestCase):
    def test_full_worker_hook_is_idempotent_from_the_native_upstream_working_directory(self):
        import subprocess
        import sys
        script="""
import json,torch
from ours.native_resume_observation import prepare_worker,state_digest,ROOT
prepare_worker()
import verl.utils.checkpoint.fsdp_checkpoint_manager as checkpoint
import verl.workers.fsdp_workers as worker
import verl.trainer.main_textarena_disagg_rsft as trainer
actor_class,driver_class=checkpoint.FSDPCheckpointManager,trainer.DisaggregatedRayTrainer
assert actor_class is worker.FSDPCheckpointManager
assert actor_class._joint_resume_observation and driver_class._joint_resume_observation
assert driver_class._compact_recorded_training
before=state_digest(actor_class.get_rng_state())
prepare_worker()
assert checkpoint.FSDPCheckpointManager is actor_class and trainer.DisaggregatedRayTrainer is driver_class
assert state_digest(actor_class.get_rng_state())==before
assert not torch.distributed.is_initialized() and not torch.cuda.is_initialized()
print(json.dumps({'kind':'native_resume_worker_hook_cpu_fixture','status':'PASS','idempotent':True,
    'shared_recording_and_compaction_retained':True,'gpu_restore_executed':False,'new_model_calls':0}))
"""
        env=dict(os.environ,WHALE_TRIAL_SEED='42',PYTHONHASHSEED='42')
        output=subprocess.check_output([sys.executable,'-c',script],cwd=observation.ROOT/'upstream/WHALE/domains/chess_puzzles',
            env=env,text=True,stderr=subprocess.STDOUT)
        report=json.loads(output.splitlines()[-1]);self.assertEqual(report['status'],'PASS')
        print(json.dumps(report),flush=True)

    def test_full_actor_observation_retains_original_restore_and_future_rng_draws(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import numpy as np
        import torch
        import verl.utils.checkpoint.fsdp_checkpoint_manager as native
        source=torch.nn.Linear(3,2)
        original=torch.optim.AdamW(source.parameters(),lr=1e-7)
        scheduler=torch.optim.lr_scheduler.LambdaLR(original,lambda _:1.)
        source(torch.ones(1,3)).sum().backward();original.step();scheduler.step()
        saved={k:v.detach().clone() for k,v in source.state_dict().items()}
        extra={'rng':native.FSDPCheckpointManager.get_rng_state(),'lr_scheduler':scheduler.state_dict()}
        next_draw=(random.random(),float(np.random.random()),torch.rand(5))
        model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-7)
        cls=observation.observed_manager_class(native.FSDPCheckpointManager,expected_device='cpu')
        manager=cls.__new__(cls);manager.model=model;manager.optimizer=optimizer;manager.rank=0;manager.world_size=1
        manager.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        manager.checkpoint_load_contents=['model','extra']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'global_step_1';actor=directory/'actor';actor.mkdir(parents=True)
            torch.save(saved,actor/'model_world_size_1_rank_0.pt')
            torch.save(extra,actor/'extra_state_world_size_1_rank_0.pt')
            plan={'condition':'whale','seed':42,'harness_sha256':'fixture',
                'resume_checkpoint':{'artifact_sha256':{str(p.relative_to(directory)):file_sha256(p) for p in actor.iterdir()}}}
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
            with patch.object(native,'get_fsdp_state_ctx',return_value=nullcontext()), \
                 patch.object(torch.distributed,'barrier'),patch.object(observation,'ROOT',root), \
                 patch.object(observation,'resume_plan',return_value=(plan_path,plan,directory)), \
                 patch.dict(os.environ,SLURM_JOB_ID='cpu-fixture'), \
                 patch.object(optimizer,'load_state_dict',wraps=optimizer.load_state_dict) as adam:
                manager.load_checkpoint(str(actor))
                adam.assert_not_called()
                self.assertEqual(random.random(),next_draw[0]);self.assertEqual(float(np.random.random()),next_draw[1])
                self.assertTrue(torch.equal(torch.rand(5),next_draw[2]))
                report=json.loads((root/'results/controlled-joint-phase2-cpu-fixture/resume-actor-rank-0.json').read_text())
                self.assertEqual(report['status'],'PASS');self.assertEqual(report['device'],'cpu')
                self.assertEqual(report['tensor_count_including_aliases'],2);self.assertEqual(report['elements_including_aliases'],8)
                self.assertEqual(report['rng_before_observation_sha256'],report['rng_after_observation_sha256'])
                with self.assertRaisesRegex(ValueError,'repeated actor restoration'):manager.load_checkpoint(str(actor))
                native.FSDPCheckpointManager.load_rng_state(extra['rng'])
                for defect in ('value','rng','scheduler','optimizer'):
                    with self.subTest(defect=defect):
                        model.load_state_dict(saved);manager.lr_scheduler.load_state_dict(extra['lr_scheduler'])
                        native.FSDPCheckpointManager.load_rng_state(extra['rng']);optimizer.state.clear()
                        if defect=='value':
                            with torch.no_grad():model.weight[0,0]+=0.25
                        elif defect=='rng':torch.rand(1)
                        elif defect=='scheduler':manager.lr_scheduler.last_epoch+=1
                        else:optimizer.state[next(iter(model.parameters()))]={'step':1}
                        with self.assertRaises(ValueError):observation.observe_actor(manager,actor,expected_device='cpu')
        print(json.dumps({'kind':'passive_native_actor_cpu_fixture','status':'PASS',
            'full_native_loader_executed':True,'exact_parameters_scheduler_rng':True,
            'observer_retains_subsequent_random_draws':True,'gpu_restore_executed':False,'task_model_calls':0}),flush=True)

    def test_pending_real_dataloader_state_is_observed_without_iterator_or_rng_change(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer,AutoProcessor
        from verl.trainer.main_ppo import create_rl_dataset,create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.utils.dataset.rl_dataset import collate_fn
        plan=json.loads(Path('results/controlled-weight-only-seed42-plan-20260910-v2.json').read_text())
        raw=Path(plan['resolved_config']).read_text();cfg=OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
        directory=Path('data/native-rsft/controlled-weight-only-seed42-222801/global_step_1').resolve()
        cfg.trainer.resume_mode='resume_path';cfg.trainer.resume_from_path=str(directory)
        cfg.trainer.del_local_ckpt_after_load=False
        tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
        processor=AutoProcessor.from_pretrained(plan['model'],local_files_only=True)
        with patch.dict(os.environ,HARNESS_PATH=plan['harness'],CHESS_PUZZLE_HARNESS_PATH=plan['harness']):
            dataset=create_rl_dataset(cfg.data.train_files,cfg.data,tokenizer,processor,is_train=True,max_samples=cfg.data.train_max_samples)
        observed=[]
        trainer=RayPPOTrainer.__new__(RayPPOTrainer);trainer.config=cfg;trainer.global_steps=0;trainer.use_critic=False
        trainer.actor_rollout_wg=SimpleNamespace(load_checkpoint=lambda *args,**kw:observed.append((args,kw)))
        trainer._create_dataloader(dataset,dataset,collate_fn,create_rl_sampler(cfg.data,dataset))
        trainer._load_checkpoint()
        digest=lambda:observation.state_digest((random.getstate(),np.random.get_state(),torch.get_rng_state()))
        before=digest()
        with patch.object(trainer.train_dataloader,'state_dict',side_effect=AssertionError('Iterator creation forbidden')):
            report=observation.observe_loader(trainer,directory)
        self.assertEqual(before,digest());self.assertIsNone(trainer.train_dataloader._iterator)
        self.assertFalse(report['iterator_created_by_observer']);self.assertEqual(report['sampler_yielded'],8)
        self.assertEqual(observed,[((str(directory/'actor'),),{'del_local_after_load':False})])
        iterator=iter(trainer.train_dataloader)
        try:ids=[row['puzzle_id'] for row in next(iterator)['extra_info']]
        finally:
            if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        expected=['004u0','00Ar2','005gP','003r5','005nD','00AB1','003UW','00B7G']
        self.assertEqual(ids,expected)
        with self.assertRaisesRegex(ValueError,'loader lifecycle'):observation.observe_loader(trainer,directory)
        print(json.dumps({'kind':'passive_native_saved_loader_fixture','status':'PASS','next_batch_ids':ids,
            'original_preflight_deviation_preserved':True,'native_actor_rpc_stubbed':True,'task_model_calls':0}),flush=True)


if __name__=='__main__':unittest.main()
