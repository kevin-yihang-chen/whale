"""Check compact state boundaries before allocating real training GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours import fast_chart_training_bootstrap as hooks
from ours.fast_chart_calibration_controller import choose_rate
from ours.fast_chart_training import RATES, checkpoint
from ours.visual_task import file_sha256


class CompactTrainingTests(unittest.TestCase):
    def test_worker_installation_preserves_native_pixel_and_rsft_hooks(self):
        script='''
from ours.fast_chart_training_bootstrap import prepare_worker
prepare_worker()
import torch
import verl.trainer.main_textarena_disagg_rsft as trainer
import verl.utils.checkpoint.fsdp_checkpoint_manager as checkpoint
import verl.workers.fsdp_workers as worker
from ours.native_resume_observation import state_digest
classes=(trainer.DisaggregatedRayTrainer,trainer.CheckpointEngineManager,checkpoint.FSDPCheckpointManager)
assert all(c._compact_observer for c in classes)
assert classes[0]._visual_training_recorder
assert worker.FSDPCheckpointManager is classes[2]
before=state_digest(classes[2].get_rng_state())
prepare_worker()
assert classes==(trainer.DisaggregatedRayTrainer,trainer.CheckpointEngineManager,checkpoint.FSDPCheckpointManager)
assert state_digest(classes[2].get_rng_state())==before
assert not torch.cuda.is_initialized()
print('PASS_COMPACT_WORKER_HOOK')
'''
        result=subprocess.check_output([sys.executable,'-c',script],
            env=dict(os.environ,WHALE_TRIAL_SEED='42',PYTHONHASHSEED='42'),text=True,stderr=subprocess.STDOUT)
        self.assertIn('PASS_COMPACT_WORKER_HOOK',result)

    def test_lr_choice_uses_all_points_accuracy_then_smaller_rate(self):
        rows={str(rate):{'status':'COMPLETE_COMPACT_NATIVE_FOLLOWUP','marginal_accuracy':.75,'paired_accuracy':.9 if rate==1e-5 else .1} for rate in RATES}
        self.assertEqual(choose_rate(rows),1e-7)
        rows[str(1e-6)]['marginal_accuracy']=.76
        self.assertEqual(choose_rate(rows),1e-6)
        del rows[str(1e-5)]
        with self.assertRaises(ValueError):choose_rate(rows)

    def test_transport_handles_four_steps_and_requires_initial_loader(self):
        from verl.utils.ray_utils import auto_await
        events=[]
        class Transport:
            @auto_await
            async def update_weights(self,global_steps=None):events.append(global_steps)
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'state').mkdir();(root/'requests').mkdir()
            path=root/'plan.json';path.write_text('{}')
            plan={'output':str(root),'stage':2}
            async def remote(method,timeout,kwargs):
                step=kwargs['global_step']
                self.assertEqual(events[-1],step)
                (root/f'state/receiver-step{step}-fixture.json').write_text(json.dumps({
                    'status':'PASS_COMPACT_RECEIVER','plan_sha256':file_sha256(path)}))
            manager=hooks.transport_class(Transport)();manager.backend='nccl'
            manager.replicas=[SimpleNamespace(servers=[SimpleNamespace(collective_rpc=SimpleNamespace(remote=remote))])]
            with patch.object(hooks,'context',return_value=(path,plan)):
                with self.assertRaises(ValueError):manager.update_weights(4)
                self.assertEqual(events,[])
                (root/'state/loader.json').write_text('{}')
                for step in range(4,9):manager.update_weights(step)
                self.assertEqual(events,list(range(4,9)))
                self.assertEqual(len(list((root/'state').glob('receiver-*'))),2)
                with self.assertRaises(ValueError):manager.update_weights(9)

    def test_native_checkpoint_rejects_missing_state_and_adam_mismatch(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)/'global_step_4';actor=root/'actor';actor.mkdir(parents=True)
            with self.assertRaises(ValueError):checkpoint(root,4)
            for name in ('model_world_size_1_rank_0.pt','extra_state_world_size_1_rank_0.pt'):(actor/name).write_bytes(b'fixture')
            (actor/'fsdp_config.json').write_text('{"world_size":1}')
            (root/'data.pt').write_bytes(b'state')
            self.assertFalse(checkpoint(root,4)['adam_moments_saved'])
            (actor/'optim_world_size_1_rank_0.pt').write_bytes(b'fixture')
            with self.assertRaises(ValueError):checkpoint(root,4)


if __name__=='__main__':unittest.main()
