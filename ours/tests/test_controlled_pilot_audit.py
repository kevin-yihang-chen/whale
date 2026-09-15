"""Match audit configuration to actual native two-batch dataloader mutations."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires native runtime')
class ControlledRuntimeAuditTests(unittest.TestCase):
    def test_native_dataloader_sets_the_same_two_step_runtime_contract(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        from omegaconf import OmegaConf
        from verl.experimental.reward_loop import migrate_legacy_reward_impl
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from ours.audit_controlled_pilot import runtime_configuration
        from ours.controlled_pilot import CPU_NAME
        plan=json.loads(Path('results/controlled-weight-only-seed42-plan-20260910-v2.json').read_text())
        job='999999'
        expected,context=runtime_configuration(plan,job)
        raw=Path(plan['resolved_config']).read_text().replace(CPU_NAME,f'controlled-weight-only-seed42-{job}')
        actual=migrate_legacy_reward_impl(OmegaConf.create(raw[raw.index('model_engine: dp\n'):]))
        # The task runner assigns the non-dedicated reward resource dimensions
        # before construction; the native trainer sets both optimizer schedules.
        actual.reward.reward_model.nnodes=actual.trainer.nnodes
        actual.reward.reward_model.n_gpus_per_node=actual.trainer.n_gpus_per_node
        trainer=RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.config=actual
        trainer._create_dataloader(list(range(128)),list(range(128)),lambda x:x,None)
        self.assertEqual(trainer.total_training_steps,2)
        self.assertEqual(len(trainer.train_dataloader),16)
        self.assertEqual(OmegaConf.to_container(actual,resolve=True),OmegaConf.to_container(expected,resolve=True))
        self.assertEqual(context['job_id'],job)
        wrong=deepcopy(plan)
        wrong['native_batch_steps']=1
        with self.assertRaisesRegex(ValueError,'stopping rule'):
            runtime_configuration(wrong,job)
        print(json.dumps({'kind':'controlled_runtime_audit_fixture','status':'PASS',
            'native_train_batches_per_epoch':16,'native_total_training_steps':2,
            'limitation':'Native CPU dataloader/configuration fixture, not task replay or GPU training.'}),flush=True)


if __name__=='__main__':
    unittest.main()
