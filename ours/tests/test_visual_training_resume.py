"""CPU restoration/control-flow fixtures; no research model or proposer calls."""
import asyncio
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours import visual_resume_bootstrap as hooks
from ours import visual_training_resume as resume
from ours.visual_task import file_sha256


class VisualResumeTest(unittest.TestCase):
    def test_worker_hook_is_idempotent_and_keeps_visual_recording(self):
        import subprocess
        import sys
        script = '''
import torch
from ours.visual_resume_bootstrap import prepare_worker, state_digest
prepare_worker()
import verl.utils.checkpoint.fsdp_checkpoint_manager as checkpoint
import verl.workers.fsdp_workers as worker
import verl.trainer.main_textarena_disagg_rsft as trainer
classes = (checkpoint.FSDPCheckpointManager, trainer.DisaggregatedRayTrainer, trainer.CheckpointEngineManager)
assert classes[0] is worker.FSDPCheckpointManager
assert all(c._visual_resume_observer for c in classes)
assert classes[1]._visual_training_recorder
assert not getattr(classes[1], '_compact_recorded_training', False)
before = state_digest(classes[0].get_rng_state())
prepare_worker()
assert classes == (checkpoint.FSDPCheckpointManager, trainer.DisaggregatedRayTrainer, trainer.CheckpointEngineManager)
assert state_digest(classes[0].get_rng_state()) == before
assert not torch.distributed.is_initialized() and not torch.cuda.is_initialized()
print('PASS_VISUAL_RESUME_HOOK_CPU_NO_GPU_OR_GENERATION')
'''
        output = subprocess.check_output([sys.executable, '-c', script],
            cwd=resume.ROOT / 'upstream/WHALE/domains/chess_puzzles',
            env=dict(os.environ, WHALE_TRIAL_SEED='42', PYTHONHASHSEED='42'), text=True, stderr=subprocess.STDOUT)
        self.assertIn('PASS_VISUAL_RESUME_HOOK_CPU_NO_GPU_OR_GENERATION', output)

    def test_resume_changes_only_declared_configuration_fields(self):
        first = resume.read(resume.ROOT / 'results/native-visual-rsft-plan-20260910-v4.json')
        original = deepcopy(first)
        cfg = resume.configuration(first, Path('/tmp/visual-resume-fixture'), Path('/tmp/fixture-plan.json'), Path('/tmp/h1.py'))
        self.assertEqual(first, original)
        def flatten(value, prefix=''):
            if isinstance(value, dict):
                return {key: val for k, v in value.items() for key, val in flatten(v, prefix + k + '.').items()}
            return {prefix[:-1]: value}
        a, b = flatten(first['config']), flatten(cfg)
        changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        self.assertEqual(changed, {
            'data.visual_harness_path', 'data.cache_dir', 'trainer.total_training_steps', 'trainer.resume_mode',
            'trainer.resume_from_path', 'trainer.experiment_name', 'trainer.default_local_dir',
            'trainer.rollout_data_dir', 'trainer.validation_data_dir', 'actor_rollout_ref.rollout.trace.experiment_name',
            'actor_rollout_ref.rollout.engine_kwargs.vllm.worker_cls',
            'ray_kwargs.ray_init.runtime_env.worker_process_setup_hook',
            'ray_kwargs.ray_init.runtime_env.env_vars.VETO_NATIVE_EVALUATION_PLAN',
            'ray_kwargs.ray_init.runtime_env.env_vars.VETO_NATIVE_EVALUATION_OUTPUT',
            'ray_kwargs.ray_init.runtime_env.env_vars.VETO_VISUAL_RESUME_PLAN',
            'ray_kwargs.ray_init.runtime_env.env_vars.VETO_VISUAL_FORWARD_AUDIT'})
        self.assertEqual(cfg['trainer']['online_rsft'], first['config']['trainer']['online_rsft'])
        self.assertEqual(cfg['actor_rollout_ref']['actor'], first['config']['actor_rollout_ref']['actor'])
        self.assertIsNone(cfg['trainer']['max_actor_ckpt_to_keep'])

    def test_unfinished_search_cannot_supply_training_harness(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'accepted_harness.txt').write_text('h1\n')
            with self.assertRaises(FileNotFoundError):
                resume.certify_handoff(root, root / 'first.json', root / 'followup.json')

    def test_text_loss_lineage_cannot_be_resumed_as_visual_training(self):
        with self.assertRaisesRegex(ValueError, 'image-conditioned'):
            resume.certify_visual_training_lineage({'kind': 'native_visual_rsft_engineering'}, Path('missing.json'))

    def test_native_actor_restore_preserves_rng_and_resets_adam(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import numpy as np
        import torch
        import verl.utils.checkpoint.fsdp_checkpoint_manager as native
        source = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(source.parameters(), lr=1e-7)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        source(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        schedule.step()
        saved = {k: v.detach().clone() for k, v in source.state_dict().items()}
        extra = {'rng': native.FSDPCheckpointManager.get_rng_state(), 'lr_scheduler': schedule.state_dict()}
        expected_draw = (random.random(), float(np.random.random()), torch.rand(5))
        model = torch.nn.Linear(3, 2)
        restored_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
        cls = hooks.checkpoint_class(native.FSDPCheckpointManager, expected_device='cpu')
        manager = cls.__new__(cls)
        manager.model, manager.optimizer, manager.rank, manager.world_size = model, restored_optimizer, 0, 1
        manager.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.)
        manager.checkpoint_load_contents = ['model', 'extra']
        with TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / 'global_step_1'
            actor = directory / 'actor'
            actor.mkdir(parents=True)
            torch.save(saved, actor / 'model_world_size_1_rank_0.pt')
            torch.save(extra, actor / 'extra_state_world_size_1_rank_0.pt')
            plan = {'output': str(root / 'output'), 'harness_sha256': 'fixture',
                'resume_checkpoint': {'artifact_sha256': {str(p.relative_to(directory)): file_sha256(p) for p in actor.iterdir()}}}
            path = root / 'plan.json'
            path.write_text(json.dumps(plan))
            with patch.object(hooks, 'context', return_value=(path, plan, directory)), \
                 patch.object(native, 'get_fsdp_state_ctx', return_value=nullcontext()), \
                 patch.object(torch.distributed, 'barrier'), patch.dict(os.environ, SLURM_JOB_ID='cpu-fixture'), \
                 patch.object(restored_optimizer, 'load_state_dict', wraps=restored_optimizer.load_state_dict) as adam:
                manager.load_checkpoint(str(actor))
                adam.assert_not_called()
                self.assertEqual(random.random(), expected_draw[0])
                self.assertEqual(float(np.random.random()), expected_draw[1])
                self.assertTrue(torch.equal(torch.rand(5), expected_draw[2]))
                report = resume.read(root / 'output/resume/actor.json')
                self.assertEqual(report['device'], 'cpu')
                self.assertTrue(report['all_values_exact_native_fp32'])
                self.assertEqual(report['optimizer_state_entries'], 0)
                with self.assertRaises(AssertionError):
                    manager.load_checkpoint(str(actor))

    def test_receiver_checks_native_cast_and_rejects_stale_values(self):
        import torch
        from ours.visual_resume_model_worker import observe_received_weights
        class ToyModel(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                self.embedding = torch.nn.Embedding.from_pretrained(values.to(torch.bfloat16))
            def embed_input_ids(self, ids):
                return self.embedding(ids)
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
        coordinates = [{'token_id': row, 'column': col, 'base': -1., 'expected': float(values[row, col].to(torch.bfloat16))}
                       for row in range(2) for col in range(4)]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / 'global_step_1/actor'
            directory.mkdir(parents=True)
            torch.save({'model.language_model.embed_tokens.weight': values}, directory / 'model_world_size_1_rank_0.pt')
            plan = {'resume_checkpoint': {'directory': str(directory.parent)}, 'resume_coordinates': coordinates}
            before = torch.get_rng_state().clone()
            good = ToyModel(values)
            before = torch.get_rng_state().clone()
            report = observe_received_weights(good, plan, 1)
            self.assertEqual(report['actual']['values'], [p['expected'] for p in coordinates])
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            with self.assertRaises(AssertionError):
                observe_received_weights(ToyModel(values + 1.), plan, 1)

    def test_transport_receipt_follows_native_sync_and_requires_restore(self):
        from verl.utils.ray_utils import auto_await
        events = []
        class BaseTransport:
            @auto_await
            async def update_weights(self, global_steps=None):
                events.append(('native_sync', global_steps))
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'resume').mkdir()
            (root / 'requests').mkdir()
            path = root / 'plan.json'
            path.write_text('{}')
            plan = {'output': str(root)}
            async def remote(method, timeout, kwargs):
                self.assertEqual(events[-1], ('native_sync', 1))
                self.assertEqual(method, 'record_native_resumed_weights')
                self.assertEqual(kwargs, {'global_step': 1})
                events.append(('receiver_probe', 1))
                (root / 'resume/receiver-step1-fixture.json').write_text(json.dumps({
                    'status': 'PASS_NATIVE_RECEIVER_COORDINATES', 'global_step': 1, 'plan_sha256': file_sha256(path)}))
            cls = hooks.transport_class(BaseTransport)
            manager = cls()
            manager.backend = 'nccl'
            manager.replicas = [SimpleNamespace(servers=[SimpleNamespace(collective_rpc=SimpleNamespace(remote=remote))])]
            with patch.object(hooks, 'context', return_value=(path, plan, root)):
                with self.assertRaises(FileNotFoundError):
                    manager.update_weights(1)
                self.assertEqual(events, [])
                for name in ('actor', 'loader'):
                    (root / f'resume/{name}.json').write_text(json.dumps({'plan_sha256': file_sha256(path)}))
                manager.update_weights(1)
                self.assertEqual(events, [('native_sync', 1), ('receiver_probe', 1)])
                self.assertEqual(resume.read(root / 'resume/transport-step1.json')['status'], 'PASS_NATIVE_WEIGHT_TRANSPORT_OBSERVED')


if __name__ == '__main__':
    unittest.main()
