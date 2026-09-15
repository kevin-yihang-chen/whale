"""Run real frozen data/native trainer flow; stub only GPU RPC/save boundaries.

These checks inspect cached-batch order and the exact actor input. They execute
no model inference or optimizer and must not be reported as training success.
"""
from contextlib import chdir
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires native training runtime')
class AlternationRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        from ours.alternation_recovery import parsed
        from ours.recovery_gate import native_runtime_config
        cls.plan_path = Path(os.environ.get('WHALE_ALTERNATION_REPLAY_TEST_PLAN',
                             'results/alternation-recovery-plan-20260909-v3.json')).resolve()
        cls.plan = json.loads(cls.plan_path.read_text())
        cls.config = native_runtime_config(parsed(Path(cls.plan['resolved_config']).read_text()))

    def setUp(self):
        import verl.utils.experimental.torch_functional as native
        from verl.checkpoint_engine import CheckpointEngineRegistry
        original_engine = CheckpointEngineRegistry.get('nccl')
        self.addCleanup(CheckpointEngineRegistry.register('nccl'), original_engine)
        original = native.FusedLinearForPPO
        env = patch.dict(os.environ, {'WHALE_RSFT_FUSED_CHUNK_SIZE': '128'})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(setattr, native, 'FusedLinearForPPO', original)

    def test_complete_native_dataset_and_trainer_preserve_all_eleven_accepted_rows(self):
        import torch
        from transformers import AutoProcessor, AutoTokenizer
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl import DataProto
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn
        import verl.trainer.main_textarena_disagg_rsft as module
        import verl.experimental.agent_loop.agent_loop as agent_module
        from ours.recovered_alternation_bootstrap import restore_batch, install_recovery, RecordedAlternationBatch
        config = deepcopy(self.config)
        tokenizer = AutoTokenizer.from_pretrained(self.plan['model'], local_files_only=True)
        processor = AutoProcessor.from_pretrained(self.plan['model'], local_files_only=True)
        dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True, max_samples=-1)
        loader = StatefulDataLoader(dataset, batch_size=8, num_workers=8, drop_last=True,
                                    collate_fn=collate_fn, sampler=create_rl_sampler(config.data, dataset))
        batch, receipt = restore_batch(self.plan['batch_directory'], self.plan['batch_audit'], self.plan)
        observed, saves, syncs, logs = [], [], [], []
        class FixtureActorGroup:
            _dispatch_info = {'actor': [0]}
            def update_sft_actor(self, data):
                observed.append(data)
                return DataProto(meta_info={'metrics': {'fixture/no_optimizer_executed': [1.]}})
        original, original_manager = module.DisaggregatedRayTrainer, agent_module.AsyncLLMServerManager
        try:
            install_recovery()
            trainer = module.DisaggregatedRayTrainer.__new__(module.DisaggregatedRayTrainer)
            with tempfile.TemporaryDirectory(prefix='whale-alternation-loop-fixture-') as directory:
                root = Path(directory)
                config.trainer.default_local_dir = str(root)
                config.trainer.rollout_data_dir = str(root / 'rollouts')
                trainer.config, trainer.tokenizer = config, tokenizer
                trainer.global_steps, trainer.total_training_steps = 0, 1
                trainer.train_dataset, trainer.train_dataloader = dataset, loader
                trainer.use_rm = trainer.use_critic = False
                trainer.actor_rollout_wg = FixtureActorGroup()
                trainer.async_rollout_manager = RecordedAlternationBatch(object(), batch, receipt, root / 'recovery', self.plan)
                trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None,
                    update_weights=lambda step: syncs.append(step))
                trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 2)
                trainer._save_checkpoint = lambda: saves.append(trainer.global_steps)
                trainer._fit_online_rsft(SimpleNamespace(log=lambda **kwargs: logs.append(kwargs), finish=lambda: None))
                received, = observed
                self.assertEqual(tuple(received.batch['responses'].shape), (11, 6321))
                self.assertEqual(tuple(received.batch['prompts'].shape), (11, 4096))
                self.assertEqual(int(received.batch['response_mask'].sum()), 30392)
                self.assertEqual(received.batch['dummy_tensor'].dtype, torch.uint8)
                self.assertEqual(int(received.batch['dummy_tensor'].sum()), 0)
                self.assertEqual(received.meta_info['sft_mini_batch_size'], 8)
                self.assertEqual(received.meta_info['sft_micro_batch_size_per_gpu'], 1)
                self.assertEqual([len(part) for part in received.split(8)], [8, 3])
                self.assertEqual((saves, syncs), ([1], [1]))
                self.assertEqual(logs[-1]['data']['online_rsft/accepted'], 11)
                self.assertEqual(len((root / 'rollouts/1.jsonl').read_text().splitlines()), 64)
                report = json.loads((root / 'recovery/recovery.json').read_text())
                self.assertEqual((report['source_job_id'], report['new_model_calls']), ('222292', 0))
                with self.assertRaisesRegex(ValueError, 'only once'):
                    trainer.async_rollout_manager.generate_sequences(None)
        finally:
            module.DisaggregatedRayTrainer, agent_module.AsyncLLMServerManager = original, original_manager

    def test_worker_hook_checks_runtime_from_foreign_cwd_and_denies_inference(self):
        import asyncio
        import verl.trainer.main_textarena_disagg_rsft as module
        import verl.experimental.agent_loop.agent_loop as agent_module
        from ours.recovered_alternation_bootstrap import install_recovery, RecordedAlternationBatch
        original, original_manager = module.DisaggregatedRayTrainer, agent_module.AsyncLLMServerManager
        def initialize_fixture_worker(trainer):
            trainer.async_rollout_manager = object()
        try:
            install_recovery()
            trainer = module.DisaggregatedRayTrainer.__new__(module.DisaggregatedRayTrainer)
            trainer.config = deepcopy(self.config)
            with tempfile.TemporaryDirectory(prefix='whale-recovery-cwd-') as directory, chdir(directory), \
                 patch.dict(os.environ, {'WHALE_ALTERNATION_REPLAY_PLAN': str(self.plan_path), 'SLURM_JOB_ID': 'cpu-preflight'}), \
                 patch.object(original, 'init_workers', initialize_fixture_worker):
                trainer.init_workers()
                self.assertIsInstance(trainer.async_rollout_manager, RecordedAlternationBatch)
                self.assertEqual(len(trainer.async_rollout_manager.batch), 64)
                manager = agent_module.AsyncLLMServerManager.__new__(agent_module.AsyncLLMServerManager)
                for method in (manager.chat_completion, manager.generate):
                    with self.assertRaisesRegex(RuntimeError, 'Fresh inference is forbidden'):
                        asyncio.run(method())
        finally:
            module.DisaggregatedRayTrainer, agent_module.AsyncLLMServerManager = original, original_manager

    def test_full_config_rejects_backend_or_objective_drift(self):
        from omegaconf import OmegaConf
        from ours.alternation_recovery import check_configuration
        actual = OmegaConf.to_container(self.config, resolve=True)
        check_configuration(self.plan, actual, 'alternation-replay-cpu-preflight', runtime=True)
        mutations = [(('actor_rollout_ref', 'model', 'use_fused_kernels'), False),
            (('actor_rollout_ref', 'actor', 'use_fused_kernels'), False),
            (('actor_rollout_ref', 'model', 'fused_kernel_options', 'impl_backend'), 'triton'),
            (('trainer', 'online_rsft', 'sft_mini_batch_size'), 11),
            (('trainer', 'online_rsft', 'score_threshold'), .0),
            (('ray_kwargs', 'ray_init', 'runtime_env', 'env_vars', 'WHALE_RSFT_FUSED_CHUNK_SIZE'), '512'),
            (('ray_kwargs', 'ray_init', 'runtime_env', 'worker_process_setup_hook'), 'wrong.hook')]
        for keys, value in mutations:
            with self.subTest(field=keys):
                wrong = deepcopy(actual)
                target = wrong
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    check_configuration(self.plan, wrong, 'alternation-replay-cpu-preflight', runtime=True)


if __name__ == '__main__':
    unittest.main()
