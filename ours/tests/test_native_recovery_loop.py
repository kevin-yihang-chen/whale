"""Run the real dataset and trainer loop; stub only GPU RPC/checkpoint boundaries.

This is an integration fixture with zero model/optimizer calls. It cannot establish
GPU training or checkpoint success. The actual accepted actor input is inspected.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Requires native training runtime')
class NativeRecoveryLoopTests(unittest.TestCase):
    def test_original_dataset_trainer_and_actor_entry(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import torch
        from omegaconf import OmegaConf
        from transformers import AutoProcessor, AutoTokenizer
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl import DataProto
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn
        import verl.trainer.main_textarena_disagg_rsft as module
        import verl.experimental.agent_loop.agent_loop as agent_module
        from ours.recovery_gate import native_runtime_config
        from ours.recovered_training_bootstrap import restore_batch, install_recovery, AuditedBatchRecovery
        plan_path = Path('results/native-rsft-recovery-plan-20260909-v2.json')
        if not plan_path.exists():
            self.skipTest('Recorded batch recovery plan not present')
        plan = json.loads(plan_path.read_text())
        config = native_runtime_config(OmegaConf.load(plan['resolved_config']))
        tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
        processor = AutoProcessor.from_pretrained(plan['model'], local_files_only=True)
        dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor,
                                    is_train=True, max_samples=-1)
        sampler = create_rl_sampler(config.data, dataset)
        loader = StatefulDataLoader(dataset, batch_size=8, num_workers=8, drop_last=True,
                                    collate_fn=collate_fn, sampler=sampler)
        batch, receipt = restore_batch(plan['batch_directory'], plan['batch_audit'])
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
            with tempfile.TemporaryDirectory(prefix='whale-native-loop-fixture-') as root:
                config.trainer.default_local_dir = root
                config.trainer.rollout_data_dir = str(Path(root) / 'rollouts')
                trainer.config, trainer.tokenizer = config, tokenizer
                trainer.global_steps, trainer.total_training_steps = 0, 1
                trainer.train_dataset, trainer.train_dataloader = dataset, loader
                trainer.use_rm = trainer.use_critic = False
                trainer.actor_rollout_wg = FixtureActorGroup()
                trainer.async_rollout_manager = AuditedBatchRecovery(object(), batch, receipt,
                    Path(root) / 'recovery', plan)
                trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None,
                    update_weights=lambda step: syncs.append(step))
                trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 2)
                trainer._save_checkpoint = lambda: saves.append(trainer.global_steps)
                logger = SimpleNamespace(log=lambda **kwargs: logs.append(kwargs), finish=lambda: None)
                trainer._fit_online_rsft(logger)
                self.assertEqual(len(observed), 1)
                received = observed[0]
                self.assertEqual(received.batch['responses'].shape, (2, 8144))
                self.assertEqual(received.batch['prompts'].shape, (2, 4096))
                self.assertEqual(received.batch['dummy_tensor'].dtype, torch.uint8)
                self.assertEqual(int(received.batch['dummy_tensor'].sum()), 0)
                self.assertEqual(int(received.batch['response_mask'].sum()), 16258)
                self.assertEqual(received.meta_info['sft_mini_batch_size'], 8)
                self.assertEqual(received.meta_info['sft_micro_batch_size_per_gpu'], 1)
                self.assertEqual((saves, syncs), ([1], [1]))
                self.assertEqual(logs[-1]['data']['online_rsft/accepted'], 2)
                self.assertEqual(len((Path(root) / 'rollouts/1.jsonl').read_text().splitlines()), 64)
        finally:
            module.DisaggregatedRayTrainer = original
            agent_module.AsyncLLMServerManager = original_manager


if __name__ == '__main__':
    unittest.main()
