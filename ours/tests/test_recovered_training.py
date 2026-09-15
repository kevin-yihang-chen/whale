"""Exercise cached E4 recovery without a server or new inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Requires native training runtime')
class RecoveredTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()

    def test_audited_batch_matches_original_sampler_and_reward_path(self):
        import numpy as np
        import pyarrow.parquet as pq
        from omegaconf import OmegaConf
        from verl import DataProto
        from verl.trainer.main_ppo import create_rl_sampler
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.trainer.ppo.reward import extract_reward
        from ours.compact_native_batch import compact_batch
        from ours.recovered_training_bootstrap import restore_batch, AuditedBatchRecovery
        directory = Path('data/native-rsft/pilot-221967/audit')
        if not directory.exists():
            self.skipTest('Recorded job221967 not present')
        batch, receipt = restore_batch(directory, 'results/native-batch-audit-221967.json')
        source = json.loads(Path('results/native-rsft-pilot-plan-20260909-v5.json').read_text())
        config = OmegaConf.load(source['resolved_config'])
        raw = pq.read_table(source['train_dataset'], use_threads=False).to_pylist()
        # StatefulDataLoader consumes sampler state differently with eight workers.
        # Use the released loader configuration, not a bare sampler iteration.
        indices = list(range(len(raw)))
        loader = StatefulDataLoader(indices, batch_size=8, num_workers=8, drop_last=True,
                                    sampler=create_rl_sampler(config.data, indices))
        order = next(iter(loader)).tolist()
        info = np.array([raw[i]['extra_info'] for i in order for _ in range(8)], dtype=object)
        uid = np.array([f'new-group-{i}' for i in order for _ in range(8)], dtype=object)
        incoming = DataProto.from_dict(tensors={'prompts': batch.batch['prompts'].clone()},
            non_tensors={'extra_info': info, 'uid': uid}, meta_info={'global_steps': 1})
        with tempfile.TemporaryDirectory(prefix='whale-native-recovery-') as root:
            wrapper = AuditedBatchRecovery(object(), batch, receipt, root, {'source_plan_sha256': 'fixture'})
            incoming.non_tensor_batch['extra_info'][0], incoming.non_tensor_batch['extra_info'][8] = info[8], info[0]
            with self.assertRaisesRegex(ValueError, 'order or content'):
                wrapper.generate_sequences(incoming)
            incoming.non_tensor_batch['extra_info'] = np.array([raw[i]['extra_info'] for i in order for _ in range(8)], dtype=object)
            output = wrapper.generate_sequences(incoming)
            union = incoming.union(output)
            rewards, details = extract_reward(union)
            self.assertEqual(int((rewards.sum(-1) > .5).sum()), 2)
            self.assertEqual(sum(details['policy_calls']), 69)
            accepted = union.select_idxs(rewards.sum(-1) > .5)
            # Native trainer adds these aliases before the shared actor adapter.
            for key in ['token_level_rewards', 'advantages', 'returns']:
                accepted.batch[key] = accepted.batch['token_level_scores'].clone()
            compact, report = compact_batch(accepted)
            self.assertEqual(report['response_after'], 8144)
            self.assertEqual(report['prompt_after'], 4096)
            self.assertEqual(report['loss_tokens'], 16258)
            self.assertEqual(json.loads((Path(root) / 'recovery.json').read_text())['new_model_calls'], 0)
            with self.assertRaisesRegex(ValueError, 'only once'):
                wrapper.generate_sequences(incoming)

    def test_runtime_migrations_match_recorded_configuration_hash(self):
        import gzip
        import hashlib
        from omegaconf import OmegaConf
        from ours.recovery_gate import native_runtime_config
        path = Path('results/native-rsft-pilot-effective-221967.log')
        if not path.exists():
            self.skipTest('Recorded job221967 not present')
        raw = path.read_text()
        complete = OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
        actual = OmegaConf.to_container(native_runtime_config(complete), resolve=True)
        with gzip.open('data/native-rsft/pilot-221967/audit/batch-1.jsonl.gz', 'rt') as f:
            expected = json.loads(next(f))['context']['configuration_sha256']
        self.assertEqual(hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(), expected)

    def test_actual_recovery_subclass_initializes_from_foreign_working_directory(self):
        from contextlib import chdir
        from unittest.mock import patch
        from omegaconf import OmegaConf
        from ours.recovery_gate import native_runtime_config, repository_path
        from ours.recovered_training_bootstrap import install_recovery, AuditedBatchRecovery
        import verl.trainer.main_textarena_disagg_rsft as module
        original = module.DisaggregatedRayTrainer
        plan_path = repository_path('results/native-rsft-recovery-plan-20260909.json')
        if not plan_path.exists():
            self.skipTest('Recorded recovery plan not present')
        plan = json.loads(plan_path.read_text())
        config = native_runtime_config(OmegaConf.load(repository_path(plan['resolved_config'])))
        def initialize_fixture_worker(trainer):
            trainer.async_rollout_manager = object()
        try:
            install_recovery()
            trainer = module.DisaggregatedRayTrainer.__new__(module.DisaggregatedRayTrainer)
            trainer.config = config
            with tempfile.TemporaryDirectory(prefix='whale-worker-cwd-') as directory, chdir(directory), \
                 patch.dict('os.environ', {'WHALE_RSFT_REPLAY_PLAN': str(plan_path)}), \
                 patch.object(original, 'init_workers', initialize_fixture_worker):
                trainer.init_workers()
                self.assertIsInstance(trainer.async_rollout_manager, AuditedBatchRecovery)
                self.assertEqual(len(trainer.async_rollout_manager.batch), 64)
        finally:
            module.DisaggregatedRayTrainer = original


if __name__ == '__main__':
    unittest.main()
