"""Check fresh-recording inheritance and original native actor dispatch on CPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires native training runtime')
class CompactRecordedTrainingTests(unittest.TestCase):
    def test_original_actor_dispatch_keeps_masked_objective_and_online_manager(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import torch
        from omegaconf import OmegaConf
        from verl import DataProto
        import verl.trainer.main_textarena_disagg_rsft as trainer_module
        import verl.experimental.agent_loop.agent_loop as agent_module
        from ours.compact_recorded_training_bootstrap import install_compact_recorded_training
        original_trainer, original_manager = trainer_module.DisaggregatedRayTrainer, agent_module.AsyncLLMServerManager
        try:
            install_compact_recorded_training()
            selected = trainer_module.DisaggregatedRayTrainer
            install_compact_recorded_training()
            self.assertIs(selected, trainer_module.DisaggregatedRayTrainer)
            self.assertTrue(issubclass(selected, original_trainer))
            self.assertTrue(issubclass(agent_module.AsyncLLMServerManager, original_manager))
            self.assertFalse(getattr(selected, '_audited_batch_recovery', False))
            prompts = torch.tensor([[0, 0, 2, 3], [0, 0, 4, 5]])
            responses = torch.tensor([[6, 7, 8, 0, 0, 0], [9, 10, 0, 0, 0, 0]])
            attention = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]])
            loss = torch.tensor([[1, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
            positions = (attention.cumsum(-1) - 1).clamp(min=0) * attention
            data = DataProto.from_dict(tensors={'prompts': prompts, 'responses': responses,
                'input_ids': torch.cat([prompts, responses], -1), 'attention_mask': attention,
                'position_ids': positions, 'response_mask': loss,
                'dummy_tensor': torch.zeros(2, 1, dtype=torch.uint8)},
                meta_info={'sft_mini_batch_size': 8, 'sft_micro_batch_size_per_gpu': 1})
            observed = []
            class ActorFixture:
                _dispatch_info = {'actor': [0]}
                def update_sft_actor(self, batch):
                    observed.append(batch)
                    return 'fixture-original-dispatch-return'
            with tempfile.TemporaryDirectory(prefix='whale-compact-online-') as root:
                trainer = selected.__new__(selected)
                trainer.global_steps = 1
                trainer.config = OmegaConf.create({'trainer': {'default_local_dir': root},
                    'actor_rollout_ref': {'rollout': {'multi_turn': {'enable': True}, 'temperature': 1.}}})
                trainer.actor_rollout_wg = ActorFixture()
                manager = object()
                trainer.async_rollout_manager = manager
                self.assertEqual(trainer._update_sft_actor(data), 'fixture-original-dispatch-return')
                self.assertIs(trainer.async_rollout_manager, manager)
                received, = observed
                self.assertEqual(tuple(received.batch['responses'].shape), (2, 3))
                torch.testing.assert_close(received.batch['prompts'], prompts)
                torch.testing.assert_close(received.batch['response_mask'], loss[:, :3])
                self.assertEqual(received.meta_info['sft_mini_batch_size'], 8)
                self.assertEqual(received.meta_info['sft_micro_batch_size_per_gpu'], 1)
                self.assertEqual(json.loads((Path(root) / 'compaction/step-1.json').read_text())['loss_tokens'], 4)
        finally:
            trainer_module.DisaggregatedRayTrainer = original_trainer
            agent_module.AsyncLLMServerManager = original_manager


if __name__ == '__main__':
    unittest.main()
