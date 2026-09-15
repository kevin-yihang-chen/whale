"""Synthetic CPU batches test full-pixel recording; no model or optimizer runs."""
from pathlib import Path
import tempfile
import unittest


class VisualTrainingRecordingTest(unittest.TestCase):
    def test_native_archive_keeps_pixels_masks_and_rejects_bad_rewards(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from verl import DataProto
        from ours.visual_training_bootstrap import install_visual_training_recorders
        import verl.trainer.main_textarena_disagg_rsft as module
        install_visual_training_recorders()
        batch = DataProto.from_single_dict({
            'token_level_scores': torch.tensor([[0., 1.], [0., 0.]]),
            'response_mask': torch.tensor([[1, 1], [1, 0]]),
            'multi_modal_inputs': np.asarray([{'pixel_values': torch.arange(12).reshape(4, 3)}
                                             for _ in range(2)], dtype=object),
            'visual_sample_id': np.asarray(['fixture-0', 'fixture-1'], dtype=object),
            'visual_harness_sha256': np.asarray(['fixture', 'fixture'], dtype=object),
            'visual_final_answer': np.asarray(['A', 'A'], dtype=object),
            'visual_committed_answer': np.asarray(['A', 'A'], dtype=object),
            'reward_model': np.asarray([{'ground_truth': 'A'}, {'ground_truth': 'B'}], dtype=object),
        })
        with tempfile.TemporaryDirectory() as directory:
            trainer = module.DisaggregatedRayTrainer.__new__(module.DisaggregatedRayTrainer)
            trainer.global_steps = 1
            trainer.config = OmegaConf.create({'trainer': {'default_local_dir': directory,
                                                          'online_rsft': {'score_threshold': .5}}})
            receipt = trainer._record_visual_batch(batch, 'rollout')
            restored = DataProto.load_from_disk(receipt['path'])
            self.assertTrue(torch.equal(restored.batch['response_mask'], batch.batch['response_mask']))
            self.assertTrue(torch.equal(restored.non_tensor_batch['multi_modal_inputs'][0]['pixel_values'],
                                        batch.non_tensor_batch['multi_modal_inputs'][0]['pixel_values']))
            self.assertEqual(receipt['scores'], [1., 0.])
            with self.assertRaisesRegex(ValueError, 'unsuccessful'):
                trainer._record_visual_batch(batch, 'accepted')
            self.assertFalse((Path(directory) / 'audit/accepted-step1.pkl').exists())
            successful = batch.select_idxs([0])
            self.assertEqual(trainer._record_visual_batch(successful, 'accepted')['examples'], 1)
            trainer.global_steps = 2
            batch.batch['token_level_scores'][1, 1] = 1.
            with self.assertRaisesRegex(ValueError, 'verifier'):
                trainer._record_visual_batch(batch, 'rollout')
            self.assertFalse((Path(directory) / 'audit/rollout-step2.pkl').exists())


if __name__ == '__main__':
    unittest.main()
