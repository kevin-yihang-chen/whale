"""Record E4's original visual RSFT batches without text-only compaction.

The native success filter and loss run unchanged. Full multimodal payloads are
retained before and after that filter; no stored evaluation trajectory is used
as training data. This is shared execution evidence, not a VETO mechanism.
"""
import json
import os
from pathlib import Path


def install_visual_training_recorders():
    import verl.trainer.main_textarena_disagg_rsft as module
    from .visual_task import binary_answer_verifier, file_sha256
    if getattr(module.DisaggregatedRayTrainer, '_visual_training_recorder', False):
        return

    class VisualTrajectoryRecordingTrainer(module.DisaggregatedRayTrainer):
        _visual_training_recorder = True

        def _record_visual_batch(self, batch, kind):
            directory = Path(self.config.trainer.default_local_dir) / 'audit'
            directory.mkdir(exist_ok=True)
            scores = batch.batch['token_level_scores'].sum(-1).tolist()
            meta = batch.non_tensor_batch
            required = {'multi_modal_inputs', 'visual_sample_id', 'visual_harness_sha256',
                        'visual_final_answer', 'visual_committed_answer', 'reward_model'}
            if not required.issubset(meta):
                raise ValueError('Visual training lost image, answer or sample evidence')
            for i, score in enumerate(scores):
                image = meta['multi_modal_inputs'][i]
                if not isinstance(image, dict) or 'pixel_values' not in image or not image['pixel_values'].numel():
                    raise ValueError('Visual training received an empty image payload')
                answer = meta['visual_committed_answer'][i]
                truth = meta['reward_model'][i]['ground_truth']
                if score != int(binary_answer_verifier(answer, truth)):
                    raise ValueError('Native visual reward differs from the shared verifier')
                if kind == 'accepted' and score <= self.config.trainer.online_rsft.score_threshold:
                    raise ValueError('Native RSFT sent an unsuccessful visual trajectory for training')
            path = directory / f'{kind}-step{self.global_steps}.pkl'
            if path.exists():
                raise FileExistsError(path)
            batch.save_to_disk(path)
            receipt = {'kind': kind, 'global_step': self.global_steps, 'examples': len(batch),
                'path': str(path), 'sha256': file_sha256(path), 'scores': scores,
                'sample_ids': meta['visual_sample_id'].tolist(),
                'assistant_loss_tokens': batch.batch['response_mask'].sum(-1).tolist(),
                'job_id': os.environ.get('SLURM_JOB_ID'), 'model_calls_by_recorder': 0,
                'limitation': 'A recorded batch or returned update is not proof of changed model parameters.'}
            with path.with_suffix('.json').open('x') as stream:
                json.dump(receipt, stream, indent=2)
            return receipt

        def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
            self._record_visual_batch(batch, 'rollout')
            return super()._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

        def _update_sft_actor(self, batch):
            receipt = self._record_visual_batch(batch, 'accepted')
            result = super()._update_sft_actor(batch)
            from .native_training_trace import json_value
            output = Path(self.config.trainer.default_local_dir) / 'audit' / f'update-step{self.global_steps}.json'
            with output.open('x') as stream:
                json.dump({'status': 'NATIVE_SFT_CALL_RETURNED', 'accepted_receipt': receipt,
                           'metrics': json_value(result.meta_info['metrics'])}, stream, indent=2, allow_nan=False)
            return result

    module.DisaggregatedRayTrainer = VisualTrajectoryRecordingTrainer


def prepare_worker():
    from .training_bootstrap import prepare_worker as original
    from .experiment_randomization import initialize_process_randomness
    from .native_fused_chunk import install_chunk_size
    from .native_transport_memory import install_transport_release
    provenance = original()
    randomness = initialize_process_randomness()
    install_chunk_size(128)
    install_transport_release()
    install_visual_training_recorders()
    return {'native_setup': provenance, 'randomness': randomness,
            'full_multimodal_batch_recording': True, 'text_only_compaction': False}
