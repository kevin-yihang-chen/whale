"""Record fresh native E4 sampling and trim only masked response tails for SFT.

This hook keeps online generation, native success filtering and the original
actor/optimizer/save/transport. It does not install the cached recovery manager.
"""
import json
from pathlib import Path


def install_compact_recorded_training():
    from .native_training_trace import install_native_recorders
    import verl.trainer.main_textarena_disagg_rsft as module
    install_native_recorders()
    if getattr(module.DisaggregatedRayTrainer, '_compact_recorded_training', False):
        return
    class CompactRecordedTrainer(module.DisaggregatedRayTrainer):
        _compact_recorded_training = True

        def _update_sft_actor(self, batch):
            from .compact_native_batch import compact_batch
            compact, report = compact_batch(batch)
            destination = Path(self.config.trainer.default_local_dir) / 'compaction'
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / f'step-{self.global_steps}.json'
            with path.open('x') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
            print(json.dumps(report), flush=True)
            return super()._update_sft_actor(compact)
    module.DisaggregatedRayTrainer = CompactRecordedTrainer


def prepare_worker():
    from .training_bootstrap import prepare_worker as prepare_original_worker
    provenance = prepare_original_worker()
    install_compact_recorded_training()
    return {'native_package_repair': provenance, 'native_chess_recording': True,
            'masked_response_tail_compaction': True, 'cached_generation_recovery': False}
