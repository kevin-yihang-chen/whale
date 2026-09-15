"""Shared fresh E4 execution for controlled trials; no cached trajectories."""
import json
import os


def prepare_worker():
    from .experiment_randomization import initialize_process_randomness
    from .compact_recorded_training_bootstrap import prepare_worker as original
    from .native_fused_chunk import install_chunk_size
    from .native_transport_memory import install_transport_release
    randomization = initialize_process_randomness()
    provenance = original()
    install_chunk_size(128)
    install_transport_release()
    report = {'kind': 'controlled_training_worker_setup', 'pid': os.getpid(),
        'randomization': randomization, 'shared_native_execution': provenance,
        'fused_token_chunk_size': 128, 'release_idle_cupy': True,
        'cached_generation_recovery': False}
    print(json.dumps(report), flush=True)
    return report
