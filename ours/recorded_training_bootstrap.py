"""Explicit Ray setup hook for future audited native Chess training runs.

Select ray_kwargs.ray_init.runtime_env.worker_process_setup_hook as
ours.recorded_training_bootstrap.prepare_worker in a new frozen plan. This is
not enabled for the currently running uninstrumented pilot.
"""


def prepare_worker():
    from .training_bootstrap import prepare_worker as prepare_original_worker
    from .native_training_trace import install_native_recorders
    provenance = prepare_original_worker()
    install_native_recorders()
    return {'native_package_repair': provenance, 'native_chess_recording': True}
