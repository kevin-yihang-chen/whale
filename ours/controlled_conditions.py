"""The four F0 controls change optimization coordinates, not shared execution.

E4 updates theta when weights=True; E3 searches h when search=True. WHALE-FST
restricts h to the released prompt subspace while keeping the same E4 updates.
This module configures native contexts; it does not claim completed trials.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from pathlib import Path
import os
from unittest.mock import patch


@dataclass(frozen=True)
class OptimizationCondition:
    name: str
    weights: bool
    search: bool
    prompt_only: bool


CONDITIONS = {
    'weight_only': OptimizationCondition('weight_only', True, False, False),
    'harness_only': OptimizationCondition('harness_only', False, True, False),
    'whale_fst': OptimizationCondition('whale_fst', True, True, True),
    'whale': OptimizationCondition('whale', True, True, False),
}


def native_training_overrides(seed):
    from .experiment_randomization import TrialRandomization
    return ['actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}',
        'actor_rollout_ref.model.use_fused_kernels=true',
        'actor_rollout_ref.actor.use_fused_kernels=true',
        'actor_rollout_ref.model.fused_kernel_options.impl_backend=torch',
        'ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=ours.controlled_training_bootstrap.prepare_worker',
        *TrialRandomization(seed).hydra_overrides()]


@contextmanager
def native_search_context(condition, root, incoming_harness):
    from .mh_phase import native_context
    from .prompt_subspace import prompt_subspace
    specification = CONDITIONS[condition]
    if not specification.search:
        raise ValueError('The weight-only condition must not search a harness')
    incoming = Path(incoming_harness).resolve()
    if not incoming.is_file():
        raise ValueError('Incoming harness is missing')
    with ExitStack() as stack:
        native, benchmark = stack.enter_context(native_context({}, root))
        stack.enter_context(patch.dict(os.environ, BASELINE_HARNESS_OVERRIDE=str(incoming)))
        provenance = {'condition': condition, 'weight_updates': specification.weights,
                      'search_subspace': 'prompts' if specification.prompt_only else 'full_harness'}
        if specification.prompt_only:
            provenance['prompt_contract'] = stack.enter_context(prompt_subspace(native))
        yield native, benchmark, provenance
