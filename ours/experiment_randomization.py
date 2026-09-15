"""Shared E3/E4 trial randomization, separate from the visual acceptance rule.

Legacy RSFT preserves accepted-row order; actor.data_loader_seed does not
shuffle that update. Training chat requests omit seed and use the vLLM engine
stream. MH requests carry their configured seed. Provider randomness is not
controlled by these local seeds, and asynchronous GPU execution is not bitwise
reproducible merely because the declared seeds agree.
"""
from dataclasses import dataclass
import hashlib
import os
import pickle
import random
import subprocess
import sys


TRIAL_SEEDS = (42, 43, 44)
HASH_PROBE = 'WHALE/F0/randomization'
_INITIALIZED = None


@dataclass(frozen=True)
class TrialRandomization:
    seed: int

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('Trial seed must be a nonnegative 31-bit integer')

    def hydra_overrides(self):
        return [f'data.seed={self.seed}',
            f'actor_rollout_ref.actor.data_loader_seed={self.seed}',
            f'actor_rollout_ref.actor.fsdp_config.seed={self.seed}',
            f'+actor_rollout_ref.rollout.engine_kwargs.vllm.seed={self.seed}',
            f"+ray_kwargs.ray_init.runtime_env.env_vars.WHALE_TRIAL_SEED='{self.seed}'",
            f"+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONHASHSEED='{self.seed}'"]

    def environment(self):
        # Must be applied before launching Python, including the Ray driver.
        return {'WHALE_TRIAL_SEED': str(self.seed), 'PYTHONHASHSEED': str(self.seed)}

    def metadata(self):
        return {'trial_seed': self.seed, 'data_sampler_seed': self.seed,
                'process_rng_seed': self.seed, 'training_vllm_engine_seed': self.seed,
                'mh_request_seed': self.seed, 'training_request_seed': None,
                'legacy_sft_minibatches': 'accepted order, no shuffle',
                'proposer_seed_controlled': False, 'bitwise_gpu_reproducibility_claimed': False}


def initialize_process_randomness():
    """Initialize once per worker and verify hash randomization at process start."""
    global _INITIALIZED
    trial = TrialRandomization(int(os.environ['WHALE_TRIAL_SEED']))
    if os.environ.get('PYTHONHASHSEED') != str(trial.seed):
        raise ValueError('PYTHONHASHSEED must be exported before Python starts')
    expected = int(subprocess.check_output([sys.executable, '-c',
        'import sys; print(hash(sys.argv[1]))', HASH_PROBE], text=True))
    if hash(HASH_PROBE) != expected:
        raise ValueError('Python hash seed was changed after interpreter startup')
    if _INITIALIZED is not None:
        if _INITIALIZED != trial.seed:
            raise ValueError('A worker cannot change trial seed midway through execution')
        return {'trial_seed': trial.seed, 'already_initialized': True}
    import numpy as np
    import torch
    random.seed(trial.seed)
    np.random.seed(trial.seed)
    torch.manual_seed(trial.seed)
    _INITIALIZED = trial.seed
    digest = lambda value: hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()
    return {**trial.metadata(), 'python_hash_probe': hash(HASH_PROBE),
            'python_rng_sha256': digest(random.getstate()),
            'numpy_rng_sha256': digest(np.random.get_state()),
            'torch_cpu_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'torch_cpu_initial_seed': torch.initial_seed(),
            'cuda_initialized': torch.cuda.is_initialized(),
            'cuda_seed_note': 'manual_seed queues CUDA seed initialization; live vLLM worker checked separately'}
