"""Native seed propagation and process-start RNG checks; no task scores."""
import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.experiment_randomization import TrialRandomization


class SeedContractTests(unittest.TestCase):
    def test_invalid_seed_types_and_ranges(self):
        for seed in (True, -1, 2**31, 1.5, '42'):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                TrialRandomization(seed)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires native training runtime')
class NativeSeedTests(unittest.TestCase):
    def test_fresh_process_rng_streams_and_idempotence(self):
        code = '''
import json,random
import numpy as np
import torch
from ours.experiment_randomization import initialize_process_randomness
r=initialize_process_randomness()
a=random.random(); again=initialize_process_randomness(); b=random.random()
expected=random.Random(r['trial_seed'])
assert [a,b]==[expected.random(),expected.random()] and again['already_initialized']
r['draws']=[a,b,np.random.random(4).tolist(),torch.rand(4).tolist()]
print(json.dumps(r))
'''
        reports = [json.loads(subprocess.check_output([sys.executable, '-c', code], text=True,
                   env={**os.environ, **TrialRandomization(seed).environment()})) for seed in (42, 42, 43, 44)]
        self.assertEqual(reports[0], reports[1])
        for key in ('python_rng_sha256', 'numpy_rng_sha256', 'torch_cpu_rng_sha256', 'python_hash_probe'):
            self.assertEqual(len({r[key] for r in reports[1:]}), 3)
        self.assertEqual([r['torch_cpu_initial_seed'] for r in reports], [42, 42, 43, 44])
        print(json.dumps({'kind': 'trial_process_rng_fixture', 'status': 'PASS',
                          'seeds': [42, 42, 43, 44], 'reports': reports}), flush=True)

    def test_late_hash_seed_change_is_rejected(self):
        code = '''
import os
os.environ.update(WHALE_TRIAL_SEED='43',PYTHONHASHSEED='43')
from ours.experiment_randomization import initialize_process_randomness
try: initialize_process_randomness()
except ValueError as e: assert 'after interpreter startup' in str(e)
else: raise AssertionError('Late hash seed change was accepted')
'''
        subprocess.run([sys.executable, '-c', code], env={**os.environ, **TrialRandomization(42).environment()}, check=True)

    def test_native_loader_and_server_cli_receive_each_seed(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        from omegaconf import OmegaConf
        from verl.utils.config import omega_conf_to_dataclass
        from verl.trainer.main_ppo import create_rl_sampler
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
        from vllm.platforms import current_platform
        from ours.alternation_recovery import parsed
        original = parsed(Path('results/alternation-recovery-plan-20260909-v3.yaml').read_text())
        orders, captured = [], []
        for seed in (42, 43, 44):
            config = deepcopy(original)
            config.data.seed = seed
            config.actor_rollout_ref.rollout.engine_kwargs.vllm.seed = seed
            orders.append(list(create_rl_sampler(config.data, list(range(128)))))
            self.assertEqual(orders[-1], list(create_rl_sampler(config.data, list(range(128)))))
            server = vLLMHttpServer.__new__(vLLMHttpServer)
            server.config = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
            server.model_config = SimpleNamespace(local_path=config.actor_rollout_ref.model.path,
                lora_rank=0, lora={}, trust_remote_code=True)
            server.node_rank, server.replica_rank, server.nnodes = 0, 0, 1
            server.profiler_controller = SimpleNamespace(config=None, tool_config=None)
            for name in ('_master_sock', '_dp_rpc_sock', '_dp_master_sock'):
                setattr(server, name, SimpleNamespace(close=lambda: None))
            async def capture(args):
                captured.append({'seed': args.seed, 'model': args.model, 'dtype': args.dtype,
                                 'gdn_prefill_backend': args.gdn_prefill_backend})
            server.run_server = capture
            # The login node has no CUDA driver. Only device discovery is a CPU
            # fixture; native CLI construction, parsing and validation run intact.
            with patch.object(current_platform, 'device_type', 'cpu'):
                asyncio.run(server.launch_server())
        self.assertEqual(len({tuple(order) for order in orders}), 3)
        self.assertEqual([row['seed'] for row in captured], [42, 43, 44])
        print(json.dumps({'kind': 'native_trial_seed_cli_fixture', 'status': 'PASS',
                          'device_discovery': 'CPU fixture; no engine started',
                          'native_cli': captured, 'native_loader_first8': [r[:8] for r in orders]}), flush=True)


if __name__ == '__main__':
    unittest.main()
