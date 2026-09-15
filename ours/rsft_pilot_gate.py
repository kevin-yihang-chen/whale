"""Require a completed nonempty bootstrap audit and a frozen native RSFT plan."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess

from .local_completion import checkpoint_manifest
from .preflight import PIN
from .visual_task import file_sha256


def check_tensor_bucket(model, actor_dtype, bucket_mebibytes):
    """Check original whole-tensor transport using headers, without loading weights."""
    from safetensors import safe_open
    dtype_bytes = {'fp32': 4, 'float32': 4, 'bf16': 2, 'bfloat16': 2, 'fp16': 2, 'float16': 2}
    if actor_dtype not in dtype_bytes:
        raise ValueError(f'Unreviewed actor transport dtype: {actor_dtype}')
    tensors = []
    for path in Path(model).glob('*.safetensors'):
        with safe_open(str(path), framework='pt', device='cpu') as reader:
            for name in reader.keys():
                shape = reader.get_slice(name).get_shape()
                tensors.append((math.prod(shape) * dtype_bytes[actor_dtype], name))
    if not tensors:
        raise ValueError('No checkpoint tensors found for transport sizing')
    nbytes, name = max(tensors)
    if nbytes > int(bucket_mebibytes) * 2**20:
        raise ValueError(f'Native checkpoint bucket cannot hold {name}: {nbytes} bytes')
    return {'largest_tensor': name, 'actor_tensor_bytes': nbytes,
            'bucket_bytes': int(bucket_mebibytes) * 2**20, 'actor_dtype': actor_dtype}


def check(path):
    plan = json.loads(path.read_text())
    if plan['kind'] != 'native_rsft_pilot_plan' or plan['veto_mode'] != 'off':
        raise ValueError('A frozen E4 training plan is required')
    upstream = 'upstream/WHALE'
    if subprocess.check_output(['git', '-C', upstream, 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise ValueError('Wrong WHALE revision')
    if subprocess.check_output(['git', '-C', upstream, 'status', '--porcelain'], text=True).strip():
        raise ValueError('Modified upstream')
    for name, expected in plan['source_sha256'].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f'Changed training source or evidence: {name}')
    if {name: version(name) for name in plan['versions']} != plan['versions']:
        raise ValueError('Training runtime versions changed')
    # The native package hides optional ImportError exceptions. Import the
    # selected transport explicitly so missing dependencies fail on the CPU.
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
    from verl.checkpoint_engine import CheckpointEngineRegistry
    if CheckpointEngineRegistry.get('nccl') is not NCCLCheckpointEngine:
        raise ValueError('The original NCCL checkpoint engine is not registered')
    audit = json.loads(Path(plan['bootstrap_audit']).read_text())
    if audit['kind'] != 'independent_chess_bootstrap_audit' or audit['status'] != 'PASS':
        raise ValueError('A complete independent bootstrap audit is required')
    if audit['trajectories'] != 64 or audit['solved'] <= 0 or audit['chunk'] != 0:
        raise ValueError('The declared first bootstrap chunk must be complete and nonempty')
    if audit['job_id'] != plan['bootstrap_job_id']:
        raise ValueError('Unexpected bootstrap job')
    if file_sha256(Path(plan['bootstrap_archive']) / 'result.json') != audit['result_sha256']:
        raise ValueError('Bootstrap result identity differs')
    if checkpoint_manifest(Path(plan['model']))['weights_sha256'] != plan['weights_sha256']:
        raise ValueError('Base checkpoint differs from the audited bootstrap')
    if plan['fresh_trajectories'] != 64 or plan['native_rsft_iterations'] != 1:
        raise ValueError('The first native update must use the declared 64-trajectory single step')
    from omegaconf import OmegaConf
    reviewed = OmegaConf.load(plan['resolved_config'])
    check_tensor_bucket(plan['model'], reviewed.actor_rollout_ref.actor.fsdp_config.model_dtype,
                        reviewed.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--effective-config', type=Path)
    args = parser.parse_args()
    plan = check(args.plan)
    if not args.check_only:
        import torch
        if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 2:
            raise RuntimeError('The first native pilot requires a two-GPU Slurm allocation')
        if Path(os.environ['RSFT_BASE_MODEL']).resolve() != Path(plan['model']).resolve():
            raise ValueError('Launcher model differs from the reviewed plan')
        if Path(os.environ['RSFT_TRAIN_FILES']).resolve() != Path(plan['train_dataset']).resolve():
            raise ValueError('Launcher dataset differs from the reviewed plan')
        if args.effective_config is None:
            raise ValueError('The actual resolved training configuration must be checked')
    if args.effective_config is not None:
        from omegaconf import OmegaConf
        raw = args.effective_config.read_text()
        marker = 'actor_rollout_ref:\n'
        actual = OmegaConf.create(raw[raw.index(marker):])
        expected_text = Path(plan['resolved_config']).read_text()
        if not args.check_only:
            expected_text = expected_text.replace('pilot-cpu-preflight', f"pilot-{os.environ['SLURM_JOB_ID']}")
        expected = OmegaConf.create(expected_text)
        if OmegaConf.to_container(actual, resolve=True) != OmegaConf.to_container(expected, resolve=True):
            raise ValueError('Effective Hydra configuration differs from the reviewed native pilot')
    report = {'kind': 'native_rsft_pilot_start_gate', 'status': 'PASS',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': file_sha256(args.plan), 'job_id': os.environ.get('SLURM_JOB_ID'),
              'training_steps_so_far': 0, 'check_only': args.check_only,
              'effective_config_sha256': file_sha256(args.effective_config) if args.effective_config else None}
    if not args.check_only:
        with Path(f"results/native-rsft-pilot-start-{report['job_id']}.json").open('x') as stream:
            stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
