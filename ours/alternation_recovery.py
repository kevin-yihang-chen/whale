"""Freeze a recorded E4 replay using WHALE's original Torch fused-output backend."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

from .alternation_training import environment, read, write_new
from .audit_native_training_batch import require
from .visual_task import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CPU_NAME = 'alternation-replay-cpu-preflight'
HOOK = 'ours.recovered_alternation_bootstrap.prepare_worker'
OVERRIDES = ('actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}',
    f'ray_kwargs.ray_init.runtime_env.worker_process_setup_hook={HOOK}',
    'actor_rollout_ref.model.use_fused_kernels=true',
    'actor_rollout_ref.actor.use_fused_kernels=true',
    'actor_rollout_ref.model.fused_kernel_options.impl_backend=torch')


def parsed(raw):
    from omegaconf import OmegaConf
    return OmegaConf.create(raw[raw.index('model_engine: dp\n'):])


def resolve(env, plan_path):
    from omegaconf import OmegaConf
    args = list(OVERRIDES) + [f'+ray_kwargs.ray_init.runtime_env.env_vars.WHALE_ALTERNATION_REPLAY_PLAN={plan_path.resolve()}',
                            "+ray_kwargs.ray_init.runtime_env.env_vars.WHALE_RSFT_FUSED_CHUNK_SIZE='128'"]
    try:
        raw = subprocess.check_output(['bash', str(ROOT / 'ours/run_native_rsft.sh'), *args,
            '--cfg', 'job', '--resolve'], cwd=ROOT, env=env, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'Native configuration resolution failed:\n{error.output}') from error
    return raw, OmegaConf.to_container(parsed(raw), resolve=True), args


def check_configuration(plan, actual, run_name, *, runtime=False):
    from omegaconf import OmegaConf
    from .recovery_gate import native_runtime_config
    raw = (ROOT / plan['resolved_config']).read_text().replace(CPU_NAME, run_name)
    config = parsed(raw)
    if runtime:
        config = native_runtime_config(config)
    require(actual == OmegaConf.to_container(config, resolve=True), 'Entire recovery configuration differs')


def check_source(source_plan, batch_audit, result_path):
    from .alternation_training import check as original_check
    source = original_check(source_plan)
    audit, result = read(batch_audit), read(result_path)
    require(audit['kind'] == 'native_alternation_batch_audit' and audit['status'] == 'PASS', 'Complete batch audit required')
    require(audit['plan_sha256'] == result['plan_sha256'] == file_sha256(source_plan), 'Wrong failed phase')
    require(result['status'] == 'FAILED' and result['native_checkpoint_exists'] is False,
            'Recovery requires the reviewed failed phase without a checkpoint')
    require(result['job_id'] == audit['job_id'] == '222292', 'Unexpected source job')
    require(audit['accepted'] == result['accepted'] == 11 and audit['accepted_loss_tokens'] == result['loss_tokens'] == 30392,
            'Unexpected recovery objective')
    require(audit['audit_source_sha256'] == file_sha256(ROOT / 'ours/audit_alternation_training.py'), 'Audit source changed')
    for hashes in (audit['artifact_sha256'], audit['audit_dependency_sha256']):
        for name, digest in hashes.items():
            require(file_sha256(ROOT / name) == digest, f'Audited evidence changed: {name}')
    accounting = audit['request_accounting']
    require(accounting['observed_requests_fully_accounted'] and accounting['completed'] == 69,
            'Source calls are incomplete')
    return source, audit


def prepare(path, source_plan, batch_audit, result_path):
    from omegaconf import OmegaConf
    require(not path.exists() and not path.with_suffix('.yaml').exists(), 'Preserve existing recovery plan')
    source, audit = check_source(source_plan, batch_audit, result_path)
    env = environment(source['model'], source['harness'], source['dataset'], CPU_NAME)
    raw, actual, _ = resolve(env, path)
    # Derive the permitted changes from the original complete native configuration.
    expected = parsed((ROOT / source['resolved_config']).read_text().replace('alternation-cpu-preflight', CPU_NAME))
    expected.actor_rollout_ref.model.use_fused_kernels = True
    expected.actor_rollout_ref.actor.use_fused_kernels = True
    expected.actor_rollout_ref.model.fused_kernel_options = {'impl_backend': 'torch'}
    expected.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook = HOOK
    expected.ray_kwargs.ray_init.runtime_env.env_vars.WHALE_ALTERNATION_REPLAY_PLAN = str(path.resolve())
    expected.ray_kwargs.ray_init.runtime_env.env_vars.WHALE_RSFT_FUSED_CHUNK_SIZE = '128'
    require(actual == OmegaConf.to_container(expected, resolve=True), 'Undeclared recovery configuration change')
    config_path = path.with_suffix('.yaml')
    config_path.write_text(raw)
    sources = set(source['source_sha256'])
    sources.update(map(str, [source_plan, batch_audit, result_path, config_path]))
    sources.update(('ours/alternation_recovery.py', 'ours/recovered_alternation_bootstrap.py',
        'ours/run_alternation_recovery.sh', 'ours/audit_alternation_training.py', 'ours/recovery_gate.py',
        'ours/native_fused_chunk.py', 'ours/tests/test_native_fused_chunk.py',
        'ours/native_transport_memory.py', 'ours/tests/test_native_transport_memory.py',
        'results/native-transport-memory-cpu-tests-20260909.log',
        'upstream/WHALE/domains/chess_puzzles/verl/checkpoint_engine/nccl_checkpoint_engine.py',
        'upstream/WHALE/domains/chess_puzzles/verl/checkpoint_engine/base.py',
        'results/alternation-recovery-222615/result.json',
        'results/alternation-recovery-222512/result.json', 'results/native-fused-chunk-cpu-tests-20260909.log',
        'ours/tests/test_alternation_recovery.py', 'ours/tests/test_native_fused_rsft.py',
        'results/native-fused-rsft-cpu-validation-20260909.json', 'results/native-fused-rsft-cpu-tests-20260909.log'))
    fused = read(ROOT / 'results/native-fused-rsft-cpu-validation-20260909.json')
    for name, digest in fused['source_sha256'].items():
        require(file_sha256(ROOT / name) == digest, f'Fused backend fixture source changed: {name}')
    sources.update(fused['source_sha256'])
    sources.update(audit['artifact_sha256'])
    sources.update(audit['audit_dependency_sha256'])
    plan = {'kind': 'native_alternation_recovery_plan', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_plan': str(source_plan), 'source_plan_sha256': file_sha256(source_plan),
        'batch_audit': str(batch_audit), 'batch_audit_sha256': file_sha256(batch_audit),
        'source_result': str(result_path), 'source_job_id': audit['job_id'],
        'batch_directory': f"data/native-rsft/alternation-{audit['job_id']}/audit", 'batch_sha256': audit['batch_sha256'],
        'model': source['model'], 'harness': source['harness'], 'dataset': source['dataset'],
        'cached_trajectories': 64, 'accepted': 11, 'loss_tokens': 30392, 'source_model_calls': 69,
        'expected_optimizer_steps': 2, 'new_model_calls': 0, 'native_fused_chunk_size': 128,
        'release_idle_cupy_after_native_finalize': True,
        'previous_attempt': 'results/alternation-recovery-222615/result.json', 'resolved_config': str(config_path),
        'source_sha256': {name: file_sha256(ROOT / name) for name in sorted(sources)},
        'versions': {**source['versions'], 'cupy-cuda13x': version('cupy-cuda13x')},
        'resources': {'gpus': 2, 'gpu_type': 'H800', 'cpus': 16, 'ram_gib': 160,
                      'time_limit_seconds': 1200, 'max_gpu_hours': 2/3},
        'limitations': ['Replay the complete audited batch from its incoming HF checkpoint, not lost in-memory optimizer state.',
            'Use original Torch fused-output backend; BF16 is not bitwise identical to the dense path.',
            'Bind the original fused autograd operator to128 tokens per chunk; SFT mini8/micro1 remain unchanged.',
            'Release only idle CuPy pool blocks after original NCCL finalize; GPU release is measured at runtime.',
            'Use a common backend across formal controls. CPU equivalence is not 4B/FSDP GPU proof.',
            'Cached source throughput is not inference performance; source model calls stay charged to222292.',
            'No VETO or heldout result is established by this engineering recovery.']}
    write_new(path, plan)
    return plan


def check(path):
    plan = read(path)
    require(plan['kind'] == 'native_alternation_recovery_plan' and plan['new_model_calls'] == 0, 'Wrong recovery plan')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(ROOT / name) == digest, f'Frozen recovery source changed: {name}')
    require({name: version(name) for name in plan['versions']} == plan['versions'], 'Runtime changed')
    source, audit = check_source(ROOT / plan['source_plan'], ROOT / plan['batch_audit'], ROOT / plan['source_result'])
    require(plan['model'] == source['model'] and plan['harness'] == source['harness'] and
            plan['dataset'] == source['dataset'] and plan['batch_sha256'] == audit['batch_sha256'], 'Recovery handoff differs')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-plan', type=Path)
    parser.add_argument('--batch-audit', type=Path)
    parser.add_argument('--source-result', type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(not args.run and all((args.source_plan, args.batch_audit, args.source_result)), 'Specify source evidence')
        plan = prepare(args.plan, args.source_plan, args.batch_audit, args.source_result)
    else:
        plan = check(args.plan)
    run_name = CPU_NAME
    if args.run:
        import torch
        require(os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 2, 'Expected two-GPU Slurm recovery')
        require(all('H800' in torch.cuda.get_device_name(i) for i in range(2)), 'Wrong allocated GPU type')
        run_name = f"alternation-replay-{os.environ['SLURM_JOB_ID']}"
        require(not (ROOT / 'data/native-rsft' / run_name).exists(), 'Recovery destination already exists')
        from .native_transport_memory import gpu_probe
        gpu_probe()
    env = environment(plan['model'], plan['harness'], plan['dataset'], run_name)
    raw, actual, command_args = resolve(env, args.plan)
    check_configuration(plan, actual, run_name)
    report = {'kind': 'native_alternation_recovery_preflight', 'status': 'PASS',
              'plan_sha256': file_sha256(args.plan), 'run_name': run_name, 'new_model_calls': 0}
    print(json.dumps(report), flush=True)
    if args.run:
        write_new(ROOT / f"results/alternation-recovery-start-{os.environ['SLURM_JOB_ID']}.json", report)
        (ROOT / f"results/alternation-recovery-effective-{os.environ['SLURM_JOB_ID']}.log").write_text(raw)
        os.execvpe('bash', ['bash', str(ROOT / 'ours/run_native_rsft.sh'), *command_args], env)
