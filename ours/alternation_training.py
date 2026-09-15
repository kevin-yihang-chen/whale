"""Bind a completed native MH selection to the next fresh E4 training batch.

This is the shared h_{k+1} -> ModelUpdate(theta_{k+1}, h_{k+1}) connection.
It does not change the success-filtered SFT objective. The engineering cursor
and HF initialization are recorded separately from a full trainer-state resume.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

from .audit_native_training_batch import require
from .mh_phase import check_plan as check_mh_plan, verify_receipt
from .visual_task import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CPU_NAME = 'alternation-cpu-preflight'
OVERRIDES = (
    'actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}',
    'ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=ours.compact_recorded_training_bootstrap.prepare_worker',
)
SOURCES = (
    'ours/alternation_training.py', 'ours/run_alternation_rsft.sh',
    'ours/run_native_rsft.sh', 'ours/compact_recorded_training_bootstrap.py',
    'ours/native_training_trace.py', 'ours/compact_native_batch.py',
    'ours/training_bootstrap.py', 'ours/rsft_pilot_gate.py',
    'ours/training_resolved.lock', 'ours/chess_bootstrap.py',
    'ours/audit_mh_phase.py', 'ours/tests/test_compact_recorded_training.py',
    'ours/tests/test_alternation_training.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def selected_from_audits(audits):
    """Independent two-candidate check of the frozen native selection rule."""
    require(set(audits) == {'h0', 'h1'}, 'Both audited candidates are required')
    for name in ('h0', 'h1'):
        a = audits[name]
        require(a['status'] == 'PASS' and a['kind'] == 'mh_phase_audit' and
                a['harness'] == name and a['examples'] == 32, 'Incomplete MH audit')
        require(type(a['solved']) is int and 0 <= a['solved'] <= 32 and
                type(a['calls']) is int and a['calls'] >= 32, 'Invalid audited score/calls')
    # Original native early stop for a perfect proposed h1 precedes frontier ties.
    if audits['h1']['solved'] == 32:
        return 'h1'
    return min(audits, key=lambda h: (-audits[h]['solved'], audits[h]['calls'], h))


def checked_search(search_plan):
    phase = check_mh_plan(Path(search_plan))
    root = Path(phase['phase_root'])
    receipt = verify_receipt(root, 'search-complete.json', Path(search_plan))
    audits = {h: read(root / name) for h, name in
              [('h0', 'baseline-audit.json'), ('h1', 'candidate-audit.json')]}
    for a in audits.values():
        require(a['plan_sha256'] == file_sha256(Path(search_plan)), 'Different audited MH plan')
        require(a['audit_source_sha256'] == file_sha256(ROOT / 'ours/audit_mh_phase.py'), 'Changed audit implementation')
        for relative, digest in a['artifact_sha256'].items():
            path = root / relative
            require(path.resolve().is_relative_to(root) and file_sha256(path) == digest,
                    f'Changed audited MH artifact: {relative}')
    selected = selected_from_audits(audits)
    require(receipt['accepted'] == selected, 'Native selection differs from independently audited objectives')
    harness = root / f'search/harnesses/{selected}/harness.py'
    require(file_sha256(harness) == receipt['accepted_harness_sha256'], 'Selected harness changed')
    return phase, harness


def checked_data(data_report):
    from .chess_bootstrap import read_rows
    report = read(data_report)
    require(report['kind'] == 'next_native_rsft_training_batch' and
            report['engineering_cursor_before'] == 8 and report['engineering_cursor_after_if_completed'] == 16 and
            report['fresh_trajectories'] == 64 and report['rollout_n'] == 8, 'Unexpected engineering batch')
    for name, digest in report['source_sha256'].items():
        require(file_sha256(ROOT / name) == digest, f'Changed next-batch source: {name}')
    pilot = read(ROOT / 'results/chess-pilot-manifest-20260909.json')
    rows = sorted(read_rows(pilot['splits']['train']['path']), key=lambda r: r['extra_info']['puzzle_id'])
    require(read_rows(report['dataset']) == rows[8:16], 'Training cursor no longer selects the next eight rows')
    require(report['previous_ids'] == [r['extra_info']['puzzle_id'] for r in rows[:8]], 'Previous batch differs')
    require(report['puzzle_ids'] == [r['extra_info']['puzzle_id'] for r in rows[8:16]], 'Next batch IDs differ')
    forbidden = set(pilot['splits']['mh_val']['puzzle_ids'] + pilot['splits']['test']['puzzle_ids'])
    require(not forbidden.intersection(report['puzzle_ids']), 'Training crosses the MH/test boundary')
    return report


def environment(model, harness, dataset, run_name):
    env = os.environ.copy()
    env.update(RSFT_BASE_MODEL=str(Path(model).resolve()), RSFT_HARNESS_PATH=str(Path(harness).resolve()),
        RSFT_TRAIN_FILES=str(Path(dataset).resolve()), RSFT_VAL_FILES=str(Path(dataset).resolve()),
        RSFT_ROLLOUT_N='8', RSFT_AGENT_NUM_WORKERS='4', RSFT_GENERATE_TIMEOUT_S='4800',
        RSFT_TEST_FREQ='-1', RSFT_WEIGHT_BUCKET_MB='3072', RSFT_RUN_NAME=run_name)
    return env


def resolve_config(env):
    from omegaconf import OmegaConf
    output = subprocess.check_output(['bash', str(ROOT / 'ours/run_native_rsft.sh'),
        *OVERRIDES, '--cfg', 'job', '--resolve'], cwd=ROOT, env=env, text=True, stderr=subprocess.STDOUT)
    marker = 'model_engine: dp\n'
    require(marker in output, 'Native resolved configuration marker missing')
    config = OmegaConf.to_container(OmegaConf.create(output[output.index(marker):]), resolve=True)
    return output, config


def check_config(plan, actual, run_name):
    from omegaconf import OmegaConf
    raw = (ROOT / plan['resolved_config']).read_text().replace(CPU_NAME, run_name)
    expected = OmegaConf.to_container(OmegaConf.create(raw[raw.index('model_engine: dp\n'):]), resolve=True)
    require(actual == expected, 'Effective next-phase configuration differs from frozen configuration')


def prepare(search_plan, data_report, path):
    phase, harness = checked_search(search_plan)
    data = checked_data(data_report)
    require(data['model'] == phase['target_config']['model'], 'Different target checkpoint in batch preflight')
    require(not path.exists() and not path.with_suffix('.yaml').exists(), 'Preserve existing training plan/config')
    raw, config = resolve_config(environment(data['model'], harness, data['dataset'], CPU_NAME))
    require(config['trainer']['online_rsft']['iterations'] == 1 and
            config['actor_rollout_ref']['rollout']['n'] == 8, 'Unexpected online training budget')
    config_path = path.with_suffix('.yaml')
    config_path.write_text(raw)
    root = Path(phase['phase_root'])
    sources = set(SOURCES) | set(phase['source_sha256']) | set(data['source_sha256'])
    # Preserve the reviewed original training closure, without reusing the old
    # bootstrap gate or silently assuming its source hashes still match.
    previous = read(ROOT / 'results/native-rsft-pilot-plan-20260909-v5.json')
    sources.update(p for p in previous['source_sha256'] if p.startswith(('ours/', 'upstream/')))
    sources.update(map(str, [search_plan, data_report, config_path, harness,
        root / 'search-complete.json', root / 'baseline-audit.json', root / 'candidate-audit.json']))
    plan = {'kind': 'native_alternation_training_plan', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'search_plan': str(search_plan), 'data_report': str(data_report), 'model': data['model'],
        'harness': str(harness), 'dataset': data['dataset'], 'resolved_config': str(config_path),
        'veto_mode': 'off', 'fresh_trajectories': 64, 'native_rsft_iterations': 1,
        'source_sha256': {p: file_sha256(ROOT / p) for p in sorted(sources)},
        'versions': {n: version(n) for n in ('torch', 'vllm', 'transformers', 'ray', 'numpy', 'pandas', 'pyarrow', 'chess')},
        'resources': {'gpus': 2, 'gpu_type': 'H800', 'cpus': 16, 'ram_gib': 160, 'time_limit_seconds': 5400, 'max_gpu_hours': 3.},
        'limitations': ['Fresh on-policy engineering phase, not a completed F0 comparison or VETO result.',
            'Explicit engineering cursor advances to training IDs8:16. It is not an original full-sampler-state resume.',
            'Initialize from the canonical HF checkpoint. No claim of restoring prior optimizer, scheduler or RNG state.',
            'Keep the released Chess launcher model/extra checkpoint policy; it does not save optimizer state.',
            'An empty success-filtered set is a measured empty phase, not an optimizer-update success.',
            'H800 MH inference has passed; original distributed training on H800 is new live validation.']}
    write_new(path, plan)
    return plan


def check(path):
    plan = read(path)
    require(plan['kind'] == 'native_alternation_training_plan' and plan['veto_mode'] == 'off', 'Wrong training plan')
    require(set(SOURCES) <= set(plan['source_sha256']), 'Training source closure is incomplete')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(ROOT / name) == digest, f'Changed frozen training source: {name}')
    require({n: version(n) for n in plan['versions']} == plan['versions'], 'Training runtime changed')
    phase, harness = checked_search(ROOT / plan['search_plan'])
    data = checked_data(ROOT / plan['data_report'])
    require(plan['harness'] == str(harness) and plan['model'] == phase['target_config']['model'] and
            plan['model'] == data['model'] and plan['dataset'] == data['dataset'], 'Training handoff differs')
    require(plan['fresh_trajectories'] == 64 and plan['native_rsft_iterations'] == 1, 'Training budget differs')
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
    from verl.checkpoint_engine import CheckpointEngineRegistry
    require(CheckpointEngineRegistry.get('nccl') is NCCLCheckpointEngine, 'Native transport is not registered')
    from .rsft_pilot_gate import check_tensor_bucket
    check_tensor_bucket(plan['model'], 'float32', 3072)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--search-plan', type=Path)
    parser.add_argument('--data-report', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    require(not (args.prepare and args.run), 'Prepare and run are separate actions')
    if args.prepare:
        require(args.search_plan is not None and args.data_report is not None, 'Completed search and next data are required')
        prepare(args.search_plan, args.data_report, args.plan)
    plan = check(args.plan)
    run_name = CPU_NAME
    if args.run:
        import torch
        require(bool(os.environ.get('SLURM_JOB_ID')) and torch.cuda.device_count() == 2, 'Expected two-GPU Slurm allocation')
        require(all('H800' in torch.cuda.get_device_name(i) for i in range(2)), 'Unexpected GPU type')
        run_name = f"alternation-{os.environ['SLURM_JOB_ID']}"
        require(not (ROOT / 'data/native-rsft' / run_name).exists(), 'Training destination already exists')
    env = environment(plan['model'], plan['harness'], plan['dataset'], run_name)
    raw, actual = resolve_config(env)
    check_config(plan, actual, run_name)
    report = {'kind': 'native_alternation_training_preflight', 'status': 'PASS',
        'plan_sha256': file_sha256(args.plan), 'run_name': run_name, 'new_model_calls': 0, 'optimizer_steps': 0}
    if args.run:
        (ROOT / f'results/alternation-rsft-effective-{os.environ["SLURM_JOB_ID"]}.log').write_text(raw)
        write_new(ROOT / f'results/alternation-rsft-start-{os.environ["SLURM_JOB_ID"]}.json', report)
    print(json.dumps(report), flush=True)
    if args.run:
        os.execvpe('bash', ['bash', str(ROOT / 'ours/run_native_rsft.sh'), *OVERRIDES], env)


if __name__ == '__main__':
    main()
