"""Bind a completed selected-harness E4 recovery to canonical export and audit."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .alternation_training import read, write_new
from .audit_native_training_batch import require
from .local_completion import checkpoint_manifest
from .visual_task import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('ours/alternation_checkpoint_export.py', 'ours/run_alternation_checkpoint_export.sh',
    'ours/canonical_native_export.py', 'ours/verify_alternation_transition.py',
    'ours/verify_native_transition.py', 'ours/local_completion.py', 'ours/visual_task.py',
    'ours/audit_native_training_batch.py', 'ours/training_bootstrap.py',
    'upstream/WHALE/domains/chess_puzzles/verl/model_merger/fsdp_model_merger.py',
    'upstream/WHALE/domains/chess_puzzles/verl/model_merger/base_model_merger.py')


def checked_recovery(plan_path, result_path):
    from .alternation_recovery import check
    result = read(result_path)
    require(result['kind'] == 'native_alternation_recovery_result' and result['status'] == 'COMPLETED',
            'A completed native recovery is required')
    plan = check(plan_path)
    require(result['plan_sha256'] == file_sha256(plan_path), 'Recovery plan/result mismatch')
    require(result['native_checkpoint_exists'] and result['completed_optimizer_steps'] == plan['expected_optimizer_steps'] == 2,
            'Two completed native optimizer steps required')
    require(result['accepted'] == plan['accepted'] == 11 and result['loss_tokens'] == plan['loss_tokens'] == 30392,
            'Training objective differs')
    require(result['new_model_calls'] == 0 and math.isfinite(result['gradient_norm']) and result['gradient_norm'] > 0,
            'Recovery accounting or gradient evidence invalid')
    native = ROOT / f"data/native-rsft/alternation-replay-{result['job_id']}/global_step_1/actor"
    require(file_sha256(native / 'model_world_size_1_rank_0.pt') == result['native_checkpoint_sha256'],
            'Completed checkpoint changed')
    return plan, result, native


def prepare(path, recovery_path, result_path):
    plan, result, native = checked_recovery(recovery_path, result_path)
    job = result['job_id']
    previous = Path('results/native-checkpoint-transition-222156.json')
    original = read(ROOT / plan['source_plan'])
    require(original['source_sha256'].get(str(previous)) == file_sha256(ROOT / previous),
            'Previous parameter handoff is not bound to the training phase')
    require(checkpoint_manifest(Path(plan['model'])) == read(ROOT / previous)['exported'], 'Incoming model changed')
    sources = set(plan['source_sha256']) | set(SOURCES) | {str(recovery_path), str(result_path), str(previous)}
    sources.update(str(p.relative_to(ROOT)) for p in native.rglob('*') if p.is_file())
    report = {'kind': 'native_alternation_export_plan', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'recovery_plan': str(recovery_path), 'recovery_result': str(result_path), 'source_job_id': job,
        'training_plan': plan['source_plan'], 'previous_transition': str(previous),
        'base': plan['model'], 'native': str(native),
        'target': str(native.parent.parent / 'hf-checkpoint-canonical'),
        'export_report': f'results/alternation-canonical-export-{job}.json',
        'transition_report': f'results/alternation-checkpoint-transition-{job}.json',
        'native_checkpoint_sha256': result['native_checkpoint_sha256'],
        'source_sha256': {name: file_sha256(ROOT / name) for name in sorted(sources)},
        'resources': {'gpus': 1, 'gpu_type': 'rtx_4090', 'cpus': 4, 'ram_gib': 64,
                      'time_limit_seconds': 1200, 'max_gpu_hours': 1/3},
        'new_model_calls': 0,
        'limitations': ['Original merger and BF16 cast with canonical names and incoming inference config.',
            'CPU execution; required minimum GPU allocation remains fully charged.',
            'Value changes do not establish task improvement or actual serving weight propagation.']}
    for name in ('target', 'export_report', 'transition_report'):
        require(not (ROOT / report[name]).exists(), f'Preserve existing {name}')
    write_new(path, report)
    return report


def run(path):
    plan = read(path)
    require(plan['kind'] == 'native_alternation_export_plan', 'Wrong export plan')
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Export must use the reviewed CPU Slurm wrapper')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(ROOT / name) == digest, f'Frozen export input changed: {name}')
    checked_recovery(ROOT / plan['recovery_plan'], ROOT / plan['recovery_result'])
    start = {'kind': 'native_alternation_export_start', 'job_id': os.environ['SLURM_JOB_ID'],
        'plan_sha256': file_sha256(path), 'source_job_id': plan['source_job_id'],
        'native_checkpoint_sha256': plan['native_checkpoint_sha256'], 'new_model_calls': 0}
    write_new(ROOT / f"results/alternation-checkpoint-export-start-{start['job_id']}.json", start)
    subprocess.run([sys.executable, '-m', 'ours.canonical_native_export', '--base', plan['base'],
        '--native', plan['native'], '--target', plan['target'], '--report', plan['export_report']], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'ours.verify_alternation_transition', '--plan', plan['training_plan'],
        '--native', plan['native'], '--exported', plan['target'], '--previous-transition', plan['previous_transition'],
        '--output', plan['transition_report']], cwd=ROOT, check=True)
    transition = read(ROOT / plan['transition_report'])
    require(transition['native_checkpoint_sha256'] == start['native_checkpoint_sha256'],
            'Native checkpoint changed during export')
    print(json.dumps({**start, 'kind': 'native_alternation_export_complete', 'status': 'PASS',
                      'transition_sha256': file_sha256(ROOT / plan['transition_report'])}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--recovery-plan', type=Path)
    p.add_argument('--result', type=Path)
    args = p.parse_args()
    os.chdir(ROOT)
    if args.prepare:
        require(args.recovery_plan and args.result, 'Provide completed recovery evidence')
        plan = prepare(args.plan, args.recovery_plan, args.result)
        print(json.dumps({'kind': plan['kind'], 'plan_sha256': file_sha256(args.plan),
                          'frozen_sources': len(plan['source_sha256'])}), flush=True)
    else:
        run(args.plan)
