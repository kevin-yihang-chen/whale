"""Compare direct answers and normalized crop coordinates on real visual pairs.

This diagnoses the all-zero first screen without changing model weights,
question text, verifier, sampling, or the three-turn limit. It is E1 measurement
and shared interface calibration, not a VETO result or strong-h0 selection.
"""
import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from . import native_visual_service as native
from .evidence import fingerprint
from .visual_task import file_sha256

ROOT = native.ROOT
MODES = ('direct', 'normalized_zoom')
EXTRA = ('ours/normalized_visual_zoom.py', 'ours/normalized_visual_tools.yaml',
         'ours/visual_coordinate_diagnostic.py', 'ours/run_visual_coordinate_diagnostic.sh')


def derive(base, mode, output):
    assert mode in MODES
    plan = deepcopy(base)
    plan.update(kind='native_visual_coordinate_diagnostic_case', mode=mode, output=str(output))
    cfg = plan['config']
    cfg['data']['cache_dir'] = str(output / 'dataset-cache')
    cfg['trainer']['experiment_name'] = output.name
    cfg['actor_rollout_ref']['rollout']['trace']['experiment_name'] = output.name
    tool = str(ROOT/'ours/normalized_visual_tools.yaml') if mode == 'normalized_zoom' else None
    cfg['data']['tool_config_path'] = tool
    cfg['actor_rollout_ref']['rollout']['multi_turn']['tool_config_path'] = tool
    plan['decode_sha256'] = fingerprint({'rollout': cfg['actor_rollout_ref']['rollout'],
        'chat_template': cfg['data']['apply_chat_template_kwargs'], 'model_assets': plan['model']['assets']})
    plan['source_sha256'].update({p: file_sha256(ROOT/p) for p in EXTRA})
    plan['limitations'] += ['Diagnostic adaptation of tool exposure or its coordinate convention only.',
                           'All-zero pixel-tool screen is retained; this is not a VETO performance gain.',
                           'Actual Ray CPU resource slots match the 12 allocated Slurm CPUs.']
    return plan


def prepare(path, output, base_path):
    base = native.check(base_path)
    assert not path.exists() and not output.exists()
    output.mkdir()
    cases = {}
    for mode in MODES:
        p = output / f'{mode}-plan.json'
        native.write_new(p, derive(base, mode, output/mode))
        cases[mode] = {'path': str(p), 'sha256': file_sha256(p)}
    plan = {'kind': 'visual_coordinate_diagnostic', 'base_plan': str(base_path),
        'base_plan_sha256': file_sha256(base_path), 'output': str(output), 'cases': cases,
        'source_sha256': {p: file_sha256(ROOT/p) for p in EXTRA},
        'bounds': {'gpus': 1, 'cpus': 12, 'time_limit_seconds': 2400,
                   'maximum_generation_calls': 96, 'new_model_checkpoints': 0, 'api_calls': 0},
        'formal_matrix_started': False}
    native.write_new(path, plan)


def check_case(path, base_path):
    case = json.loads(path.read_text())
    base = native.check(base_path)
    assert case == derive(base, case['mode'], Path(case['output'])), 'Changed diagnostic case'
    return case


def run_case(path, base_path):
    case = check_case(path, base_path)
    assert int(os.environ['SLURM_CPUS_PER_TASK']) == 12
    import ray
    original_init = ray.init
    def allocated_init(*args, **kwargs):
        assert kwargs['num_cpus'] == 16 and kwargs['num_gpus'] == 1
        return original_init(*args, **{**kwargs, 'num_cpus': 12})
    ray.init = allocated_init
    try:
        asyncio.run(native.run(case, path))
    finally:
        ray.init = original_init
    # Native source snapshots are retained by run(); retain the additional
    # diagnostic adapter files as well, whose identities are in the case plan.
    for name in EXTRA:
        target = Path(case['output'])/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write((ROOT/name).read_bytes())


def run(path):
    plan = json.loads(path.read_text())
    assert plan['kind'] == 'visual_coordinate_diagnostic' and tuple(plan['cases']) == MODES
    assert file_sha256(Path(plan['base_plan'])) == plan['base_plan_sha256']
    for source, sha in plan['source_sha256'].items():
        assert file_sha256(ROOT/source) == sha
    root = Path(plan['output'])
    results = {}
    for mode, item in plan['cases'].items():
        assert file_sha256(Path(item['path'])) == item['sha256']
        with (root/f'{mode}.log').open('x') as stream:
            subprocess.run([sys.executable, '-m', 'ours.visual_coordinate_diagnostic', '--phase', 'case',
                            '--plan', item['path'], '--base-plan', plan['base_plan']],
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
        result_path = root/mode/'result.json'
        result = json.loads(result_path.read_text())
        assert result['status'] == 'COMPLETE_REAL_NATIVE_VISUAL_EVALUATION'
        assert result['job_id'] == os.environ['SLURM_JOB_ID'] and result['plan_sha256'] == item['sha256']
        results[mode] = {'paired_accuracy': result['paired_accuracy'],
                        'marginal_accuracy': result['marginal_accuracy'],
                        'policy_calls': result['observed']['policy_calls'],
                        'result_path': str(result_path), 'result_sha256': file_sha256(result_path)}
    native.write_new(root/'result.json', {'status': 'COMPLETE_VISUAL_COORDINATE_DIAGNOSTIC',
        'job_id': os.environ['SLURM_JOB_ID'], 'results': results, 'plan_sha256': file_sha256(path),
        'scientific_method_verified': False, 'formal_matrix_started': False, 'optimizer_steps': 0})
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'case'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--base-plan', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.plan.resolve(), args.output.resolve(), args.base_plan.resolve())
    elif args.phase == 'case':
        run_case(args.plan.resolve(), args.base_plan.resolve())
    else:
        run(args.plan.resolve())
