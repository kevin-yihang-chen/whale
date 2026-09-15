"""Measure shared E1 inference concurrency on V without changing model selection.

The completed batch8 h0 calibration is the reference. Two prespecified full
passes measure batch16 and32. Only evaluation scheduling changes; the selected
h0, learning-rate calibration and E4 trajectory budget remain fixed.
"""
import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from .fast_chart_protocol import ROOT, OUTPUT, storage_check
from .fast_chart_evaluation import decode_identity, read, evaluate
from .local_completion import checkpoint_manifest
from .native_visual_service import write_new
from .visual_task import file_sha256

CONCURRENCIES = (16, 32)
EXTRA = ('ours/fast_chart_throughput.py', 'ours/run_fast_chart_throughput.sh')


def normalized(config):
    """Ignore only output locations and the declared scheduling parameter."""
    value = deepcopy(config)
    value['data']['cache_dir'] = '<OUTPUT>'
    value['trainer']['experiment_name'] = '<OUTPUT>'
    value['actor_rollout_ref']['rollout']['trace']['experiment_name'] = '<OUTPUT>'
    value['actor_rollout_ref']['rollout']['max_num_seqs'] = '<CONCURRENCY>'
    return value


def validate_case(plan, reference):
    if (plan['kind'] != 'compact_chart_throughput_case' or plan['batch_size'] not in CONCURRENCIES or
            plan['partition'] != 'V' or reference['partition'] != 'V' or reference['batch_size'] != 8):
        raise ValueError('Throughput calibration requires the declared complete V passes')
    for key in ('manifest', 'manifest_sha256', 'audit_data_sha256', 'verifier_sha256',
                'model', 'worker_coordinates', 'harness_sha256', 'seed'):
        if plan[key] != reference[key]:
            raise ValueError(f'Throughput calibration changed the shared input: {key}')
    if normalized(plan['config']) != normalized(reference['config']):
        raise ValueError('Throughput calibration changed more than inference concurrency')
    if plan['config']['actor_rollout_ref']['rollout']['max_num_seqs'] != plan['batch_size']:
        raise ValueError('Dispatch concurrency differs from model scheduling')
    if plan['bounds']['images'] != 512 or plan['bounds']['maximum_generation_calls'] != 1536:
        raise ValueError('Throughput calibration changed complete V coverage or generation budget')


def prepare(path, output, calibration):
    path, output, calibration = map(lambda p: Path(p).resolve(), (path, output, calibration))
    if path.exists() or output.exists() or not output.is_relative_to(OUTPUT):
        raise ValueError('Require fresh compact throughput outputs')
    shared = read(calibration)
    if shared['status'] != 'COMPLETE_SHARED_H0_CALIBRATION':
        raise ValueError('Complete the shared h0 calibration before throughput measurement')
    reference_path = calibration.parent / f"{shared['selected']}-plan.json"
    reference = read(reference_path)
    old_result = Path(shared['results'][shared['selected']]['result'])
    if (read(old_result)['plan_sha256'] != file_sha256(reference_path) or
            read(old_result)['status'] != 'COMPLETE_COMPACT_CHART_PAIR_EVALUATION' or
            file_sha256(old_result) != shared['results'][shared['selected']]['result_sha256']):
        raise ValueError('Shared h0 evidence differs from its completed calibration')
    evidence = {str(p): file_sha256(p) for p in (calibration, reference_path, old_result)}
    for name, digest in reference['source_sha256'].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f'Throughput reference source changed: {name}')
    storage_check(12)
    output.mkdir()
    cases = {}
    for batch_size in CONCURRENCIES:
        case = deepcopy(reference)
        case_output = output / f'batch{batch_size}'
        cfg = case['config']
        cfg['data']['cache_dir'] = str(case_output / 'dataset-cache')
        cfg['trainer']['experiment_name'] = case_output.name
        cfg['actor_rollout_ref']['rollout']['trace']['experiment_name'] = case_output.name
        cfg['actor_rollout_ref']['rollout']['max_num_seqs'] = batch_size
        case.update(kind='compact_chart_throughput_case', phase='shared-inference-throughput',
                    output=str(case_output), batch_size=batch_size,
                    reference=str(reference_path), evidence_sha256=evidence,
                    decode_sha256=decode_identity(cfg, case['model']['assets']))
        case['bounds']['time_limit_seconds'] = 3600
        case['source_sha256'].update({name: file_sha256(ROOT/name) for name in EXTRA})
        validate_case(case, reference)
        case_path = output / f'batch{batch_size}-plan.json'
        write_new(case_path, case)
        cases[str(batch_size)] = {'plan': str(case_path), 'sha256': file_sha256(case_path)}
    suite = {'kind': 'compact_chart_throughput_calibration', 'output': str(output),
        'cases': cases, 'reference': str(reference_path), 'reference_result': str(old_result),
        'evidence_sha256': evidence,
        'source_sha256': {name: file_sha256(ROOT/name) for name in EXTRA},
        'selection': 'Lowest observed seconds per V image among batch8/16/32; tie chooses smaller batch.',
        'scope': 'Shared endpoint evaluation scheduling; h0/LR and E4 training remain frozen.',
        'bounds': {'gpus': 1, 'cpus': 12, 'time_limit_seconds': 3600,
                   'new_images_evaluated': 1024, 'api_calls': 0, 'new_model_checkpoints': 0},
        'limitations': ['One measurement per concurrency; no guarantee across longer future harness outputs.',
            'Batched greedy inference can change answers; report every paired result and answer change.',
            'No test data, performance-based h0/LR reselection, or automatic formal expansion.']}
    write_new(path, suite)
    return suite


def check_case(path):
    plan = read(path)
    validate_case(plan, read(plan['reference']))
    if Path(plan['output']).exists() or not Path(plan['output']).is_relative_to(OUTPUT):
        raise ValueError('Existing or unaccounted throughput output')
    for name, digest in {**plan['source_sha256'], **plan['evidence_sha256']}.items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError('Frozen throughput source or reference changed')
    if (file_sha256(Path(plan['manifest'])) != plan['manifest_sha256'] or
            checkpoint_manifest(Path(plan['model']['path'])) != plan['model'] or
            decode_identity(plan['config'], plan['model']['assets']) != plan['decode_sha256']):
        raise ValueError('Throughput data, model or decoder identity changed')
    return plan


def check(path):
    plan = read(path)
    if (plan['kind'] != 'compact_chart_throughput_calibration' or
            tuple(plan['cases']) != tuple(map(str, CONCURRENCIES))):
        raise ValueError('Changed throughput calibration scope')
    for name, digest in {**plan['source_sha256'], **plan['evidence_sha256']}.items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError('Frozen throughput suite identity changed')
    for item in plan['cases'].values():
        if file_sha256(Path(item['plan'])) != item['sha256']:
            raise ValueError('Changed throughput case plan')
        check_case(item['plan'])
    storage_check(12)
    return plan


def select_fastest(measurements):
    if set(measurements) != {'8', '16', '32'}:
        raise ValueError('Do not select concurrency from incomplete measurements')
    return min((8, 16, 32), key=lambda n: (measurements[str(n)]['seconds_per_image'], n))


def summarize(path):
    suite = read(path)
    reference = read(suite['reference_result'])
    baseline_records_path = Path(suite['reference_result']).parent/'evaluation/result.json'
    baseline = {r['sample_id']: r for r in read(baseline_records_path)['records']}
    if len(baseline) != 512:
        raise ValueError('Reference throughput pass lacks complete V coverage')
    paths = {'8': (Path(suite['reference']), Path(suite['reference_result']))}
    paths.update({n: (Path(item['plan']), Path(read(item['plan'])['output'])/'result.json')
                  for n, item in suite['cases'].items()})
    measurements = {}
    evidence = {**suite['evidence_sha256'], str(baseline_records_path): file_sha256(baseline_records_path)}
    for n, (plan_path, result_path) in paths.items():
        result = read(result_path)
        record_path = result_path.parent/'evaluation/result.json'
        rows = {r['sample_id']: r for r in read(record_path)['records']}
        if (result['status'] != 'COMPLETE_COMPACT_CHART_PAIR_EVALUATION' or
                result['plan_sha256'] != file_sha256(plan_path) or set(rows) != set(baseline)):
            raise ValueError('Incomplete or mismatched throughput measurement')
        measurements[n] = {'seconds_per_image': result['timing_seconds']['evaluation']/512,
            'timing_seconds': result['timing_seconds'],
            'marginal_accuracy': result['marginal_accuracy'], 'paired_accuracy': result['paired_accuracy'],
            'generated_tokens': result['observed']['generated_tokens'],
            'committed_answers_changed_from_batch8': sum(rows[k]['committed_answer'] != baseline[k]['committed_answer'] for k in rows),
            'correctness_changed_from_batch8': sum(rows[k]['correct'] != baseline[k]['correct'] for k in rows),
            'result': str(result_path), 'plan': str(plan_path)}
        evidence.update({str(p): file_sha256(p) for p in (plan_path, result_path, record_path)})
    selected = select_fastest(measurements)
    result = {'status': 'COMPLETE_SHARED_INFERENCE_THROUGHPUT', 'plan_sha256': file_sha256(path),
        'job_id': os.environ['SLURM_JOB_ID'], 'measurements': measurements, 'selected_batch_size': selected,
        'evidence_sha256': evidence, 'shared_h0_lr_reselected': False,
        'scientific_method_verified': False, 'formal_expansion_admitted': False,
        'endpoint_policy': 'Use the same measured concurrency for all T/R/ChartQA method comparisons; retain full coverage.',
        'limits': suite['limitations']}
    write_new(Path(suite['output'])/'result.json', result)
    return result


def run(path):
    suite = check(path)
    output = Path(suite['output'])
    try:
        for n, case in suite['cases'].items():
            with (output/f'batch{n}.log').open('x') as stream:
                subprocess.run([sys.executable, '-m', 'ours.fast_chart_throughput', 'run-case',
                    '--plan', case['plan']], stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        summarize(path)
    except BaseException as exc:
        write_new(output/'failure.json', {'status': 'INCOMPLETE_THROUGHPUT_NO_POLICY',
            'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'check', 'run', 'run-case'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.plan, args.output, args.calibration)
    elif args.phase == 'check':
        print(json.dumps({'status': 'PASS_THROUGHPUT_PREFLIGHT', 'bounds': check(args.plan)['bounds']}))
    elif args.phase == 'run-case':
        asyncio.run(evaluate(check_case(args.plan), args.plan.resolve()))
    else:
        run(args.plan.resolve())
