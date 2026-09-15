"""Freeze one measured E1 execution policy before independent endpoint access."""
import argparse
import json
import math
from pathlib import Path

from .fast_chart_protocol import ROOT, OUTPUT
from .fast_chart_throughput import select_fastest, validate_case
from .native_visual_service import write_new
from .visual_task import file_sha256

POLICY = OUTPUT/'shared-inference-policy-v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def verify_measurement(result_path, suite_path):
    result_path, suite_path = Path(result_path).resolve(), Path(suite_path).resolve()
    result, suite = read(result_path), read(suite_path)
    if (result['status'] != 'COMPLETE_SHARED_INFERENCE_THROUGHPUT' or
            suite['kind'] != 'compact_chart_throughput_calibration' or
            result['plan_sha256'] != file_sha256(suite_path) or
            Path(suite['output']) != result_path.parent):
        raise ValueError('Require a complete matching throughput calibration')
    for name, digest in {**result['evidence_sha256'], **suite['evidence_sha256']}.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError('Measured throughput evidence changed')
    for name, digest in suite['source_sha256'].items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError('Measured throughput implementation changed')
    allocation = OUTPUT/'allocations'/suite_path.stem
    submission, terminal = read(allocation/'submission.json'), read(allocation/'allocation-result.json')
    if (submission['plan_sha256'] != file_sha256(suite_path) or submission['gpus'] != 1 or
            terminal['state'] != 'COMPLETED' or str(terminal['job_id']) != str(result['job_id'])):
        raise ValueError('Throughput policy requires a matching completed real GPU allocation')
    reference = read(suite['reference'])
    expected = {'8': str(suite['reference']), **{n: item['plan'] for n,item in suite['cases'].items()}}
    for n, measurement in result['measurements'].items():
        if measurement['plan'] != expected[n]:
            raise ValueError('Throughput measurement belongs to a different case')
        plan = read(measurement['plan'])
        if n != '8':
            validate_case(plan, reference)
        raw = read(measurement['result'])
        if (raw['status'] != 'COMPLETE_COMPACT_CHART_PAIR_EVALUATION' or
                raw['plan_sha256'] != file_sha256(Path(measurement['plan'])) or
                measurement['seconds_per_image'] != raw['timing_seconds']['evaluation']/512 or
                not math.isfinite(measurement['seconds_per_image']) or measurement['seconds_per_image'] <= 0 or
                (n != '8' and str(raw['job_id']) != str(result['job_id']))):
            raise ValueError('Throughput estimate differs from complete real measurement')
    selected = select_fastest(result['measurements'])
    if selected != result['selected_batch_size']:
        raise ValueError('Execution policy is not the prespecified fastest complete pass')
    return result, suite, reference


def freeze(result_path, suite_path):
    if POLICY.exists():
        raise ValueError('The shared endpoint policy is already frozen')
    for receipt in (OUTPUT/'allocations').glob('*/submission.json'):
        allocation = read(receipt)
        if read(allocation['plan'])['kind'] == 'compact_registered_endpoint_evaluation':
            raise ValueError('Do not choose endpoint scheduling after independent evaluation submission')
    result, suite, reference = verify_measurement(result_path, suite_path)
    record = {'kind': 'compact_shared_endpoint_inference_policy',
        'throughput_result': str(Path(result_path).resolve()), 'throughput_suite': str(Path(suite_path).resolve()),
        'evidence_sha256': {str(Path(p).resolve()): file_sha256(Path(p)) for p in (result_path,suite_path)},
        'batch_size': result['selected_batch_size'], 'agent_workers': 2,
        'reference_model_assets': reference['model']['assets'],
        'verifier_sha256': reference['verifier_sha256'], 'h0_sha256': reference['harness_sha256'],
        'selection': suite['selection'], 'scope': 'All methods and all seeds, T/R/ChartQA only.',
        'development_and_training_changed': False, 'scientific_method_verified': False}
    write_new(POLICY, record)
    return record


def load():
    policy = read(POLICY)
    if policy['kind'] != 'compact_shared_endpoint_inference_policy':
        raise ValueError('Wrong endpoint execution policy')
    for name, digest in policy['evidence_sha256'].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError('Frozen execution evidence changed')
    measured, _, reference = verify_measurement(policy['throughput_result'], policy['throughput_suite'])
    if (policy['batch_size'] != measured['selected_batch_size'] or policy['agent_workers'] != 2 or
            policy['reference_model_assets'] != reference['model']['assets'] or
            policy['verifier_sha256'] != reference['verifier_sha256'] or
            policy['h0_sha256'] != reference['harness_sha256']):
        raise ValueError('Policy differs from its real inference measurement')
    return policy


def bind(plan, policy):
    """All endpoints receive the same scheduling, with original scores/bounds."""
    if (plan['model']['assets'] != policy['reference_model_assets'] or
            plan['verifier_sha256'] != policy['verifier_sha256']):
        raise ValueError('Endpoint processor or shared scoring differs from calibrated execution')
    plan['batch_size'] = policy['batch_size']
    rollout = plan['config']['actor_rollout_ref']['rollout']
    rollout['max_num_seqs'] = policy['batch_size']
    rollout['agent']['num_workers'] = policy['agent_workers']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.result, args.suite), indent=2))
