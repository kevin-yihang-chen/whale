"""Freeze one measured E1 execution policy before independent endpoint access."""
import argparse
import json
import math
from pathlib import Path

from .fast_chart_protocol import ROOT, OUTPUT
from .fast_chart_throughput import select_fastest, validate_case
from .native_visual_service import write_new
from .visual_task import file_sha256
from .evidence import fingerprint

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
        'reference_model_path': reference['model']['path'],
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
            policy['reference_model_path'] != reference['model']['path'] or
            policy['reference_model_assets'] != reference['model']['assets'] or
            policy['verifier_sha256'] != reference['verifier_sha256'] or
            policy['h0_sha256'] != reference['harness_sha256']):
        raise ValueError('Policy differs from its real inference measurement')
    return policy


def compare_processors(base, updated):
    """Compare loaded behavior, including BPE merges, rather than filenames."""
    properties = {}
    for field in ('image_processor', 'video_processor'):
        a, b = getattr(base, field), getattr(updated, field)
        if type(a) is not type(b) or a.to_dict() != b.to_dict():
            raise ValueError(f'Endpoint {field} behavior differs')
        properties[field] = a.to_dict()
    if base.to_dict() != updated.to_dict() or base.chat_template != updated.chat_template:
        raise ValueError('Endpoint multimodal processor or template differs')
    tokenizers = [json.loads(p.tokenizer.backend_tokenizer.to_str()) for p in (base, updated)]
    if tokenizers[0] != tokenizers[1]:
        raise ValueError('Endpoint tokenizer normalization, merges or encoding differs')
    return {'processor_sha256': fingerprint(base.to_dict()),
            'tokenizer_backend_sha256': fingerprint(tokenizers[0]),
            'component_sha256': {k: fingerprint(v) for k, v in properties.items()}}


def processor_contract(model, policy):
    """Retain raw identities and validate the canonical export's semantics."""
    from .probe_updated_vllm import inference_contract
    from transformers import AutoProcessor
    base, updated = Path(policy['reference_model_path']), Path(model['path'])
    for directory, expected in ((base, policy['reference_model_assets']), (updated, model['assets'])):
        actual = {p.name: file_sha256(p) for p in sorted(directory.iterdir())
                  if p.is_file() and p.suffix in {'.json', '.jinja', '.txt'}}
        if actual != expected:
            raise ValueError('Endpoint processor assets changed from recorded identity')
    inference = inference_contract(base, updated)
    processors = [AutoProcessor.from_pretrained(p, local_files_only=True) for p in (base, updated)]
    return {'reference_assets_sha256': fingerprint(policy['reference_model_assets']),
            'endpoint_assets_sha256': fingerprint(model['assets']),
            'inference_contract': inference, **compare_processors(*processors)}


def bind(plan, policy):
    """All endpoints receive the same scheduling, with original scores/bounds."""
    if plan['verifier_sha256'] != policy['verifier_sha256']:
        raise ValueError('Endpoint shared scoring differs from calibrated execution')
    if plan['model']['assets'] != policy['reference_model_assets']:
        plan['processor_equivalence'] = processor_contract(plan['model'], policy)
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
