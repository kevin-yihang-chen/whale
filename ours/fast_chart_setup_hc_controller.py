"""Finish measured E1 scheduling, then one authorized setup H/C measurement.

This bounded handoff does not launch proposals, training, test evaluation,
cleanup, or the formal matrix. It preserves completed-allocation visibility
waits and never retries an ambiguous or failed submission automatically.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from .fast_chart_admission import verified_calibration, read
from .fast_chart_calibration_controller import terminal, completed_artifact, command
from .fast_chart_forecast import forecast
from .fast_chart_inference_policy import POLICY, freeze, load as load_policy
from .fast_chart_protocol import ROOT, OUTPUT, CompactExecutionBudget, storage_check
from .fast_chart_search_evaluation import certify
from .native_visual_service import write_new
from .visual_task import file_sha256


def admission(path):
    record = read(path)
    if (record['status'] != 'PREPARED_BOUNDED_SETUP_HC_MEASUREMENT' or
            record['phase'] != 'setup' or record['maximum_additional_gpu_hours'] != 1 or
            record['formal_expansion_admitted'] is not False):
        raise ValueError('Controller requires the bounded setup measurement scope')
    for name in ('plan', 'configuration'):
        if file_sha256(Path(record[name])) != record[name+'_sha256']:
            raise ValueError('Bounded setup evidence changed')
    plan = read(record['plan'])
    cfg = verified_calibration(record['configuration'])
    if (plan['kind'] != 'compact_chart_search_evaluation' or plan['candidate'] != 'h0' or
            plan['seed'] != 42 or plan['with_audit'] is not True or
            plan['bounds']['gpus'] != 1 or plan['bounds']['time_limit_seconds'] != 3600 or
            plan['bounds']['images'] != 256 or plan['bounds']['api_calls'] != 0 or
            plan['bounds']['new_model_checkpoints'] != 0 or
            plan['model']['path'] != str(Path(cfg['results'][str(cfg['learning_rate'])]['result_path']).parent/'hf-canonical')):
        raise ValueError('Setup measurement changed its model, candidate or resource scope')
    return record


def advance(directory, admission_path, throughput_path):
    record = admission(admission_path)
    suite = read(throughput_path)
    if suite['kind'] != 'compact_chart_throughput_calibration':
        raise ValueError('Wrong prerequisite throughput suite')
    state = terminal(throughput_path)
    if state is None:
        raise ValueError('The prerequisite throughput job was not submitted')
    if state != 'COMPLETED':
        return {'status': 'WAITING_THROUGHPUT_ALLOCATION'}
    result_path = Path(suite['output'])/'result.json'
    if completed_artifact(directory, result_path) is None:
        return {'status': 'WAITING_THROUGHPUT_ARTIFACT'}
    if not POLICY.exists():
        freeze(result_path, throughput_path)
    policy = load_policy()
    if (policy['throughput_result'] != str(result_path) or
            policy['throughput_suite'] != str(throughput_path)):
        raise ValueError('Existing endpoint policy belongs to another measurement')
    projection = directory/'measured-expansion-forecast.json'
    if not projection.exists():
        write_new(projection, forecast(record['configuration'], measured_endpoint_policy=True))
    # This single H/C measurement belongs to the authorized setup closure.
    # A projection does not grant broad formal expansion or cleanup permission.
    search = Path(record['plan'])
    state = terminal(search)
    if state is None:
        command(directory, 'setup-hc-submit', ['ours.fast_chart_submit', '--kind', 'search',
            '--phase', 'setup', '--plan', str(search)])
        return {'status': 'SUBMITTED_SETUP_HC', 'plan': str(search)}
    if state != 'COMPLETED':
        return {'status': 'WAITING_SETUP_HC_ALLOCATION'}
    output = Path(read(search)['output'])
    if completed_artifact(directory, output/'result.json') is None:
        return {'status': 'WAITING_SETUP_HC_ARTIFACT'}
    allocation = OUTPUT/'allocations'/search.stem/'allocation-result.json'
    evidence = certify(search, allocation)
    result = {'status': 'COMPLETE_BOUNDED_SETUP_HC',
        'plan': str(search), 'plan_sha256': file_sha256(search),
        'allocation': str(allocation), 'allocation_sha256': file_sha256(allocation),
        'result': str(output/'result.json'), 'result_sha256': file_sha256(output/'result.json'),
        'H_accuracy': evidence['candidate'].accuracy, 'C_paired_accuracy': evidence['audit'].paired_accuracy,
        'shared_endpoint_policy': str(POLICY), 'policy_sha256': file_sha256(POLICY),
        'forecast': str(projection), 'forecast_sha256': file_sha256(projection),
        'budget': CompactExecutionBudget().compact_summary(), 'storage': storage_check(),
        'scope': 'One fixed incoming harness on optimization H/C; no method benefit established.',
        'formal_expansion_admitted': False, 'scientific_method_verified': False,
        'next_step': 'Review actual search costs and proceed with bounded closure under existing limits.'}
    write_new(directory/'result.json', result)
    return result


def run(directory, admission_path, throughput_path):
    directory, admission_path, throughput_path = [Path(p).resolve() for p in
        (directory, admission_path, throughput_path)]
    if directory.exists() or not directory.is_relative_to(OUTPUT):
        raise ValueError('Require a fresh controller directory')
    admission(admission_path)
    directory.mkdir()
    with (directory/'controller.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_new(directory/'start.json', {'pid': os.getpid(),
            'admission': str(admission_path), 'admission_sha256': file_sha256(admission_path),
            'throughput': str(throughput_path), 'throughput_sha256': file_sha256(throughput_path),
            'source_sha256': file_sha256(Path(__file__))})
        started = time.monotonic()
        try:
            while time.monotonic()-started < 3*3600:
                report = advance(directory, admission_path, throughput_path)
                (directory/'status.json').write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps({'status': report['status']}), flush=True)
                if report['status'] == 'COMPLETE_BOUNDED_SETUP_HC':
                    return
                time.sleep(30)
            raise TimeoutError('Bounded controller wait expired; inspect the same jobs, do not resubmit')
        except BaseException as exc:
            failure = {'status': 'STOPPED_REQUIRES_DIAGNOSIS', 'error_type': type(exc).__name__,
                'error': str(exc), 'no_automatic_retry': True}
            write_new(directory/'failure.json', failure)
            (directory/'status.json').write_text(json.dumps(failure, indent=2)+'\n')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'admission', 'throughput'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.admission, args.throughput)
