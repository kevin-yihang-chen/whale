"""Bounded E1 -> E2/E3 -> E4 handoff for the already proposed seed42 cycle.

Reuses existing submissions; no API calls, automatic retries, cleanup, other
conditions, independent test evaluation or formal-matrix admission.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from .fast_chart_calibration_controller import command, completed_artifact, terminal
from .fast_chart_protocol import ROOT, OUTPUT, CompactExecutionBudget, storage_check
from .fast_chart_search import load as load_search, read
from .native_visual_service import write_new
from .visual_task import file_sha256

ADMISSION = ROOT/'results/fast-chart-first-cycle-bounded-admission-20260915-v1.json'
SEARCH = OUTPUT/'formal-v1/seed42/first-search'
CONFIGURATION = OUTPUT/'lr-calibration-v3/result.json'
TRAIN = ROOT/'results/fast-chart-seed42-veto-stage2-plan-20260915-v1.json'
FOLLOWUP = ROOT/'results/fast-chart-seed42-veto-stage2-followup-plan-20260915-v1.json'
BRANCH = OUTPUT/'formal-v1/seed42/veto'


def admission():
    record = read(ADMISSION)
    if (record['status'] != 'BOUNDED_FIRST_CYCLE_FITS_EXISTING_SETUP_CAP' or
            record['seed'] != 42 or record['worst_case_additional_gpu_hours'] != 6 or
            record['formal_expansion_admitted'] is not False):
        raise ValueError('Wrong first-cycle scope')
    for name, digest in record['evidence_sha256'].items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError('Bounded admission evidence changed')
    search = load_search(SEARCH)
    if search['seed'] != 42 or not search['parent_plan'] or not (SEARCH/'proposal-ready.json').is_file():
        raise ValueError('Require the completed single proposal on the seed42 updated prefix')
    return search


def allocation(path):
    return OUTPUT/'allocations'/Path(path).stem/'allocation-result.json'


def wait_or_submit(directory, path, kind):
    """Submission logs and receipts make an ambiguous attempt non-repeatable."""
    record = read(path)
    expected = 2 if kind == 'training' else 1
    if (record['bounds']['gpus'] != expected or record['bounds']['time_limit_seconds'] != 3600 or
            record['bounds'].get('api_calls', 0) != 0 or record['seed'] != 42):
        raise ValueError('Allocation exceeds the bounded first cycle')
    receipt = allocation(path).with_name('submission.json')
    if receipt.exists():
        submitted = read(receipt)
        if (submitted['phase'] != 'setup' or submitted['plan_sha256'] != file_sha256(path) or
                not str(submitted.get('job_id', '')).isdigit()):
            raise ValueError('Submission scope changed or submission is ambiguous; inspect, do not retry')
    state = terminal(path)
    if state is None:
        command(directory, path.stem+'-submit', ['ours.fast_chart_submit', '--kind', kind,
            '--phase', 'setup', '--plan', str(path)])
        return False
    return state == 'COMPLETED'


def candidate(directory, name):
    path = ROOT/f'results/fast-chart-seed42-first-search-{name}-plan-20260915-v1.json'
    request = read(SEARCH/f'evaluation-request-{name}.json')
    if not path.exists():
        command(directory, name+'-prepare', ['ours.fast_chart_search_evaluation', 'prepare',
            '--plan', str(path), '--output', str(SEARCH.parent/f'first-search-evaluation/{name}'),
            '--reference', request['reference'], '--harness', request['harness'],
            '--seed', '42', '--name', name, '--phase', request['identity']['phase']])
    plan = read(path)
    if (name not in ('h1', 'h2', 'h3') or plan['kind'] != 'compact_chart_search_evaluation' or
            plan['candidate'] != name or plan['identity'] != request['identity'] or
            plan['reference_plan'] != request['reference'] or plan['harness_sha256'] != request['harness_sha256']):
        raise ValueError('Candidate differs from its original fixed-weight request')
    end = allocation(path)
    if end.exists() and read(end)['state'] != 'COMPLETED':
        failure = completed_artifact(directory, Path(plan['output'])/'failure.json')
        if failure is None:
            return 'WAITING_FAILED_CANDIDATE_EVIDENCE'
        command(directory, name+'-failed-evaluation', ['ours.fast_chart_search', 'failed-evaluation',
            '--root', str(SEARCH), '--candidate-plan', str(path), '--allocation', str(end)])
        return 'CONSUMED_FAILED_CANDIDATE'
    if not wait_or_submit(directory, path, 'search'):
        return 'WAITING_CANDIDATE_ALLOCATION'
    if completed_artifact(directory, Path(plan['output'])/'result.json') is None:
        return 'WAITING_CANDIDATE_ARTIFACT'
    command(directory, name+'-attach', ['ours.fast_chart_search', 'attach', '--root', str(SEARCH),
        '--candidate-plan', str(path), '--allocation', str(end)])
    return 'ATTACHED_CANDIDATE'


def advance(directory, search):
    if not (SEARCH/'result.json').exists():
        missing = [name for name in ('h1', 'h2', 'h3')
            if not (SEARCH/f'evaluation-{name}.json').exists() and not (SEARCH/f'failure-{name}.json').exists()]
        if missing and (SEARCH/f'evaluation-request-{missing[0]}.json').exists():
            return {'status': candidate(directory, missing[0]), 'candidate': missing[0]}
        step = missing[0] if missing else 'selection'
        command(directory, step+'-advance', ['ours.fast_chart_search', 'advance', '--root', str(SEARCH)])
        return {'status': 'ADVANCED_NATIVE_SEARCH'}
    selection = read(SEARCH/'result.json')
    if selection['status'] != 'COMPLETE_COMPACT_SEARCH_AND_SELECTION':
        raise ValueError('Search did not finish all allocated slots')
    # VETO was fixed prospectively, before viewing candidate outcomes.
    BRANCH.mkdir(parents=True, exist_ok=True)
    if not TRAIN.exists():
        command(directory, 'stage2-prepare', ['ours.fast_chart_condition', '--plan', str(TRAIN),
            '--output', str(BRANCH/'training'), '--configuration', str(CONFIGURATION),
            '--parent-plan', search['parent_plan'], '--condition', 'veto', '--search', str(SEARCH)])
    train = read(TRAIN)
    if (train['stage'] != 2 or train['parent_plan'] != search['parent_plan'] or
            train['selection'] != str(SEARCH/'selection-veto.json') or train['augmentation'] is not None or
            train['rate'] != read(CONFIGURATION)['learning_rate'] or
            train['bounds']['native_batches'] != 4 or train['bounds']['trajectories'] != 256 or
            train['harness_sha256'] != read(SEARCH/'selection-veto.json')['harness_sha256']):
        raise ValueError('Continuation differs from the prospective VETO branch')
    if not wait_or_submit(directory, TRAIN, 'training'):
        return {'status': 'WAITING_VETO_CONTINUATION'}
    if completed_artifact(directory, BRANCH/'training/execution-result.json') is None:
        return {'status': 'WAITING_CONTINUATION_ARTIFACT'}
    if not FOLLOWUP.exists():
        command(directory, 'followup-prepare', ['ours.fast_chart_followup', 'prepare', '--plan', str(FOLLOWUP),
            '--output', str(BRANCH/'followup'), '--training-plan', str(TRAIN)])
    if read(FOLLOWUP)['training_plan_sha256'] != file_sha256(TRAIN):
        raise ValueError('Followup does not verify this continuation')
    if not wait_or_submit(directory, FOLLOWUP, 'followup'):
        return {'status': 'WAITING_CONTINUATION_FOLLOWUP'}
    result = completed_artifact(directory, BRANCH/'followup/result.json')
    if result is None:
        return {'status': 'WAITING_FOLLOWUP_ARTIFACT'}
    if result['status'] != 'COMPLETE_COMPACT_NATIVE_FOLLOWUP' or result['plan_sha256'] != file_sha256(FOLLOWUP):
        raise ValueError('Followup completion identity differs')
    paths = [SEARCH/'result.json', SEARCH/'selection-veto.json', TRAIN, FOLLOWUP,
        allocation(TRAIN), allocation(FOLLOWUP), BRANCH/'followup/result.json', BRANCH/'followup/transition.json']
    evidence = {'status': 'COMPLETE_BOUNDED_FIRST_VISUAL_CYCLE', 'seed': 42, 'condition': 'veto',
        'selections': selection['selections'], 'equivalent_decisions': selection['equivalent_decisions'],
        'development': result, 'evidence_sha256': {str(p): file_sha256(p) for p in paths},
        'budget': CompactExecutionBudget().compact_summary(), 'storage': storage_check(),
        'formal_expansion_admitted': False, 'scientific_method_verified': False,
        'scope': 'One real cycle; development measurement is not an independent comparative method result.'}
    write_new(directory/'result.json', evidence)
    return evidence


def run(directory):
    directory = Path(directory).resolve()
    if directory.exists() or not directory.is_relative_to(OUTPUT):
        raise ValueError('Require a fresh controller directory; inspect stopped work before recovery')
    search = admission()
    directory.mkdir()
    with (directory/'controller.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_new(directory/'start.json', {'pid': os.getpid(), 'admission': str(ADMISSION),
            'admission_sha256': file_sha256(ADMISSION), 'source_sha256': file_sha256(Path(__file__)),
            'search_plan_sha256': file_sha256(SEARCH/'plan.json'), 'no_api_or_cleanup': True})
        started = time.monotonic()
        try:
            while time.monotonic()-started < 8*3600:
                report = advance(directory, search)
                (directory/'status.json').write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps(report), flush=True)
                if report['status'] == 'COMPLETE_BOUNDED_FIRST_VISUAL_CYCLE':
                    return
                time.sleep(30)
            raise TimeoutError('Bounded wait expired; inspect the same jobs without automatic retry')
        except BaseException as exc:
            write_new(directory/'failure.json', {'status': 'STOPPED_REQUIRES_DIAGNOSIS',
                'error_type': type(exc).__name__, 'error': str(exc), 'no_automatic_retry': True})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
