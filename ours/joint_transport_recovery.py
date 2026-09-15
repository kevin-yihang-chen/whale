"""Audited continuation of the empty, timed-out fourth joint Chess proposal.

No completed candidate or evaluation is regenerated. One transport retry uses
the unchanged proposal request, within the original 60-request phase ceiling;
the unsuccessful attempt and unresolved CNY reservations remain intact. This
closes the original WHALE E3-to-E4 engineering path, not a VETO experiment.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from unittest.mock import patch

from .audit_native_training_batch import require
from .controlled_conditions import native_search_context
from .glm_gateway import BudgetJournal
from .joint_staged_search import JointStagedSearch, check_plan
from .native_search import tree_hashes, write_json
from .research_budget import ExecutionBudget, terminal_allocation
from .staged_candidate_recovery import compatible_proposal
from .staged_search import native_arguments
from .visual_task import file_sha256

SOURCES = ('ours/joint_transport_recovery.py', 'ours/research_budget.py')
API_LEDGER = Path('data/glm-budget/ledger.jsonl')
GPU_LEDGER = Path('data/veto-execution-budget-20260910/ledger.jsonl')


def request_ids():
    return [r['id'] for r in map(json.loads, API_LEDGER.read_text().splitlines()) if r['event'] == 'reserve']


def available_requests(snapshot, iteration, identifiers):
    require(iteration in (4, 5), 'Outside the two remaining frozen rounds')
    prefix = snapshot['ledger_request_ids']
    require(identifiers[:len(prefix)] == prefix, 'API reservation history changed')
    used = len(identifiers) - snapshot['reservations_before_search']
    cap = min(12, 60 - used - (12 if iteration == 4 else 0))
    require(cap > 0, 'Original 60-request phase ceiling exhausted')
    return cap


def provider_errors(raw):
    errors = []
    for line in raw.splitlines():
        if not line.startswith('data:') or line[5:].strip() == '[DONE]':
            continue
        event = json.loads(line[5:])
        if event.get('type') == 'error':
            errors.append(event.get('error', {}))
    return errors


def snapshot(plan, plan_path):
    root = Path(plan['phase_root']).resolve()
    state = json.loads((root/'state.json').read_text())
    require(state['status'] == 'FAILED' and state['error'] == 'No allocated candidate was created',
        'Expected the recorded empty fourth proposal')
    try:
        os.kill(state['controller_pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Original controller is still alive')
    require(not (root/'result.json').exists() and not (root/'recovery-transport-4').exists(), 'Already recovered or completed')
    evaluations = json.loads((root/'evaluations.json').read_text())
    expected = [f'h{i}' for i in range(10)]
    require([e['harness'] for e in evaluations] == expected and
        {p.name for p in (root/'evaluations').iterdir()} == set(expected), 'Expected exactly ten audited evaluations')
    controller = JointStagedSearch(plan, plan_path)
    controller.evaluations = evaluations
    controller.verify_public()
    for number in (1, 2, 3):
        comparison = json.loads((root/f'search/logs/iteration_{number:03d}/comparison.json').read_text())
        require(comparison['iteration'] == number and comparison['early_stop'] is None, 'Incomplete preceding round')
    require((root/'search/logs/accepted_harness.txt').read_text().strip() == comparison['accepted_harness'], 'Changed incoming selection')
    request = json.loads((root/'proposal-request-4.json').read_text())
    require(request['iteration'] == 4 and request['next_names'] == ['h10', 'h11', 'h12'], 'Changed fourth allocation')
    failed = root/'proposal-4'
    receipt = json.loads((failed/'proposer-result.json').read_text())
    require(receipt['exit_code'] == 124 and receipt['iteration'] == 4 and
        receipt['allocated_slots'] == request['next_names'], 'Not a timed-out fourth attempt')
    search = tree_hashes(root/'search')
    view = tree_hashes(failed/'proposer-workspace', ignore_cli_state=True)
    protected = {k:v for k,v in search.items() if k != 'pending_eval.json'}
    require(all(view.get(k) == v for k,v in protected.items()), 'Failed proposer changed protected evidence')
    require(all(k.startswith('logs/claude_sessions/') for k in view.keys() - protected.keys()),
        'Timed-out proposer produced unreviewed artifacts')
    require({p.name for p in (root/'search/harnesses').iterdir()} == set(expected), 'Unaccounted candidate')
    identifiers = request_ids()
    captures = {p.stem for directory in root.glob('proposal-[1-4]') for p in (directory/'raw-sse').glob('*.sse')}
    require(captures and captures <= set(identifiers), 'Provider calls lack budget reservations')
    offset = min(identifiers.index(ident) for ident in captures)
    require(set(identifiers[offset:]) == captures, 'Unaccounted request in interrupted search')
    errors = []
    for path in (failed/'raw-sse').glob('*.sse'):
        errors.extend(provider_errors(path.read_text()))
    require(errors and all(e.get('message') == 'Internal Network Failure' for e in errors),
        'Expected the observed provider network failures')
    result = {'failed_state': state, 'search_sha256': search, 'proposal_sha256': view,
        'failed_capture_sha256': tree_hashes(failed/'raw-sse'), 'provider_network_errors': len(errors),
        'receipt_sha256': file_sha256(failed/'proposer-result.json'),
        'request_sha256': file_sha256(root/'proposal-request-4.json'),
        'evaluations_sha256': file_sha256(root/'evaluations.json'),
        'ledger_request_ids': identifiers, 'reservations_before_search': offset,
        'requests_used': len(identifiers) - offset,
        'native_config_sha256': file_sha256(root/'native-config.json')}
    result['retry_request_cap'] = available_requests(result, 4, identifiers)
    return result


def prepare(plan_path, amendment_path):
    require(not amendment_path.exists(), 'Preserve the existing amendment')
    plan = check_plan(plan_path)
    record = {'kind': 'joint_fourth_proposal_transport_recovery',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'original_plan': str(plan_path.resolve()), 'original_plan_sha256': file_sha256(plan_path),
        'source_sha256': {str(Path(p).resolve()): file_sha256(Path(p)) for p in SOURCES},
        'snapshot': snapshot(plan, plan_path),
        'change': 'One new attempt for empty timed-out iteration 4; same prompt, model, timeout, slots and candidate validation.',
        'continuation': 'Native start_iteration=4, iterations=2; reuse all h0-h9 evaluations.',
        'api_phase_ceiling': 60, 'gpu_ledger': str(GPU_LEDGER.resolve()),
        'limitations': ['Transport retry is an explicit execution amendment; it does not replace a generated candidate.',
            'Failed provider requests retain their full unresolved CNY reservations.',
            'Per-attempt request cap is reduced to reserve room for the final round; the total phase ceiling is unchanged.',
            'Another failure is retained and stops this controller; no automatic retry loop.',
            'The original plan and all earlier candidate, proposal and evaluation evidence remain unchanged.']}
    write_json(amendment_path, record)
    return record


def check(amendment_path):
    record = json.loads(amendment_path.read_text())
    require(record['kind'] == 'joint_fourth_proposal_transport_recovery', 'Wrong recovery kind')
    path = Path(record['original_plan'])
    require(file_sha256(path) == record['original_plan_sha256'], 'Original plan changed')
    require(record['source_sha256'] == {str(Path(p).resolve()): file_sha256(Path(p)) for p in SOURCES}, 'Recovery source changed')
    plan = check_plan(path)
    require(snapshot(plan, path) == record['snapshot'], 'Interrupted evidence changed')
    return record, plan, path


class TransportRecoveredSearch(JointStagedSearch):
    def evaluate(self, **kwargs):
        budget = ExecutionBudget(GPU_LEDGER)
        original = __import__('subprocess').check_output
        reservation = {}
        def submit(argv, *args, **options):
            if argv[:2] == ['sbatch', '--parsable']:
                require(not reservation, 'Repeated allocation submission')
                ident = budget.reserve(stage='pilot', gpus=2, time_limit_seconds=2400,
                    purpose=f"Chess seed42 continuation {Path(kwargs['harness_path']).parent.name}")
                reservation['id'] = ident
                raw = original(argv, *args, **options)
                reservation['job_id'] = raw.strip()
                budget.bind(ident, raw.strip())
                return raw
            return original(argv, *args, **options)
        try:
            with patch('subprocess.check_output', submit):
                return super().evaluate(**kwargs)
        finally:
            private = self.root/'evaluations'/Path(kwargs['harness_path']).parent.name
            if reservation.get('job_id') and (private/'slurm-terminal.txt').exists():
                terminal = terminal_allocation((private/'slurm-terminal.txt').read_text(), reservation['job_id'])
                if terminal is not None:
                    budget.settle(reservation['id'], job_id=reservation['job_id'],
                        gpu_hours=terminal['gpu_hours'], terminal_path=private/'slurm-terminal.txt')

    def run_recovered(self, amendment, amendment_path):
        recovery = self.root/'recovery-transport-4'
        recovery.mkdir(exist_ok=False)
        (recovery/'amendment.json').write_bytes(amendment_path.read_bytes())
        (recovery/'failed-state.json').write_bytes((self.root/'state.json').read_bytes())
        self.evaluations = json.loads((self.root/'evaluations.json').read_text())
        journal = BudgetJournal(API_LEDGER)
        self.state('RECOVERING_EMPTY_FOURTH_PROPOSAL', amendment_sha256=file_sha256(amendment_path))
        args = native_arguments(self.plan, self.root)
        args.start_iteration, args.iterations = 4, 2
        require(args.early_stop_min_iters == 0, 'Cannot reset active patience')
        require(file_sha256(self.root/'native-config.json') == amendment['snapshot']['native_config_sha256'], 'Changed native config')
        with native_search_context(self.plan['condition'], self.root, self.plan['incoming_harness']) as (native, benchmark, provenance):
            original_sweep = native.run_sweep
            def sweep(config, harnesses, logs_dir, **kwargs):
                self.verify_public()
                rows = original_sweep(config, harnesses, logs_dir, **kwargs)
                require(len(rows) == len(harnesses) and all(ok for _,ok in rows), 'Native evaluation failed')
                require(all(n in {v['harness'] for v in self.evaluations} for n,_ in harnesses), 'Unaudited native score')
                self.verify_public()
                return rows
            def propose(**kwargs):
                self.verify_public()
                number = kwargs['iteration']
                cap = available_requests(amendment['snapshot'], number, request_ids())
                request = self.root/f'proposal-request-{number}.json'
                serial = {k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()}
                if number == 4:
                    require(serial == json.loads(request.read_text()), 'Retry changed the original request')
                    require(tree_hashes(self.root/'search') == amendment['snapshot']['search_sha256'], 'Retry inputs changed')
                    destination = recovery/'retry-proposal-4'
                else:
                    require(number == 5 and not request.exists(), 'Repeated final proposal')
                    write_json(request, serial)
                    destination = self.root/'proposal-5'
                self.state('PROPOSING', iteration=number, slots=kwargs['next_names'], max_api_requests=cap, budget=journal.summary())
                result = compatible_proposal(native, dict(self.plan, max_api_requests=cap), destination, journal, **kwargs)
                require(len(request_ids()) - amendment['snapshot']['reservations_before_search'] <= 60, 'Phase request ceiling exceeded')
                return result
            try:
                with patch.object(benchmark, 'evaluate_harness', self.evaluate), patch.object(native, 'run_sweep', sweep), \
                        patch.object(native, 'propose_claude_with_retries', propose):
                    native.run_evolve(args)
                self.verify_public()
                accepted = native.get_accepted_harness(self.root/'search')
                write_json(self.root/'result.json', {'status': 'COMPLETED_NATIVE_SEARCH', 'plan_sha256': file_sha256(self.plan_path),
                    'recovery_amendment_sha256': file_sha256(amendment_path), 'accepted': accepted,
                    'accepted_harness_sha256': file_sha256(self.root/f'search/harnesses/{accepted}/harness.py'),
                    'search_sha256': tree_hashes(self.root/'search'), 'evaluations': self.evaluations,
                    'budget': journal.summary(), 'provenance': provenance,
                    'limitations': self.plan['limitations'] + amendment['limitations']})
                self.state('COMPLETED', accepted=accepted, recovery_amendment_sha256=file_sha256(amendment_path))
            except BaseException as error:
                self.state('FAILED', error_type=type(error).__name__, error=str(error), budget=journal.summary())
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'check', 'coordinate'), required=True)
    parser.add_argument('--amendment', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--submit-evaluations', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        require(args.plan is not None, 'Original plan required')
        report = prepare(args.plan, args.amendment)
        print(json.dumps({'status': 'PREPARED', 'prior_api_requests': report['snapshot']['requests_used'],
            'retry_request_cap': report['snapshot']['retry_request_cap']}), flush=True)
    else:
        record, plan, path = check(args.amendment)
        if args.phase == 'check':
            print(json.dumps({'status': 'PASS', 'amendment_sha256': file_sha256(args.amendment)}), flush=True)
        else:
            require('SLURM_JOB_ID' not in os.environ, 'Coordinator requires the networked login node')
            TransportRecoveredSearch(plan, path, submit=args.submit_evaluations).run_recovered(record, args.amendment)
