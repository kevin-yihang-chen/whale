"""Compact experiment DAG and resource caps; not a new VETO component."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import shutil
import uuid

from .research_budget import ExecutionBudget
from .visual_task import file_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/fast-chart-20260915-v1'
PROTOCOL = ROOT / 'ours/fast_chart_protocol_20260915.md'
LEDGER = ROOT / 'data/veto-execution-budget-20260910/ledger.jsonl'
SEEDS = (42, 43, 44)
CONDITIONS = ('weight_only', 'harness_only', 'whale', 'veto', 'marginal_gate', 'counterfactual_augmentation')
PHASE_CAPS = {k: Decimal(str(v)) for k, v in dict(setup=20, core=40, ablation=15, evaluation=20, retry=5).items()}


def schedule():
    rows = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            rows.append({'run_id': f'charts-4b-{condition}-seed{seed}', 'seed': seed,
                'condition': condition, 'status': 'PLANNED_NOT_EXECUTED',
                'first_stage_id': None if condition == 'harness_only' else f'common-theta1-seed{seed}',
                'search_id': (None if condition == 'weight_only' else
                    f'{"theta0" if condition == "harness_only" else "theta1"}-search-seed{seed}'),
                'acceptance_mode': {'veto': 'paired', 'marginal_gate': 'marginal_gate'}.get(condition, 'off'),
                'result_path': None, 'failure_path': None})
    return {'kind': 'compact_veto_research_matrix', 'protocol': str(PROTOCOL),
        'protocol_sha256': file_sha256(PROTOCOL), 'seeds': SEEDS, 'runs': rows,
        'logical_conditions': 18, 'common_first_stages': 3, 'trained_second_stages': 15,
        'native_batches_per_stage': 4, 'tasks_per_batch': 8, 'trajectories_per_task': 8,
        'candidate_attempts_per_search': 3, 'search_rounds': 1,
        'gpu_cap_hours': 100, 'includes_existing_pilot_costs': True,
        'phase_caps': {k: str(v) for k, v in PHASE_CAPS.items()},
        'storage_cap_gib': 400, 'api_global_cap_cny': 45,
        'paused': ['51-condition-matrix', 'chess', 'clevr', '2b'],
        'launch_gates': ['shared_scoring', 'corrected_visual_lineage', 'frozen_h0_lr', 'measured_throughput', 'storage_capacity'],
        'scientific_status': 'UNVALIDATED', 'goal_complete': False}


class CompactExecutionBudget(ExecutionBudget):
    def __init__(self, path=LEDGER):
        super().__init__(path)

    @staticmethod
    def phase_balances(rows):
        pending, charged, jobs = ExecutionBudget.balances(rows)
        reservations = {r['id']: r for r in rows if r['event'] == 'reserve'}
        used, held = dict.fromkeys(PHASE_CAPS, Decimal(0)), dict.fromkeys(PHASE_CAPS, Decimal(0))
        for row in rows:
            if row['event'] == 'settle':
                phase = reservations[row['id']].get('fast_phase', 'setup')
                used[phase] += Decimal(row['gpu_hours'])
        for row in pending.values():
            held[row.get('fast_phase', 'setup')] += Decimal(row['gpu_hours'])
        return used, held

    def reserve(self, *, phase, gpus, time_limit_seconds, purpose):
        if phase not in PHASE_CAPS or type(gpus) is not int or not 1 <= gpus <= 4:
            raise ValueError('Invalid compact phase or GPU count')
        if type(time_limit_seconds) is not int or time_limit_seconds <= 0:
            raise ValueError('Time limit must be a positive integer')
        amount = Decimal(gpus * time_limit_seconds) / 3600
        with self.locked() as (stream, rows):
            used, held = self.phase_balances(rows)
            if used[phase] + held[phase] + amount > PHASE_CAPS[phase] or sum(used.values()) + sum(held.values()) + amount > 100:
                raise ValueError('Compact phase or total GPU-hour cap would be exceeded')
            ident = uuid.uuid4().hex
            self.append(stream, {'event': 'reserve', 'id': ident, 'stage': 'pilot', 'fast_phase': phase,
                'gpus': gpus, 'time_limit_seconds': time_limit_seconds, 'gpu_hours': str(amount), 'purpose': purpose})
        return ident

    def compact_summary(self):
        with self.locked() as (_, rows):
            used, held = self.phase_balances(rows)
        return {'charged_gpu_hours': {k: str(v) for k, v in used.items()},
                'reserved_gpu_hours': {k: str(v) for k, v in held.items()},
                'total_cap': 100, 'remaining_unreserved': str(Decimal(100)-sum(used.values())-sum(held.values()))}


def storage_check(additional_gib=0):
    if additional_gib < 0:
        raise ValueError('Negative storage reservation')
    inodes = {}
    if OUTPUT.exists():
        for path in OUTPUT.rglob('*'):
            if path.is_symlink():
                raise ValueError('Unaccounted symlink in compact output root')
            if path.is_file():
                stat = path.stat()
                inodes[(stat.st_dev, stat.st_ino)] = stat.st_blocks * 512
    used = sum(inodes.values()) / 1024**3
    free = shutil.disk_usage(ROOT).free / 1024**3
    if used + additional_gib > 400 or free < additional_gib + 40:
        raise ValueError('Compact400GiB cap or40GiB free margin would be exceeded')
    return {'new_root_allocated_gib': used, 'requested_gib': additional_gib,
            'filesystem_free_gib': free, 'cap_gib': 400, 'minimum_free_margin_gib': 40,
            'hardlinks_counted_once_within_new_root': True, 'deletion_authorized': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {'schedule': schedule(), 'budget': CompactExecutionBudget().compact_summary(), 'storage': storage_check()}
    if args.output:
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print(json.dumps(report if not args.output else {'status': 'PLANNED_NOT_EXECUTED', 'output': str(args.output)}, indent=2))
