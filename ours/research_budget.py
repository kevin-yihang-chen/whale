"""Durable GPU allocation caps for the approved VETO protocol.

Reserve the full Slurm time limit before submitting; ambiguous submissions keep
their reservation. Actual terminal allocation time, including failures, is the
charge. This is execution infrastructure, not an E1--E3 method component.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import re
import uuid

LIMITS = {'pilot': Decimal('100'), 'main': Decimal('280'),
          'mechanism': Decimal('160'), 'verification': Decimal('60')}


def terminal_allocation(raw, job_id):
    fields = dict(re.findall(r'(\w+)=([^\s]+)', raw))
    if fields.get('JobId') != str(job_id):
        raise ValueError('Wrong terminal Slurm identity')
    state = fields['JobState']
    if state in {'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED'}:
        return None
    if state not in {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}:
        raise ValueError('Unknown terminal Slurm state')
    runtime = fields['RunTime'].split('-')
    days = int(runtime[0]) if len(runtime) == 2 else 0
    hours, minutes, seconds = map(int, runtime[-1].split(':'))
    seconds += 60 * minutes + 3600 * hours + 86400 * days
    allocated = dict(v.split('=', 1) for v in fields.get('AllocTRES', '').split(',') if '=' in v)
    gpus = int(allocated.get('gres/gpu', 0))
    if seconds and gpus == 0:
        raise ValueError('Missing GPU allocation for a nonempty runtime')
    return {'state': state, 'gpus': gpus, 'seconds': seconds,
        'gpu_hours': Decimal(gpus * seconds) / 3600}


class ExecutionBudget:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with self.path.open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            rows = [json.loads(line) for line in stream]
            yield stream, rows

    @staticmethod
    def append(stream, record):
        record['time_utc'] = datetime.now(timezone.utc).isoformat()
        stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())

    @staticmethod
    def balances(rows):
        pending, charged, jobs, identifiers = {}, dict.fromkeys(LIMITS, Decimal(0)), {}, set()
        for row in rows:
            ident = row['id']
            if row['event'] == 'reserve':
                if ident in identifiers or row['stage'] not in LIMITS:
                    raise ValueError('Invalid allocation reservation history')
                identifiers.add(ident)
                pending[ident] = row
            elif row['event'] == 'bind':
                if ident not in pending or ident in jobs or row['job_id'] in jobs.values():
                    raise ValueError('Repeated or unreserved Slurm allocation')
                jobs[ident] = row['job_id']
            elif row['event'] == 'settle':
                if ident not in pending or jobs.get(ident) != row['job_id']:
                    raise ValueError('Terminal cost without its submitted allocation')
                reservation = pending.pop(ident)
                amount = Decimal(row['gpu_hours'])
                if not amount.is_finite() or amount < 0:
                    raise ValueError('Invalid terminal GPU allocation cost')
                charged[reservation['stage']] += amount
            else:
                raise ValueError('Unknown GPU budget event')
        return pending, charged, jobs

    def reserve(self, *, stage, gpus, time_limit_seconds, purpose):
        if stage != 'pilot':
            raise ValueError('Expansion remains closed until the scientific go decision is implemented and verified')
        if type(gpus) is not int or not 1 <= gpus <= 4 or type(time_limit_seconds) is not int or time_limit_seconds < 1:
            raise ValueError('Expected a bounded one-to-four-GPU allocation')
        amount = Decimal(gpus * time_limit_seconds) / 3600
        with self.locked() as (stream, rows):
            pending, charged, _ = self.balances(rows)
            held = sum((Decimal(r['gpu_hours']) for r in pending.values() if r['stage'] == stage), Decimal(0))
            total = sum(charged.values()) + sum((Decimal(r['gpu_hours']) for r in pending.values()), Decimal(0))
            if charged[stage] + held + amount > LIMITS[stage] or total + amount > sum(LIMITS.values()):
                raise ValueError('Approved GPU-hour budget exhausted or reserved')
            ident = uuid.uuid4().hex
            self.append(stream, {'event': 'reserve', 'id': ident, 'stage': stage,
                'gpus': gpus, 'time_limit_seconds': time_limit_seconds,
                'gpu_hours': str(amount), 'purpose': purpose})
        return ident

    def bind(self, ident, job_id):
        if not isinstance(job_id, str) or not job_id.isdigit():
            raise ValueError('Ambiguous submission; retain full reservation')
        with self.locked() as (stream, rows):
            pending, _, jobs = self.balances(rows)
            if ident not in pending or ident in jobs or job_id in jobs.values():
                raise ValueError('Invalid or repeated Slurm submission binding')
            self.append(stream, {'event': 'bind', 'id': ident, 'job_id': job_id})

    def settle(self, ident, *, job_id, gpu_hours, terminal_path):
        from .visual_task import file_sha256
        terminal_path = Path(terminal_path)
        terminal = terminal_allocation(terminal_path.read_text(), job_id)
        if terminal is None or abs(Decimal(str(terminal['gpu_hours'])) - Decimal(str(gpu_hours))) > Decimal('0.00000001'):
            raise ValueError('Cost must match an actual terminal Slurm record')
        with self.locked() as (stream, rows):
            pending, _, jobs = self.balances(rows)
            if ident not in pending or jobs.get(ident) != job_id:
                raise ValueError('Unbound or already settled allocation')
            if terminal['gpus'] not in (0, pending[ident]['gpus']):
                raise ValueError('Slurm GPU count differs from its reservation')
            self.append(stream, {'event': 'settle', 'id': ident, 'job_id': job_id,
                'gpu_hours': str(gpu_hours), 'terminal_path': str(terminal_path.resolve()),
                'terminal_sha256': file_sha256(terminal_path)})

    def summary(self):
        with self.locked() as (_, rows):
            pending, charged, jobs = self.balances(rows)
        return {'limits_gpu_hours': {k: str(v) for k, v in LIMITS.items()},
            'charged_gpu_hours': {k: str(v) for k, v in charged.items()},
            'reserved_gpu_hours': {k: str(sum((Decimal(r['gpu_hours']) for r in pending.values() if r['stage'] == k), Decimal(0))) for k in LIMITS},
            'pending_allocations': [{**r, 'job_id': jobs.get(i)} for i, r in pending.items()],
            'scope': 'New allocations under the user-approved 2026-09-10 VETO plan; prior costs remain in their original ledger.',
            'expansion_enabled': False}
