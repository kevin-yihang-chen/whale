"""Persist Slurm terminal evidence promptly and charge its actual GPU allocation.

Transient observation failures never trigger a new job. This observer records
allocation status only; successful Slurm exit is not a scientific result.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import subprocess
import sys
import time

from .research_budget import ExecutionBudget, terminal_allocation
from .audit_native_training_batch import require
from .visual_task import file_sha256


def observe(receipt_path):
    receipt = json.loads(receipt_path.read_text())
    root, job = receipt_path.parent, receipt['job_id']
    require(isinstance(job, str) and job.isdigit(), 'Validation failed: isinstance(job, str) and job.isdigit()')
    with (root / 'observer-start.json').open('x') as stream:
        json.dump({'pid': os.getpid(), 'job_id': job, 'receipt_sha256': file_sha256(receipt_path),
                   'source_sha256': file_sha256(Path(__file__))}, stream, indent=2)
    while True:
        result = subprocess.run(['scontrol', 'show', 'job', job], capture_output=True, text=True)
        if result.returncode:
            with (root / 'observation-errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(),
                    'returncode': result.returncode, 'stderr': result.stderr}) + '\n')
            time.sleep(30)
            continue
        allocation = terminal_allocation(result.stdout, job)
        if allocation is not None:
            break
        time.sleep(30)
    terminal_path = root / 'slurm-terminal.txt'
    with terminal_path.open('x') as stream:
        stream.write(result.stdout)
        stream.flush()
        os.fsync(stream.fileno())
    budget = ExecutionBudget(receipt['gpu_ledger'])
    budget.settle(receipt['reservation_id'], job_id=job, gpu_hours=allocation['gpu_hours'], terminal_path=terminal_path)
    record = {'job_id': job, 'state': allocation['state'], 'seconds': allocation['seconds'],
        'gpu_hours': str(allocation['gpu_hours']), 'terminal_sha256': file_sha256(terminal_path),
        'budget': budget.summary(), 'scientific_completion_verified_by_observer': False}
    with (root / 'allocation-result.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    observe(Path(sys.argv[1]).resolve())
