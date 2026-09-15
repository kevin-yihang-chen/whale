"""Account for one explicitly submitted visual screen, including failure costs."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from .research_budget import ExecutionBudget, terminal_allocation
from .visual_task import file_sha256
from .veto_milestone_controller import write_new


def observe(receipt_path):
    receipt = json.loads(receipt_path.read_text())
    root = receipt_path.parent
    job = receipt['job_id']
    assert re.fullmatch(r'[0-9]+', job)
    budget = ExecutionBudget(receipt['gpu_ledger'])
    while True:
        raw = subprocess.check_output(['scontrol', 'show', 'job', job], text=True)
        terminal = terminal_allocation(raw, job)
        if terminal is not None:
            break
        time.sleep(30)
    terminal_path = root / 'slurm-terminal.txt'
    with terminal_path.open('x') as f:
        f.write(raw)
    budget.settle(receipt['reservation_id'], job_id=job,
                  gpu_hours=terminal['gpu_hours'], terminal_path=terminal_path)
    plan_path = Path(receipt['visual_plan'])
    assert file_sha256(plan_path) == receipt['visual_plan_sha256']
    plan = json.loads(plan_path.read_text())
    visual_path = Path(plan['output']) / 'result.json'
    success = terminal['state'] == 'COMPLETED' and re.search(r'\bExitCode=0:0(?:\s|$)', raw)
    value = {'status': 'INCOMPLETE_VISUAL_SCREEN', 'job_id': job,
             'slurm_state': terminal['state'], 'gpu_hours': str(terminal['gpu_hours']),
             'budget': budget.summary(), 'formal_matrix_started': False}
    if success and visual_path.exists():
        visual = json.loads(visual_path.read_text())
        assert visual['status'] == 'COMPLETE_REAL_NATIVE_VISUAL_EVALUATION'
        assert visual['job_id'] == job and visual['plan_sha256'] == file_sha256(plan_path)
        value.update(status='COMPLETED_VISUAL_SCREEN_ONLY', visual_result=str(visual_path),
                     visual_result_sha256=file_sha256(visual_path),
                     paired_accuracy=visual['paired_accuracy'],
                     marginal_accuracy=visual['marginal_accuracy'],
                     scientific_method_verified=False)
    write_new(root / 'result.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    observe(Path(sys.argv[1]).resolve())
