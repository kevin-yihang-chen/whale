"""Submit one bounded compact allocation and immediately preserve its terminal state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .fast_chart_protocol import ROOT, OUTPUT, LEDGER, PROTOCOL, CompactExecutionBudget, storage_check
from .visual_task import file_sha256


def submit(path, phase, kind='evaluation'):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    if kind == 'evaluation' and plan['kind'] == 'compact_initial_harness_calibration':
        from .fast_chart_evaluation import check, PROMPTS
        if tuple(plan['cases']) != tuple(PROMPTS):
            raise ValueError('Changed calibration candidates')
        for case in plan['cases'].values():
            if file_sha256(Path(case['plan'])) != case['sha256']:
                raise ValueError('Changed case identity')
            check(Path(case['plan']))
        wrapper = ROOT / 'ours/run_fast_chart_evaluation.sh'
        expected, extra, working = (1,12,5400), [], 12
        purpose = 'Shared chart h0 calibration,3x512 V images'
    elif kind == 'selection-probe' and plan['kind'] == 'compact_selection_probe_suite':
        from .chart_selection_diagnostic import check
        check(path)
        if phase != 'core': raise ValueError('Selection diagnostic uses the existing core allocation')
        wrapper = ROOT / 'ours/run_chart_selection_diagnostic.sh'
        expected, extra, working = (1,12,7200), [], 12
        purpose = 'Exploratory theta1 h1/h3 R512 selection diagnostic; no new weights or API'
    elif kind == 'throughput' and plan['kind'] == 'compact_chart_throughput_calibration':
        from .fast_chart_throughput import check
        check(path)
        wrapper = ROOT / 'ours/run_fast_chart_throughput.sh'
        expected, extra, working = (1,12,3600), [], 12
        purpose = 'Shared V inference concurrency calibration,batch16/32,complete512 images each'
    elif kind == 'training' and plan['kind'] == 'compact_chart_rsft_stage':
        from .fast_chart_training import check
        check(path)
        wrapper = ROOT / 'ours/run_fast_chart_training.sh'
        expected, extra, working = (2,24,3600), [str(plan['seed'])], 35
        purpose = f"Compact E4 stage{plan['stage']} seed{plan['seed']} lr{plan['rate']},256 trajectories"
    elif kind == 'followup' and plan['kind'] == 'compact_chart_native_followup':
        from .fast_chart_followup import check
        check(path)
        wrapper = ROOT / 'ours/run_fast_chart_followup.sh'
        expected, extra, working = (1,12,3600), [], 25
        purpose = 'Full native parameter comparison, canonical serving export and512-image V calibration'
    elif kind == 'search' and plan['kind'] == 'compact_chart_search_evaluation':
        from .fast_chart_search_evaluation import check
        check(path)
        wrapper=ROOT/'ours/run_fast_chart_search_evaluation.sh'
        expected,extra,working=(1,12,3600),[],12
        purpose=f"One fixed chart candidate {plan['candidate']},H128+C64pairs"
    elif kind == 'endpoint' and plan['kind']=='compact_registered_endpoint_evaluation':
        from .fast_chart_endpoint_evaluation import check
        check(path)
        wrapper=ROOT/'ours/run_fast_chart_endpoint_evaluation.sh'
        expected,extra,working=(1,12,plan['bounds']['time_limit_seconds']),[],12
        if expected[2] not in (3600,7200):raise ValueError('Unregistered endpoint evaluation allocation')
        purpose=f"Registered {plan['condition']} seed{plan['seed']} {plan['partition']} endpoint"
    else:
        raise ValueError('Unknown implemented compact allocation kind')
    bounds = plan['bounds']
    if (bounds['gpus'], bounds['cpus'], bounds['time_limit_seconds']) != expected:
        raise ValueError('Allocation differs from the reviewed Slurm wrapper')
    queue = subprocess.run(['squeue', '-u', 'yihangc', '-h', '-o', '%i'], capture_output=True, text=True, check=True)
    if queue.stdout.strip():
        raise ValueError('An existing allocation occupies the one-job account limit')
    directory = OUTPUT / 'allocations' / path.stem
    directory.mkdir(parents=True, exist_ok=False)
    space = storage_check(working)
    budget = CompactExecutionBudget()
    ident = budget.reserve(phase=phase, gpus=expected[0], time_limit_seconds=expected[2], purpose=purpose)
    receipt = {'plan': str(path), 'plan_sha256': file_sha256(path), 'protocol_sha256': file_sha256(PROTOCOL),
        'gpu_ledger': str(LEDGER), 'reservation_id': ident, 'phase': phase, 'gpus': expected[0],
        'time_limit_seconds': expected[2], 'wrapper_sha256': file_sha256(wrapper), 'storage': space,
        'gpu_choice': 'Native RSFT uses one actor and one rollout GPU; standalone inference uses one GPU with plan-bound concurrency.',
        'notifications': {'mail_type': 'ALL', 'mail_user': 'yihangc@connect.hku.hk', 'delivery_verified': False}}
    receipt_path = directory / 'submission.json'
    def save():
        with receipt_path.open('w') as stream:
            json.dump(receipt, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    save()
    # Command-line limit binds the same full reservation even when a shared
    # wrapper has a larger default for another registered evaluation size.
    result = subprocess.run(['sbatch', '--parsable','--time',str(expected[2]//60),str(wrapper),str(path),*extra], capture_output=True,text=True)
    receipt.update(submission_returncode=result.returncode, submission_stdout=result.stdout, submission_stderr=result.stderr)
    save()
    job = result.stdout.strip().split(';')[0]
    if result.returncode or not job.isdigit():
        raise RuntimeError('Submission is failed or ambiguous; reservation retained, no automatic resubmission')
    budget.bind(ident, job)
    receipt['job_id'] = job
    save()
    with (directory / 'observer.log').open('x') as log:
        observer = subprocess.Popen([sys.executable, '-m', 'ours.observe_research_allocation', str(receipt_path)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with (directory / 'observer-process.json').open('x') as stream:
        json.dump({'pid': observer.pid, 'job_id': job}, stream)
    print(json.dumps({'status': 'SUBMITTED', 'job_id': job, 'receipt': str(receipt_path),
                      'budget': budget.compact_summary()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=('setup','core','ablation','evaluation','retry'), required=True)
    parser.add_argument('--kind', choices=('evaluation','training','followup','search','endpoint','throughput','selection-probe'), default='evaluation')
    args = parser.parse_args()
    submit(args.plan, args.phase, args.kind)
