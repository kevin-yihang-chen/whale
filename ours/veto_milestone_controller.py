"""Budgeted sequential handoff from the approved Chess closure to visual E1.

This executes existing, source-bound entrypoints. It does not enable the formal
matrix, retry failures, open heldout data or change native E4 configurations.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .research_budget import ExecutionBudget, terminal_allocation
from .visual_task import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('ours/veto_milestone_controller.py', 'ours/research_budget.py',
    'ours/search_completion.py', 'ours/joint_training_phase2.py',
    'ours/run_joint_training_phase2.sh', 'ours/audit_joint_training_phase2.py',
    'ours/joint_training_phase2_result.py', 'ours/joint_checkpoint_export.py',
    'ours/run_joint_checkpoint_export.sh', 'ours/native_visual_service.py',
    'ours/run_native_visual_service.sh')


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(path, *, search_plan, recovery_receipt, visual_plan, output):
    paths = [Path(p).resolve() for p in (search_plan, recovery_receipt, visual_plan)]
    search, recovery, visual = [json.loads(p.read_text()) for p in paths]
    if search['kind'] != 'controlled_joint_staged_mh_plan' or search['condition'] != 'whale' or search['seed'] != 42:
        raise ValueError('Only the approved seed42 WHALE closure is in scope')
    if visual['role'] != 'engineering' or type(recovery['pid']) is not int:
        raise ValueError('Expected existing recovery and engineering visual plan')
    output = Path(output).resolve()
    if Path(path).exists() or output.exists():
        raise ValueError('Preserve earlier milestone attempts')
    plan = {'kind': 'veto_pilot_sequential_milestones', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'search_plan': str(paths[0]), 'recovery_receipt': str(paths[1]), 'visual_plan': str(paths[2]),
        'output': str(output), 'python': str(Path(sys.executable).absolute()),
        'gpu_ledger': str(ROOT/'data/veto-execution-budget-20260910/ledger.jsonl'),
        'input_sha256': {str(p): file_sha256(p) for p in paths},
        'source_sha256': {n: file_sha256(ROOT/n) for n in SOURCES},
        'stages': ['wait_existing_search', 'certify_search', 'prepare_native_phase2',
            'run_native_phase2', 'audit_native_phase2', 'certify_native_phase2',
            'prepare_canonical_export', 'run_canonical_export', 'certify_canonical_export',
            'check_visual_service', 'run_visual_service'],
        'limits': {'account_active_jobs': 1, 'pilot_gpu_hours': 100, 'free_workspace_margin_gib': 40},
        'limitations': ['No automatic retry or resubmission after interruption.',
            'Ambiguous submissions retain the full reservation for reconciliation.',
            'Native phase2 has its original Adam-reset contract.',
            'Canonical phase2 export is not a new real-server weight inspection.',
            'Visual service uses engineering pairs and performs no parameter update.',
            'The scientific go decision and all 51 formal conditions remain unexecuted.']}
    write_new(path, plan)
    return plan


def check(plan):
    if plan['kind'] != 'veto_pilot_sequential_milestones':
        raise ValueError('Wrong milestone plan')
    for name, digest in {**plan['input_sha256'], **plan['source_sha256']}.items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError(f'Changed milestone input: {name}')


def active_jobs(raw):
    values = raw.split()
    if any(re.fullmatch(r'[0-9]+(?:_[0-9]+)?', value) is None for value in values):
        raise ValueError('Ambiguous account queue response')
    return values


def recovery_finished(state, *, process_alive):
    if state.get('status') == 'FAILED':
        raise ValueError('Existing Chess recovery failed; preserve it for diagnosis')
    if not process_alive and state.get('status') != 'COMPLETED':
        raise ValueError('Recovery stopped without a completed search')
    return state.get('status') == 'COMPLETED' and not process_alive


class Milestones:
    def __init__(self, plan):
        self.plan = plan
        self.root = Path(plan['output'])
        self.budget = ExecutionBudget(plan['gpu_ledger'])

    def event(self, stage, **fields):
        value = {'time_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **fields}
        with (self.root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(value, allow_nan=False), flush=True)

    def queue(self):
        return active_jobs(subprocess.check_output(['squeue', '-h', '-u', 'yihangc', '-o', '%i'], text=True))

    def idle(self):
        while self.queue():
            time.sleep(30)

    def wait_search(self):
        search = json.loads(Path(self.plan['search_plan']).read_text())
        pid = json.loads(Path(self.plan['recovery_receipt']).read_text())['pid']
        self.event('wait_existing_search', recovery_pid=pid)
        while True:
            state = json.loads((Path(search['phase_root'])/'state.json').read_text())
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            if recovery_finished(state, process_alive=alive):
                return
            time.sleep(30)

    def command(self, stage, module, *args, visual=False):
        check(self.plan)
        self.event(stage, status='STARTED')
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
            PYTHONPATH=':'.join(str(ROOT/p) for p in
                (['data/visual-runtime-overlay-v1'] if visual else []) +
                ['ours/compat', 'upstream/WHALE/domains/chess_puzzles', '.']))
        with (self.root/f'{stage}.log').open('x') as stream:
            subprocess.run([self.plan['python'], '-m', module, *map(str, args)], cwd=ROOT,
                env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        self.event(stage, status='COMPLETED')

    def allocation(self, stage, wrapper, plan_path, *, gpus, seconds, workspace_gib):
        self.idle()
        check(self.plan)
        if shutil.disk_usage(ROOT/'data').free < (workspace_gib+40)*1024**3:
            raise ValueError('Insufficient working space; explicit cleanup approval is required before deleting files')
        if self.queue():
            raise ValueError('Account slot changed before submission; no job submitted')
        ident = self.budget.reserve(stage='pilot', gpus=gpus, time_limit_seconds=seconds, purpose=stage)
        self.event(stage, status='RESERVED', reservation_id=ident)
        # No retry on errors: submission may have succeeded even if its reply was lost.
        raw = subprocess.check_output(['sbatch', '--parsable', str(ROOT/wrapper), str(plan_path)], cwd=ROOT, text=True)
        job = raw.strip()
        self.budget.bind(ident, job)
        write_new(self.root/f'{stage}.submission.json', {'job_id': job, 'reservation_id': ident,
            'plan': str(plan_path), 'plan_sha256': file_sha256(Path(plan_path)), 'wrapper_sha256': file_sha256(ROOT/wrapper)})
        self.event(stage, status='SUBMITTED', job_id=job)
        while True:
            raw = subprocess.check_output(['scontrol', 'show', 'job', job], text=True)
            terminal = terminal_allocation(raw, job)
            if terminal is not None:
                path = self.root/f'{stage}.slurm-terminal.txt'
                with path.open('x') as stream:
                    stream.write(raw)
                self.budget.settle(ident, job_id=job, gpu_hours=terminal['gpu_hours'], terminal_path=path)
                self.event(stage, status=terminal['state'], job_id=job, gpu_hours=str(terminal['gpu_hours']))
                if terminal['state'] != 'COMPLETED' or re.search(r'\bExitCode=0:0(?:\s|$)', raw) is None:
                    raise ValueError(f'Allocation {job} ended without successful completion')
                return job, path
            time.sleep(30)

    def run(self):
        self.root.mkdir(exist_ok=False)
        write_new(self.root/'plan.json', self.plan)
        with (self.root/'controller.lock').open('x') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.wait_search()
                proof, phase2 = self.root/'search-completion.json', self.root/'phase2-plan.json'
                self.command('certify_search', 'ours.search_completion', '--plan', self.plan['search_plan'], '--output', proof)
                self.command('prepare_native_phase2', 'ours.joint_training_phase2', '--phase', 'prepare', '--plan', phase2,
                    '--search-plan', self.plan['search_plan'], '--search-completion', proof)
                job, terminal = self.allocation('run_native_phase2', 'ours/run_joint_training_phase2.sh', phase2,
                    gpus=2, seconds=3600, workspace_gib=30)
                audit, result = self.root/'phase2-audit.json', self.root/'phase2-result.json'
                self.command('audit_native_phase2', 'ours.audit_joint_training_phase2', '--plan', phase2,
                    '--directory', ROOT/f'data/native-rsft/controlled-whale-seed42-phase2-{job}/audit', '--output', audit)
                self.command('certify_native_phase2', 'ours.joint_training_phase2_result', '--plan', phase2, '--audit', audit,
                    '--slurm', terminal, '--log', ROOT/f'results/controlled-joint-phase2-{job}.log', '--output', result)
                export = self.root/'phase2-export-plan.json'
                self.command('prepare_canonical_export', 'ours.joint_checkpoint_export', '--phase', 'prepare', '--plan', export,
                    '--training-plan', phase2, '--result', result)
                _, terminal = self.allocation('run_canonical_export', 'ours/run_joint_checkpoint_export.sh', export,
                    gpus=1, seconds=1200, workspace_gib=10)
                self.command('certify_canonical_export', 'ours.joint_checkpoint_export', '--phase', 'close', '--plan', export,
                    '--slurm', terminal, '--output', self.root/'phase2-export-result.json')
                visual = Path(self.plan['visual_plan'])
                self.command('check_visual_service', 'ours.native_visual_service', '--phase', 'check', '--plan', visual, visual=True)
                job, terminal = self.allocation('run_visual_service', 'ours/run_native_visual_service.sh', visual,
                    gpus=1, seconds=2400, workspace_gib=2)
                visual_result = Path(json.loads(visual.read_text())['output'])/'result.json'
                observed = json.loads(visual_result.read_text())
                if observed['status'] != 'COMPLETE_REAL_NATIVE_VISUAL_EVALUATION' or observed['job_id'] != job or observed['plan_sha256'] != file_sha256(visual):
                    raise ValueError('Visual completion does not match the submitted engineering plan')
                write_new(self.root/'result.json', {'status': 'COMPLETED_ENGINEERING_MILESTONES',
                    'visual_result': str(visual_result), 'visual_result_sha256': file_sha256(visual_result),
                    'budget': self.budget.summary(), 'limitations': self.plan['limitations']})
                self.event('complete', status='COMPLETED_ENGINEERING_MILESTONES')
            except BaseException as error:
                write_new(self.root/'failure.json', {'status': 'STOPPED_INCOMPLETE',
                    'error_type': type(error).__name__, 'error': str(error), 'budget': self.budget.summary()})
                self.event('stopped', status='STOPPED_INCOMPLETE', error_type=type(error).__name__, error=str(error))
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run'), required=True)
    for name in ('plan', 'search-plan', 'recovery-receipt', 'visual-plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=name == 'plan')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.phase == 'prepare':
        if any(v is None for v in (args.search_plan, args.recovery_receipt, args.visual_plan, args.output)):
            parser.error('Preparation requires all input paths and a fresh output directory')
        prepare(args.plan, search_plan=args.search_plan, recovery_receipt=args.recovery_receipt,
            visual_plan=args.visual_plan, output=args.output)
        print(json.dumps({'status': 'PREPARED', 'plan_sha256': file_sha256(args.plan)}), flush=True)
    else:
        plan = json.loads(args.plan.read_text())
        check(plan)
        Milestones(plan).run()
