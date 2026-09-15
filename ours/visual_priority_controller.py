"""Run the bounded visual E1 screen without a Chess continuation dependency.

Reuse the source-bound visual launcher and existing accounting. The screen is
not evidence for E2--E4 effectiveness and cannot activate the formal matrix.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil

from .veto_milestone_controller import Milestones, check, write_new
from .visual_task import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def execute_visual(controller, visual):
    controller.command('check_visual_service', 'ours.native_visual_service',
                       '--phase', 'check', '--plan', visual, visual=True)
    return controller.allocation('run_visual_service', 'ours/run_native_visual_service.sh',
                                 visual, gpus=1, seconds=2400, workspace_gib=2)


def prepare(path, output):
    legacy = ROOT / 'results/veto-milestone-plan-20260910-v1.json'
    interruption = ROOT / 'results/veto-chess-milestone-priority-interruption-20260910-v1.json'
    prior = json.loads(legacy.read_text())
    check(prior)
    visual = Path(prior['visual_plan'])
    assert not output.exists() and not path.exists()
    assert not Path(json.loads(visual.read_text())['output']).exists()
    inputs = [legacy, interruption, visual, Path(__file__).resolve()]
    plan = {
        'kind': 'storage_bounded_visual_priority_amendment',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User asks to prioritize direct visual validation and reiterates no additional space.',
        'legacy_milestone_plan': str(legacy), 'interruption': str(interruption),
        'visual_plan': str(visual), 'output': str(output),
        'input_sha256': {str(p): file_sha256(p) for p in inputs},
        'stages': ['check_visual_service', 'run_visual_service'],
        'chess_phase2': 'DEFERRED_NOT_A_VISUAL_PREREQUISITE',
        'formal_51_run_matrix': 'HISTORICAL_PLAN_NOT_CURRENT_EXECUTION_SCOPE',
        'storage': {'additional_existing_data_cleanup_authorized': False,
                    'mimic_images': 'RETAIN', 'available_bytes_at_prepare': shutil.disk_usage(ROOT).free,
                    'screen_working_space_estimate_gib': 2, 'minimum_free_margin_gib': 40,
                    'new_model_checkpoints': 0},
        'resources': {'gpus': 1, 'time_limit_seconds': 2400, 'new_proposer_api_calls': 0,
                      'gpu_choice': 'Only 16 images; one existing-model replica avoids duplicate loading and storage. Scientific 2/4-GPU choices remain deferred.'},
        'scope': 'Eight existing engineering image pairs, current 4B baseline, real E1 responses only.',
        'next_research_gate': 'Select a storage-feasible development diagnostic after this screen; no automatic training, matrix expansion, or deletion.',
        'limitations': ['No training, harness search, scientific holdout, or VETO efficacy claim.',
                        'Account queue must be empty before submission; no search-completion prerequisite.',
                        'Retain all failed attempts and allocation charges; no automatic retries.'],
    }
    write_new(path, plan)
    return plan


def run(path):
    plan = json.loads(path.read_text())
    assert plan['kind'] == 'storage_bounded_visual_priority_amendment'
    assert plan['stages'] == ['check_visual_service', 'run_visual_service']
    for p, sha in plan['input_sha256'].items():
        assert file_sha256(Path(p)) == sha, p
    interrupted = json.loads(Path(plan['interruption']).read_text())
    try:
        os.kill(interrupted['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Old milestone controller is still alive; refuse competing submission')
    legacy = json.loads(Path(plan['legacy_milestone_plan']).read_text())
    check(legacy)
    root = Path(plan['output'])
    root.mkdir(exist_ok=False)
    write_new(root / 'plan.json', plan)
    # Only the reusable command/submission/accounting methods are called. The
    # legacy run() method and its Chess continuation stages are never invoked.
    controller = Milestones({**legacy, 'output': str(root)})
    with (root / 'controller.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            visual = Path(plan['visual_plan'])
            job, terminal = execute_visual(controller, visual)
            result_path = Path(json.loads(visual.read_text())['output']) / 'result.json'
            result = json.loads(result_path.read_text())
            assert result['status'] == 'COMPLETE_REAL_NATIVE_VISUAL_EVALUATION'
            assert result['job_id'] == job and result['plan_sha256'] == file_sha256(visual)
            write_new(root / 'result.json', {
                'status': 'COMPLETED_VISUAL_SCREEN_ONLY', 'visual_result': str(result_path),
                'visual_result_sha256': file_sha256(result_path), 'slurm_terminal': str(terminal),
                'budget': controller.budget.summary(), 'scientific_method_verified': False,
                'formal_matrix_started': False, 'limitations': plan['limitations']})
            controller.event('complete', status='COMPLETED_VISUAL_SCREEN_ONLY')
        except BaseException as error:
            write_new(root / 'failure.json', {'status': 'STOPPED_INCOMPLETE',
                'error_type': type(error).__name__, 'error': str(error),
                'budget': controller.budget.summary()})
            controller.event('stopped', status='STOPPED_INCOMPLETE', error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run'], required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.phase == 'prepare':
        if args.output is None:
            parser.error('--output is required for prepare')
        prepare(args.plan.resolve(), args.output.resolve())
        print('PREPARED_VISUAL_PRIORITY', flush=True)
    else:
        run(args.plan.resolve())
