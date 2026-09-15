"""Repeat fixed E1 evaluations to measure inference variability before attribution.

Each model answers the entire same development subset once more, without
training or selecting a preferred replicate. Reuse the two actual evaluators;
this is a repeatability diagnostic, not an independent dataset or seed trial.
"""
import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from . import native_visual_followup as followup
from . import native_visual_service as native
from . import visual_development_screen as development
from .evidence import fingerprint
from .local_completion import checkpoint_manifest
from .visual_task import file_sha256


def case_for(plan, label):
    item = plan['cases'][label]
    assert file_sha256(Path(item['original_plan'])) == item['original_plan_sha256']
    case = deepcopy(followup.read(item['original_plan']))
    output = Path(plan['output']) / label
    case['output'] = str(output)
    case['config']['data']['cache_dir'] = str(output / 'dataset-cache')
    case['config']['actor_rollout_ref']['rollout']['trace']['experiment_name'] = output.name
    case['config']['trainer']['experiment_name'] = output.name
    case['source_sha256'].update(plan['source_sha256'])
    case['decode_sha256'] = fingerprint({'config': case['config'], 'mode': case['mode'],
                                        'model_assets': case['model']['assets']})
    case['limitations'] = plan['limitations']
    return case


def check(path):
    plan = followup.read(path)
    assert plan['kind'] == 'visual_checkpoint_repeatability' and tuple(plan['cases']) == ('initial', 'trained')
    followup.verify_files(plan['source_sha256'])
    for label, item in plan['cases'].items():
        case = case_for(plan, label)
        followup.verify_files(case['source_sha256'])
        assert case['mode'] == 'direct' and case['role'] == 'V' and case['bounds']['images'] == 128
        assert checkpoint_manifest(Path(case['model']['path'])) == case['model']
        assert file_sha256(Path(item['original_result'])) == item['original_result_sha256']
        assert file_sha256(Path(case['manifest'])) == case['manifest_sha256']
    return plan


def run(path):
    plan = check(path)
    root = Path(plan['output'])
    root.mkdir()
    native.write_new(root / 'start.json', {'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(path)})
    try:
        comparisons = {}
        for label, item in plan['cases'].items():
            case = case_for(plan, label)
            case_path = root / (label + '-plan.json')
            native.write_new(case_path, case)
            with (root / (label + '.log')).open('x') as stream:
                subprocess.run([sys.executable, '-m', 'ours.visual_followup_repeatability', '--phase', 'case',
                    '--plan', str(path), '--label', label], check=True, stdout=stream, stderr=subprocess.STDOUT)
            original = followup.read(item['original_result'])
            repeated = followup.read(root / label / 'evaluation/result.json')
            old = {r['sample_id']: r['committed_answer'] for r in original['records']}
            new = {r['sample_id']: r['committed_answer'] for r in repeated['records']}
            assert len(old) == len(new) == 128 and set(old) == set(new)
            matched = followup.matched_results(Path(item['original_result']).parent.parent,
                root / label, Path(case['manifest']))
            matched['limitations'] = plan['limitations']
            native.write_new(root / (label + '-comparison.json'), matched)
            comparisons[label] = {'original': {k: original[k] for k in ('marginal_accuracy', 'paired_accuracy')},
                'repeated': {k: repeated[k] for k in ('marginal_accuracy', 'paired_accuracy')},
                'changed_answers': sum(old[k] != new[k] for k in old),
                'actual_input_and_decoding_multisets_equal': matched['actual_input_and_decoding_multisets_equal'],
                'comparison_sha256': file_sha256(root / (label + '-comparison.json'))}
        native.write_new(root / 'result.json', {'status': 'COMPLETE_VISUAL_CHECKPOINT_REPEATABILITY',
            'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(path), 'comparisons': comparisons,
            'new_optimizer_steps': 0, 'new_model_checkpoints': 0, 'api_calls': 0,
            'limitations': plan['limitations']})
        print(json.dumps(comparisons), flush=True)
    except BaseException as error:
        native.write_new(root / 'failure.json', {'status': 'INCOMPLETE', 'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=('run', 'case', 'check'), required=True)
    parser.add_argument('--label', choices=('initial', 'trained'))
    args = parser.parse_args()
    if args.phase == 'run':
        run(args.plan.resolve())
    elif args.phase == 'check':
        check(args.plan.resolve())
    else:
        plan = followup.read(args.plan)
        case = case_for(plan, args.label)
        case_path = Path(plan['output']) / (args.label + '-plan.json')
        assert case == followup.read(case_path)
        evaluator = development.evaluate if args.label == 'initial' else followup.evaluate
        asyncio.run(evaluator(case, case_path))
