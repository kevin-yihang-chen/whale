"""Restore source asset absence and re-audit the completed untrained BF16 cast.

The source checkpoint has no generation_config.json. Transformers added one on
save, changing two diagnostic-output defaults (False versus None). Preserve that
generated file in the recovery receipt and restore source inference semantics.
No parameter file is changed or recomputed by this finalization.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from .alternation_training import write_new
from .audit_native_training_batch import require
from .canonical_initialization import audit_cast
from .local_completion import checkpoint_manifest
from .probe_updated_vllm import inference_contract
from .visual_task import file_sha256

ORIGINAL_PLAN = Path('results/canonical-initialization-plan-20260910.json')
FAILURE = Path('results/canonical-initialization-222794/result.json')


def checked_difference(base, target):
    from transformers import AutoConfig, GenerationConfig
    require(not (base / 'generation_config.json').exists(), 'Original generation asset is present')
    a = GenerationConfig.from_model_config(AutoConfig.from_pretrained(base, local_files_only=True)).to_dict()
    b = GenerationConfig.from_pretrained(target, local_files_only=True).to_dict()
    difference = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    require(difference == {'output_hidden_states': [False, None], 'output_attentions': [False, None]},
            'Unreviewed generation-configuration difference')
    return difference


def make_plan(path):
    original = json.loads(ORIGINAL_PLAN.read_text())
    failure = json.loads(FAILURE.read_text())
    require(failure['status'] == 'FAILED' and failure['plan_sha256'] == file_sha256(ORIGINAL_PLAN), 'Wrong failure source')
    sources = set(original['source_sha256']) | {str(ORIGINAL_PLAN), str(FAILURE),
        'ours/finalize_canonical_initialization.py', 'ours/run_finalize_canonical_initialization.sh'}
    target = Path(original['target'])
    difference = checked_difference(Path(original['base']['path']), target)
    write_new(path, {'kind': 'canonical_initialization_finalization_plan',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'original_plan': str(ORIGINAL_PLAN),
        'failed_result': str(FAILURE), 'target_before': checkpoint_manifest(target),
        'source_sha256': {name: file_sha256(Path(name)) for name in sorted(sources)},
        'generated_asset_sha256': file_sha256(target / 'generation_config.json'),
        'reviewed_config_difference': difference, 'new_model_calls': 0, 'optimizer_steps': 0,
        'resources': original['resources']})


def check(path):
    from importlib.metadata import version
    plan = json.loads(path.read_text())
    require(plan['kind'] == 'canonical_initialization_finalization_plan', 'Wrong finalization plan')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(Path(name)) == digest, f'Changed finalization source: {name}')
    original = json.loads(Path(plan['original_plan']).read_text())
    require({n: version(n) for n in original['versions']} == original['versions'], 'Changed runtime')
    require(checkpoint_manifest(Path(original['base']['path'])) == original['base'], 'Changed base')
    require(checkpoint_manifest(Path(original['target'])) == plan['target_before'], 'Changed converted model')
    expected_assets = {k: v for k, v in original['base']['assets'].items()
                       if k != 'model.safetensors.index.json'}
    actual_assets = {k: v for k, v in plan['target_before']['assets'].items()
                     if k != 'generation_config.json'}
    require(actual_assets == expected_assets, 'Unreviewed inference asset difference')
    require(checked_difference(Path(original['base']['path']), Path(original['target'])) ==
            plan['reviewed_config_difference'], 'Changed configuration difference')
    require(not Path(original['report']).exists(), 'Preserve existing final report')
    return plan, original


def run(args, plan, original):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Expected CPU Slurm wrapper')
    job, started = os.environ['SLURM_JOB_ID'], time.monotonic()
    root = Path(f'results/canonical-initialization-finalization-{job}')
    root.mkdir(exist_ok=False)
    base, target = Path(original['base']['path']), Path(original['target'])
    write_new(root / 'start.json', {'kind': 'initialization_finalization_start', 'job_id': job,
        'plan_sha256': file_sha256(args.plan), 'target_before': plan['target_before'],
        'new_model_calls': 0, 'optimizer_steps': 0})
    asset = target / 'generation_config.json'
    require(file_sha256(asset) == plan['generated_asset_sha256'], 'Generated asset changed')
    shutil.move(str(asset), root / 'removed-generation_config.json')
    review = json.loads(Path(original['graph_review']).read_text())
    audit = audit_cast(base, target, review['findings'])
    require(audit['original_fp32_elements'] == 3840 and audit['original_fp32_tensors'] == 48 and
            audit['original_bf16_changed_elements'] == 0, 'Unexpected base precision accounting')
    contract = inference_contract(base, target)
    require(checkpoint_manifest(base) == original['base'], 'Original model changed during audit')
    after = checkpoint_manifest(target)
    require(after['weights'] == plan['target_before']['weights'], 'Parameter files changed during finalization')
    expected_assets = dict(plan['target_before']['assets'])
    del expected_assets['generation_config.json']
    source_assets = {k: v for k, v in original['base']['assets'].items()
                     if k != 'model.safetensors.index.json'}
    require(after['assets'] == expected_assets == source_assets, 'Unexpected inference asset change')
    report = {'kind': 'canonical_untrained_initialization', 'status': 'PASS', 'job_id': job,
        'plan_sha256': file_sha256(args.plan), 'original_plan_sha256': file_sha256(Path(plan['original_plan'])),
        'base': original['base'], 'exported': after, 'audit': audit, 'inference_contract': contract,
        'optimizer_steps': 0, 'new_model_calls': 0, 'parameter_files_changed_during_finalization': False,
        'removed_asset_sha256': file_sha256(root / 'removed-generation_config.json'),
        'seconds': time.monotonic() - started, 'limitations': original['limitations']}
    write_new(Path(original['report']), report)
    write_new(root / 'measurement.json', {'status': 'PASS', 'report': original['report'],
        'report_sha256': file_sha256(Path(original['report'])), 'job_id': job})
    print(json.dumps({'status': 'PASS', 'report': original['report']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--write-plan', action='store_true')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.write_plan:
        make_plan(args.plan)
    else:
        plan, original = check(args.plan)
        if args.check_only:
            print(json.dumps({'status': 'PREFLIGHT_PASS', 'plan_sha256': file_sha256(args.plan)}))
        else:
            run(args, plan, original)
