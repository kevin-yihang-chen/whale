"""Construct the common F0 theta0 = Q_BF16(theta_pretrained), without updates.

All four controls start from the same reviewed active graph, inference contract
and serialization precision. Every exported value is checked against its source
BF16 cast. This is experimental control, not a proposed learning mechanism.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time

from .alternation_training import write_new
from .audit_native_training_batch import require
from .local_completion import checkpoint_manifest
from .probe_updated_vllm import inference_contract
from .verify_native_transition import comparison_names, summary, tensor_delta, verify_aliases
from .visual_task import file_sha256

REVIEW = Path('results/native-checkpoint-graph-review-20260909.json')
BASE_TRANSITION = Path('results/native-checkpoint-transition-222156.json')
SOURCES = ('ours/canonical_initialization.py', 'ours/run_canonical_initialization.sh',
    'ours/tests/test_canonical_initialization.py', 'ours/verify_native_transition.py',
    'ours/probe_updated_vllm.py', 'ours/local_completion.py', 'ours/visual_task.py',
    'ours/audit_native_training_batch.py', 'ours/alternation_training.py')


def audit_cast(base, exported, findings):
    """Require complete graph coverage and exact BF16 conversion, then measure it."""
    import torch
    from safetensors import safe_open
    with ExitStack() as stack:
        maps = []
        for root in (base, exported):
            mapping = {}
            for path in sorted(root.glob('*.safetensors')):
                reader = stack.enter_context(safe_open(str(path), framework='pt', device='cpu'))
                for name in reader.keys():
                    require(name not in mapping, 'Duplicate parameter across shards')
                    mapping[name] = reader
            maps.append(mapping)
        original, target = maps
        names = comparison_names(original, target, findings['base_only_inactive_mtp_names'], findings['alias'])
        require(len(names) == findings['independent_comparison_names'], 'Wrong independent active graph size')
        verify_aliases(target, findings['alias'], lambda name: target[name].get_tensor(name))
        rows = []
        for name in names:
            a, b = original[name].get_tensor(name), target[name].get_tensor(name)
            require(a.dtype in (torch.float32, torch.bfloat16) and b.dtype == torch.bfloat16, 'Wrong precision')
            rows.append({'name': name, **tensor_delta(a, b, native=a)})
    return {'exact_source_bf16_cast': True, 'comparison': summary(rows),
            'original_fp32_elements': sum(r['numel'] for r in rows if r['base_dtype'] == 'torch.float32'),
            'original_fp32_tensors': sum(r['base_dtype'] == 'torch.float32' for r in rows),
            'original_bf16_changed_elements': sum(r['changed_elements'] for r in rows if r['base_dtype'] == 'torch.bfloat16'),
            'reviewed_inactive_names': findings['base_only_inactive_mtp_names'],
            'verified_aliases': findings['alias']}


def make_plan(path):
    review = json.loads(REVIEW.read_text())
    base = json.loads(BASE_TRANSITION.read_text())['base']
    sources = set(SOURCES) | {str(REVIEW), str(BASE_TRANSITION)} | set(review['source_sha256'])
    write_new(path, {'kind': 'canonical_untrained_initialization_plan',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'base': base,
        'target': str(Path('data/models/qwen3.5-4b-common-bf16-v1').resolve()),
        'report': 'results/canonical-initialization-20260910.json',
        'graph_review': str(REVIEW), 'new_model_calls': 0, 'optimizer_steps': 0,
        'source_sha256': {name: file_sha256(Path(name)) for name in sorted(sources)},
        'versions': {name: version(name) for name in ('torch', 'transformers', 'safetensors')},
        'resources': {'gpus': 1, 'gpu_type': 'rtx_4090', 'cpus': 4, 'ram_gib': 64,
                      'time_limit_seconds': 1200, 'max_gpu_hours': 1/3},
        'limitations': ['Untrained common BF16 initialization, not a task result.',
            'CPU-only construction/audit; required minimum GPU allocation is fully charged.',
            'The reviewed 15 inactive MTP tensors are omitted; not applicable if MTP is enabled.',
            'Live inference/actual native training require subsequent execution checks.']})


def check(path):
    plan = json.loads(path.read_text())
    require(plan['kind'] == 'canonical_untrained_initialization_plan', 'Wrong initialization plan')
    require(plan['new_model_calls'] == plan['optimizer_steps'] == 0, 'Not an untrained initialization')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(Path(name)) == digest, f'Changed initialization input: {name}')
    require({n: version(n) for n in plan['versions']} == plan['versions'], 'Changed runtime')
    require(checkpoint_manifest(Path(plan['base']['path'])) == plan['base'], 'Changed original checkpoint')
    review = json.loads(Path(plan['graph_review']).read_text())
    require(file_sha256(Path(plan['base']['path']) / 'config.json') == review['base_config_sha256'], 'Wrong graph config')
    require(not Path(plan['target']).exists() and not Path(plan['report']).exists(), 'Preserve existing initialization')
    return plan, review


def run(args, plan, review):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Expected CPU Slurm wrapper')
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    started, job = time.monotonic(), os.environ['SLURM_JOB_ID']
    base, target = Path(plan['base']['path']), Path(plan['target'])
    write_new(Path(f'results/canonical-initialization-start-{job}.json'),
        {'kind': 'canonical_initialization_start', 'job_id': job, 'plan_sha256': file_sha256(args.plan),
         'base': plan['base'], 'optimizer_steps': 0, 'new_model_calls': 0})
    model = Qwen3_5ForConditionalGeneration.from_pretrained(base, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa')
    model.to(dtype=torch.bfloat16)
    require(all(p.dtype == torch.bfloat16 and p.device.type == 'cpu' for p in model.parameters()), 'Wrong model precision/device')
    model.save_pretrained(target, save_original_format=False, max_shard_size='12GB')
    del model
    gc.collect()
    # Preserve the exact original inference and tokenizer assets. Do not copy an
    # index naming the old shards or inactive MTP tensors into the new checkpoint.
    for source in base.iterdir():
        if source.is_file() and source.suffix in {'.json', '.jinja', '.txt'} and source.name != 'model.safetensors.index.json':
            shutil.copyfile(source, target / source.name)
    audit = audit_cast(base, target, review['findings'])
    require(audit['original_fp32_elements'] == 3840 and audit['original_fp32_tensors'] == 48 and
            audit['original_bf16_changed_elements'] == 0, 'Unexpected base precision accounting')
    contract = inference_contract(base, target)
    require(checkpoint_manifest(base) == plan['base'], 'Original checkpoint changed during conversion')
    write_new(Path(plan['report']), {'kind': 'canonical_untrained_initialization', 'status': 'PASS',
        'job_id': job, 'plan_sha256': file_sha256(args.plan), 'base': plan['base'],
        'exported': checkpoint_manifest(target), 'audit': audit, 'inference_contract': contract,
        'optimizer_steps': 0, 'new_model_calls': 0, 'seconds': time.monotonic() - started,
        'limitations': plan['limitations']})
    print(json.dumps({'status': 'PASS', 'report': plan['report'], 'report_sha256': file_sha256(Path(plan['report']))}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--write-plan', action='store_true')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.write_plan:
        make_plan(args.plan)
    else:
        plan, review = check(args.plan)
        if args.check_only:
            print(json.dumps({'status': 'PREFLIGHT_PASS', 'plan_sha256': file_sha256(args.plan)}))
        else:
            run(args, plan, review)
