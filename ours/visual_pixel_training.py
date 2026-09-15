"""Restart the first visual E4 batch after repairing Qwen3.5 pixel forwarding.

Uses the original initial model, h0, W subset and native RSFT settings. Earlier
text-loss checkpoints stay archived; they are not this restart's parent.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from . import native_visual_training as base
from .visual_task import file_sha256

ROOT = base.ROOT
ORIGINAL_CONFIGURATION = base.configuration
AMENDMENT = ROOT / 'results/visual-pixel-forward-correction-20260910-v2.json'
SOURCES = ('ours/visual_pixel_training.py', 'ours/visual_pixel_bootstrap.py',
    'ours/qwen35_visual_fused.py', 'ours/run_visual_pixel_training.sh',
    'upstream/WHALE/domains/chess_puzzles/verl/models/transformers/dense_common.py',
    'upstream/WHALE/domains/chess_puzzles/verl/models/transformers/monkey_patch.py',
    'upstream/WHALE/domains/chess_puzzles/verl/utils/experimental/torch_functional.py',
    'data/training-runtime-v1/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py')


def configuration(output, path):
    cfg = ORIGINAL_CONFIGURATION(output, path)
    cfg.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook = 'ours.visual_pixel_bootstrap.prepare_worker'
    cfg.ray_kwargs.ray_init.runtime_env.env_vars['VETO_VISUAL_FORWARD_AUDIT'] = str(output / 'checkpoints/audit')
    return cfg


def prepare(path, output):
    from omegaconf import OmegaConf
    assert not path.exists() and not output.exists()
    assert AMENDMENT.exists()
    assert shutil.disk_usage(ROOT).free >= 80 * 1024**3
    base.native.check(ROOT / 'results/native-visual-service-plan-20260910-v1.json')
    output.mkdir()
    cfg = configuration(output, path)
    preflight = base.inspect_dataset(cfg)
    prior_path = ROOT / 'results/native-visual-rsft-plan-20260910-v4.json'
    prior = json.loads(prior_path.read_text())
    assert preflight == prior['cpu_preflight']
    plan = deepcopy(prior)
    plan.update(output=str(output), config=OmegaConf.to_container(cfg, resolve=True), cpu_preflight=preflight,
        execution_variant='qwen35_visual_pixel_restart', parent_checkpoint=None)
    plan['inputs_sha256'].update({str(AMENDMENT): file_sha256(AMENDMENT), str(prior_path): file_sha256(prior_path)})
    plan['source_sha256'].update({name: file_sha256(ROOT / name) for name in SOURCES})
    plan['storage'] = {'additional_working_space_gib': 40, 'minimum_free_margin_gib': 40}
    plan['limitations'] += [
        'Restart from the original model after confirmed missing pixels in the earlier fused training forward.',
        'Earlier theta1 inference and search remain real measurements on a text-loss-updated model, not a valid multimodal RSFT lineage.',
        'Actual visual backbone pixels and visual feature gradients are recorded; full parameter comparison is still required.',
        'No new proposer requests, no formal expansion, no checkpoint deletion.']
    base.native.write_new(path, plan)
    base.native.write_new(output / 'cpu-preflight.json', preflight)


def run(path):
    from .visual_pixel_bootstrap import prepare_worker
    plan = json.loads(path.read_text())
    assert plan['execution_variant'] == 'qwen35_visual_pixel_restart' and plan['parent_checkpoint'] is None
    assert shutil.disk_usage(ROOT).free >= 80 * 1024**3
    os.environ['VETO_VISUAL_FORWARD_AUDIT'] = str(Path(plan['output']) / 'checkpoints/audit')
    prepare_worker()
    base.configuration = configuration
    base.run(path)
    directory = Path(plan['output']) / 'checkpoints/audit'
    records = [json.loads(line) for p in directory.glob('visual-backbone-*.jsonl') for line in p.read_text().splitlines()]
    inputs = {r['event'] for r in records if r['kind'] == 'visual_backbone_input'}
    gradients = [r for r in records if r['kind'] == 'visual_feature_gradient']
    assert inputs and gradients and all(r['event'] in inputs and r['finite'] for r in gradients)
    assert any(r['nonzero_elements'] > 0 for r in gradients)
    base.native.write_new(Path(plan['output']) / 'visual-forward-result.json', {
        'status': 'OBSERVED_IMAGE_BACKBONE_AND_LOSS_GRADIENT', 'input_forwards': len(inputs),
        'gradient_events': len(gradients), 'nonzero_gradient_events': sum(r['nonzero_elements'] > 0 for r in gradients),
        'plan_sha256': file_sha256(path), 'job_id': os.environ['SLURM_JOB_ID'],
        'artifacts_sha256': {str(p): file_sha256(p) for p in directory.glob('visual-backbone-*.jsonl')},
        'parameter_change_verified': False, 'scientific_method_verified': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.plan.resolve(), args.output.resolve())
    else:
        run(args.plan.resolve())
