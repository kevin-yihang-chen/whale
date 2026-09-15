"""Measure a shared initial visual harness on V before any E4 training.

Only the literal system prompt changes. E1 measures this initial condition;
neither VETO acceptance nor a weight update occurs during calibration.
"""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path

from . import native_visual_service as native
from . import visual_development_screen as development
from .evidence import fingerprint
from .visual_native_evaluation import pair_inputs
from .visual_task import file_sha256

ROOT = native.ROOT
HARNESS = ROOT / 'ours/visual_harnesses/evidence_first_harness.py'
SOURCES = ('ours/visual_harness_calibration.py', 'ours/run_visual_harness_calibration.sh',
           'ours/visual_harnesses/evidence_first_harness.py')


def configure(base, manifest_path, output):
    from omegaconf import OmegaConf
    from .visual_harness import load_visual_harness
    manifest, pairs, _ = pair_inputs(manifest_path)
    assert manifest['role'] == 'V' and manifest['partition'] == 'V' and len(pairs) == 256
    reference = ROOT / 'ours/visual_harnesses/canonical_answer_harness.py'
    case = deepcopy(base)
    case.update(kind='initial_visual_harness_calibration', role='V', mode='normalized_zoom',
        output=str(output), manifest=str(manifest_path), manifest_sha256=file_sha256(manifest_path),
        audit_data_sha256=manifest['audit_data_sha256'])
    cfg = native.configuration(Path(base['model']['path']), manifest_path, output)
    load_visual_harness(HARNESS, prompt_reference=reference)
    cfg.data.visual_harness_path = str(HARNESS)
    cfg.data.tool_config_path = str(ROOT / 'ours/normalized_visual_tools.yaml')
    cfg.actor_rollout_ref.rollout.multi_turn.tool_config_path = cfg.data.tool_config_path
    case['config'] = OmegaConf.to_container(cfg, resolve=True)
    case['harness_sha256'] = file_sha256(HARNESS)
    case['decode_sha256'] = fingerprint({'config': case['config'], 'mode': case['mode'],
                                       'model_assets': base['model']['assets']})
    case['bounds'] = {**base['bounds'], 'images': 512, 'maximum_generation_calls': 1536,
        'maximum_generated_assistant_tokens': 512 * 1024, 'time_limit_seconds': 3600,
        'cpus': 12, 'new_model_checkpoints': 0, 'api_calls': 0}
    case['source_sha256'].update({name: file_sha256(ROOT / name)
                                 for name in (*development.EXTRA, *SOURCES)})
    case['calibration'] = {
        'candidate': 'one manually specified evidence-first prompt with available crop tool',
        'data': 'all 256 frozen V pairs; no selection by observed pair failures',
        'selection_criterion': 'ordinary development accuracy and final-answer completion; paired score is diagnostic',
        'change': 'literal system prompt only; five callbacks, parser, tool, decode and reward retained',
        'reason': 'previous tool condition exhausted turns without an answer; establish a usable common h0',
        'resources': 'one model replica; previous small runs were dominated by startup and CPU callbacks, so duplicating replicas is not expected to halve total time',
    }
    case['limitations'] = ['Development calibration on one fixed model, not sealed test or official PlotQA performance.',
        'No VETO E2/E3 selection, training, proposer calls, or method-efficacy claim.',
        'The first 64 V pairs were already inspected in the earlier diagnostic; this is not an independent re-evaluation pool.',
        'Tools remain available; a prompt instruction recommends at most one crop but does not change the native turn cap.']
    return case


def prepare(path, output, base_path):
    base = native.check(base_path)
    assert not path.exists() and not output.exists()
    manifest = ROOT / 'data/plotqa-evidence-pairs-20260910-v1/rendered/V/manifest.json'
    case = configure(base, manifest, output)
    case['base_plan'] = str(base_path)
    case['base_plan_sha256'] = file_sha256(base_path)
    native.write_new(path, case)


def run(path):
    case = json.loads(path.read_text())
    base_path = Path(case['base_plan'])
    assert file_sha256(base_path) == case['base_plan_sha256']
    base = native.check(base_path)
    expected = configure(base, Path(case['manifest']), Path(case['output']))
    expected.update(base_plan=str(base_path), base_plan_sha256=file_sha256(base_path))
    assert case == expected
    asyncio.run(development.evaluate(case, path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base-plan', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.plan.resolve(), args.output.resolve(), args.base_plan.resolve())
    else:
        run(args.plan.resolve())
