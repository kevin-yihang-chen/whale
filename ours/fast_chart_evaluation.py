"""Real native VLM evaluation under the compact shared E1 answer protocol."""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import native_visual_service as native
from .chart_answer_protocol import PROTOCOL
from .evidence import EvaluationIdentity, fingerprint
from .fast_chart_protocol import OUTPUT, storage_check
from .local_completion import checkpoint_manifest
from .visual_native_evaluation import pair_inputs, evaluate_pairs, verifier_identity
from .visual_task import file_sha256

ROOT = native.ROOT
EXTRA = ('ours/fast_chart_evaluation.py', 'ours/fast_chart_protocol.py',
    'ours/fast_chart_protocol_20260915.md', 'ours/chart_answer_protocol.py',
    'ours/run_fast_chart_evaluation.sh', 'ours/visual_dataset_materialization.py',
    'ours/isolated_visual_harness.py', 'ours/visual_callback_worker.py', 'ours/callback_confinement.py')
PROMPTS = {
    'direct': 'Read the chart and answer the question from the supplied image. Return only the requested answer on the last line.',
    'structured': 'Read the chart carefully: identify the requested categories and series, check the axes and units, then determine the answer. Put only the requested answer after "Final answer:" on the last line.',
    'brief_reasoning': 'Use the chart image to solve the question. Briefly explain the relevant visual evidence and calculation. End with "Final answer:" followed by only the requested answer.'}


def read(path):
    return json.loads(Path(path).read_text())


def decode_identity(config, assets):
    cfg = deepcopy(config)
    cfg['data'].pop('visual_harness_path')
    cfg['data'].pop('cache_dir')
    cfg.pop('trainer')
    cfg['actor_rollout_ref']['rollout'].pop('trace')
    return fingerprint({'config': cfg, 'model_assets': assets})


def prepare(path, *, manifest_path, harness, model, output, seed=42, batch_size=8, phase='calibration'):
    from omegaconf import OmegaConf
    path, manifest_path, harness, model, output = map(lambda p: Path(p).resolve(), (path, manifest_path, harness, model, output))
    if path.exists() or output.exists() or not output.is_relative_to(OUTPUT):
        raise ValueError('Require fresh output under the compact experiment root')
    manifest, pairs, _ = pair_inputs(manifest_path)
    if (manifest.get('answer_protocol') != PROTOCOL or
            manifest.get('partition') not in {'C', 'C-v3', 'V', 'H-pair'} or
            (manifest.get('partition') == 'H-pair') != (manifest.get('role') == 'H')):
        raise ValueError('This preparer only opens registered compact chart pairs')
    if seed not in (42,43,44) or batch_size not in (1,2,4,8):
        raise ValueError('Unregistered seed or evaluation concurrency')
    cfg = native.configuration(model, manifest_path, output)
    cfg.data.visual_answer_protocol = PROTOCOL
    cfg.data.visual_harness_path = str(harness)
    cfg.data.tool_config_path = None
    rollout = cfg.actor_rollout_ref.rollout
    rollout.multi_turn.tool_config_path = None
    rollout.agent.num_workers = min(2, batch_size)
    rollout.max_num_seqs = batch_size
    rollout.enable_prefix_caching = False
    rollout.enable_chunked_prefill = False
    rollout.engine_kwargs.vllm.seed = seed
    from .visual_harness import configured_visual_harness
    configured_visual_harness(cfg.data).unchanged()
    model_manifest = checkpoint_manifest(model)
    config = OmegaConf.to_container(cfg, resolve=True)
    sources = set(native.SOURCES) | set(EXTRA)
    sources.add(str(harness.relative_to(ROOT)))
    plan = {'kind': 'compact_chart_pair_evaluation', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'output': str(output), 'role': manifest['role'], 'partition': manifest['partition'], 'phase': phase,
        'manifest': str(manifest_path), 'manifest_sha256': file_sha256(manifest_path),
        'audit_data_sha256': manifest['audit_data_sha256'], 'verifier_sha256': verifier_identity(PROTOCOL),
        'model': model_manifest, 'worker_coordinates': native.embedding_coordinates(model),
        'harness_sha256': file_sha256(harness), 'config': config, 'seed': seed, 'batch_size': batch_size,
        'decode_sha256': decode_identity(config, model_manifest['assets']),
        'source_sha256': {str(p): file_sha256(ROOT / p) for p in sorted(sources)},
        'bounds': {'images': 2*len(pairs), 'maximum_generation_calls': 6*len(pairs),
            'maximum_generated_assistant_tokens': 2*len(pairs)*1024, 'gpus': 1, 'cpus': 12,
            'time_limit_seconds': 5400, 'new_model_checkpoints': 0, 'api_calls': 0},
        'limitations': ['Proposal/optimization/development result, not independent test performance.',
            'Greedy batched inference is not claimed to be bitwise repeatable.',
            'One predeclared complete pass; no selection of a better repeated score.']}
    storage_check(12)
    native.write_new(path, plan)
    return plan


def check(path):
    plan = read(path)
    if plan['kind'] != 'compact_chart_pair_evaluation':
        raise ValueError('Wrong compact evaluation plan')
    if Path(plan['output']).exists() or not Path(plan['output']).is_relative_to(OUTPUT):
        raise ValueError('Existing or unaccounted output')
    for name, digest in plan['source_sha256'].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    if file_sha256(Path(plan['manifest'])) != plan['manifest_sha256'] or checkpoint_manifest(Path(plan['model']['path'])) != plan['model']:
        raise ValueError('Data or model identity changed')
    if decode_identity(plan['config'], plan['model']['assets']) != plan['decode_sha256'] or verifier_identity(PROTOCOL) != plan['verifier_sha256']:
        raise ValueError('Decoder or shared scoring changed')
    storage_check(12)
    return plan


async def evaluate(plan, path):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import ray
    import torch
    from omegaconf import OmegaConf
    from .native_visual_agent_observation import ObservedVisualAgentManager
    from .visual_native_evaluation import write_pair_parquet
    if torch.cuda.device_count() != 1 or 'H800' not in torch.cuda.get_device_name(0):
        raise ValueError('Expected the frozen single-H800 allocation')
    output = Path(plan['output'])
    output.mkdir()
    for name in ('requests', 'workers'):
        (output / name).mkdir()
    for name in plan['source_sha256']:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (output / 'plan.json').write_bytes(path.read_bytes())
    os.environ.update(VETO_NATIVE_EVALUATION_PLAN=str(path), VETO_NATIVE_EVALUATION_OUTPUT=str(output))
    native.write_new(output / 'start.json', {'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(path)})
    started = time.monotonic()
    try:
        parquet = output / 'pairs.parquet'
        native.write_new(output / 'materialization.json', write_pair_parquet(plan['manifest'], parquet))
        dataset = native.dataset_for(plan, parquet)
        env = {k:v for k,v in os.environ.items() if k.startswith(('VETO_', 'HF_', 'TRANSFORMERS_', 'VERL_', 'VLLM_')) or
            k in ('PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','NO_PROXY','no_proxy')}
        ray.init(num_cpus=12, num_gpus=1, include_dashboard=False, object_store_memory=2*1024**3,
            runtime_env={'env_vars': env, 'worker_process_setup_hook': 'ours.training_bootstrap.prepare_worker'})
        manager = await ObservedVisualAgentManager.create(OmegaConf.create(plan['config']))
        loaded = time.monotonic()
        identity = EvaluationIdentity(plan['model']['weights_sha256'], plan['manifest_sha256'],
            plan['audit_data_sha256'], plan['decode_sha256'], plan['verifier_sha256'], plan['phase'])
        receipt, result = await evaluate_pairs(manager, dataset, manifest_path=plan['manifest'], identity=identity,
            output=output / 'evaluation', batch_size=plan['batch_size'])
        observed = native.audit_requests(plan, output, result)
        report = {'status': 'COMPLETE_COMPACT_CHART_PAIR_EVALUATION', 'job_id': os.environ['SLURM_JOB_ID'],
            'plan_sha256': file_sha256(path), 'role': plan['role'], 'partition': plan['partition'],
            'audit': asdict(receipt), 'paired_accuracy': receipt.paired_accuracy,
            'marginal_accuracy': receipt.marginal_accuracy, 'observed': observed,
            'timing_seconds': {'loading': loaded-started, 'evaluation': time.monotonic()-loaded,
                               'total': time.monotonic()-started}, 'scientific_method_verified': False}
        native.write_new(output / 'result.json', report)
        print(json.dumps({k: report[k] for k in ('status','paired_accuracy','marginal_accuracy','timing_seconds')}), flush=True)
    except BaseException as exc:
        native.write_new(output / 'failure.json', {'status': 'INCOMPLETE', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        ray.shutdown()


def prepare_calibration(path, output, manifest):
    output = Path(output).resolve()
    output.mkdir()
    (output / 'harnesses').mkdir()
    reference = (ROOT / 'ours/visual_harnesses/canonical_answer_harness.py').read_text()
    cases = {}
    for name, prompt in PROMPTS.items():
        lines = reference.splitlines()
        lines = [f'SYSTEM_PROMPT = {prompt!r}' if line.startswith('SYSTEM_PROMPT = ') else line for line in lines]
        harness = output / 'harnesses' / f'{name}.py'
        with harness.open('x') as stream:
            stream.write('\n'.join(lines)+'\n')
        case_path = output / f'{name}-plan.json'
        prepare(case_path, manifest_path=manifest, harness=harness,
            model=ROOT / 'data/models/qwen3.5-4b-common-bf16-v1', output=output / name)
        cases[name] = {'plan': str(case_path), 'sha256': file_sha256(case_path)}
    suite = {'kind': 'compact_initial_harness_calibration', 'output': str(output), 'cases': cases,
        'selection': 'ordinary V accuracy, then fewer total generated tokens, then direct/structured/brief_reasoning order',
        'bounds': {'gpus': 1, 'cpus': 12, 'time_limit_seconds': 5400, 'api_calls': 0, 'new_model_checkpoints': 0}}
    native.write_new(Path(path), suite)
    return suite


def run_suite(path):
    suite = read(path)
    if suite['kind'] != 'compact_initial_harness_calibration' or tuple(suite['cases']) != tuple(PROMPTS):
        raise ValueError('Changed calibration candidates')
    results = {}
    for name, case in suite['cases'].items():
        case_path = Path(case['plan'])
        if file_sha256(case_path) != case['sha256']:
            raise ValueError('Changed calibration plan')
        with (Path(suite['output']) / f'{name}.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'ours.fast_chart_evaluation', 'run', '--plan', str(case_path)],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        result_path = Path(read(case_path)['output']) / 'result.json'
        result = read(result_path)
        if result['status'] != 'COMPLETE_COMPACT_CHART_PAIR_EVALUATION' or result['plan_sha256'] != case['sha256']:
            raise ValueError('Incomplete calibration result')
        results[name] = {'accuracy': result['marginal_accuracy'], 'paired_accuracy': result['paired_accuracy'],
            'generated_tokens': result['observed']['generated_tokens'], 'result': str(result_path),
            'result_sha256': file_sha256(result_path)}
    winner = min(PROMPTS, key=lambda name: (-results[name]['accuracy'], results[name]['generated_tokens'], list(PROMPTS).index(name)))
    native.write_new(Path(suite['output']) / 'result.json', {'status': 'COMPLETE_SHARED_H0_CALIBRATION',
        'plan_sha256': file_sha256(path), 'job_id': os.environ['SLURM_JOB_ID'], 'selected': winner,
        'harness': read(suite['cases'][winner]['plan'])['config']['data']['visual_harness_path'],
        'results': results, 'scientific_method_verified': False})


def run_plan(path):
    """Dispatch the two plan kinds admitted by the shared evaluation wrapper."""
    kind = read(path).get('kind')
    if kind == 'compact_initial_harness_calibration':
        return run_suite(path)
    if kind == 'compact_chart_pair_evaluation':
        return asyncio.run(evaluate(check(path), path.resolve()))
    raise ValueError('Changed or unsupported evaluation plan kind')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare-calibration','prepare','check','run','run-suite','run-plan'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--harness', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--weight-phase', default='calibration')
    args = parser.parse_args()
    if args.phase == 'prepare-calibration':
        prepare_calibration(args.plan.resolve(), args.output, args.manifest)
    elif args.phase == 'prepare':
        prepare(args.plan, manifest_path=args.manifest, harness=args.harness, model=args.model,
                output=args.output, phase=args.weight_phase)
    elif args.phase == 'check':
        print(json.dumps({'status': 'PASS_PREFLIGHT', 'bounds': check(args.plan)['bounds']}))
    elif args.phase == 'run':
        asyncio.run(evaluate(check(args.plan), args.plan.resolve()))
    elif args.phase == 'run-suite':
        run_suite(args.plan.resolve())
    else:
        run_plan(args.plan.resolve())
