"""Bounded F0 seed propagation check, independent of any task evaluation.

Two GPU processes cover three seed trials in two waves. Each engine executes
eight identical toy prompts without a request seed, as native training does.
The checkpoint is the audited engineering theta2, not the F0 initialization.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .alternation_training import write_new
from .audit_native_training_batch import require
from .experiment_randomization import HASH_PROBE, TRIAL_SEEDS, TrialRandomization
from .local_completion import checkpoint_manifest
from .probe_updated_vllm import managed_inference_engine
from .visual_task import file_sha256

SOURCE_FILES = (
    'ours/probe_trial_randomization.py', 'ours/run_trial_randomization_probe.sh',
    'ours/vllm_randomization_worker.py', 'ours/experiment_randomization.py',
    'ours/controlled_conditions.py', 'ours/controlled_training_bootstrap.py',
    'ours/tests/test_experiment_randomization.py', 'ours/tests/test_controlled_conditions.py',
    'results/controlled-native-fixtures-20260910.log',
    'upstream/WHALE/domains/chess_puzzles/verl/workers/rollout/vllm_rollout/vllm_async_server.py',
    'upstream/WHALE/domains/chess_puzzles/verl/trainer/main_ppo.py',
    'upstream/WHALE/domains/chess_puzzles/verl/experimental/agent_loop/agent_loop.py',
)


def make_plan(path):
    old_path = Path('results/alternation-vllm-handoff-plan-20260909.json')
    old = json.loads(old_path.read_text())
    prior_path = Path('results/alternation-vllm-handoff-222779.json')
    prior = json.loads(prior_path.read_text())
    require(prior['status'] == 'PASS' and prior['plan_sha256'] == file_sha256(old_path), 'Missing prior handoff')
    transition = json.loads(Path(old['transition']).read_text())
    engine = dict(old['engine_options'])
    engine.pop('seed')
    engine['worker_extension_cls'] = 'ours.vllm_randomization_worker.TrialRandomizationWorkerExtension'
    sources = set(SOURCE_FILES) | set(old['source_sha256']) | {str(old_path), str(prior_path)}
    write_new(path, {'kind': 'trial_randomization_probe_plan',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': old['model'], 'checkpoint_manifest': transition['exported'],
        'seeds': list(TRIAL_SEEDS), 'gpu_assignments': [[42, 44], [43]],
        'engine_options': engine, 'sampling': {'temperature': 1., 'top_p': 1., 'top_k': 20,
                                              'seed': None, 'max_tokens': 4},
        'prompt_token_ids': prior['contract']['prompt_token_ids'],
        'coordinates': prior['changed_coordinates'],
        'requests_per_seed': 8, 'max_new_model_calls': 24, 'max_output_tokens': 96,
        'source_sha256': {name: file_sha256(Path(name)) for name in sorted(sources)},
        'versions': {name: version(name) for name in ('torch', 'vllm', 'transformers', 'safetensors')},
        'resources': {'gpus': 2, 'gpu_type': 'H800', 'cpus': 16, 'ram_gib': 128,
                      'time_limit_seconds': 900, 'max_gpu_hours': .5},
        'resource_decision': {'forecast_hkt': '2026-09-10T00:20:33+08:00',
            'one_gpu': 'Immediate forecast, three sequential engines; estimated 240s / 0.067GPUh.',
            'two_gpus': 'Immediate forecast, two waves; estimated 160s / 0.089GPUh. Selected for elapsed time.',
            'three_gpus': 'Rejected by partition QOSMaxGRESPerUser in test-only forecast.',
            'merge_risk': 'Distinct per-seed receipts; require all three, no score aggregation.'},
        'limitations': ['Engineering seed check using theta2, not independent F0 optimization trials.',
            'Native CLI and data-sampler propagation are CPU fixtures; live engines are standalone vLLM.',
            'No full Ray training, MH provider sampling, task scoring or bitwise determinism claim.',
            'Eight actual embedding coordinates per engine, not a full served weight fingerprint.',
            'No API call or train/MH/test input.']})


def check_plan(path, *, weights):
    plan = json.loads(path.read_text())
    require(plan['kind'] == 'trial_randomization_probe_plan', 'Wrong seed plan')
    require(plan['seeds'] == [42, 43, 44] and plan['gpu_assignments'] == [[42, 44], [43]], 'Wrong seed coverage')
    require(plan['requests_per_seed'] == 8 and plan['max_new_model_calls'] == 24 and
            plan['max_output_tokens'] == 96, 'Wrong request budget')
    require(plan['sampling'] == {'temperature': 1., 'top_p': 1., 'top_k': 20, 'seed': None, 'max_tokens': 4},
            'Wrong sampling scope')
    require('seed' not in plan['engine_options'], 'Engine seed must come from each trial')
    require({n: version(n) for n in plan['versions']} == plan['versions'], 'Changed runtime')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(Path(name)) == digest, f'Changed source: {name}')
    if weights:
        require(checkpoint_manifest(Path(plan['model'])) == plan['checkpoint_manifest'], 'Changed model')
    return plan


def run_seed(args, plan):
    from .experiment_randomization import initialize_process_randomness
    process = initialize_process_randomness()
    require(process['trial_seed'] == args.seed and args.seed in plan['seeds'], 'Wrong trial environment')
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Expected GPU allocation')
    from vllm import SamplingParams
    root = args.output
    root.mkdir(exist_ok=False)
    write_new(root / 'start.json', {'kind': 'trial_seed_start', 'seed': args.seed,
        'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(args.plan),
        'process_randomness': process, 'max_calls': 8, 'max_tokens': 32})
    started = time.monotonic()
    engine_plan = {'model': plan['model'], 'engine_options': dict(plan['engine_options'], seed=args.seed)}
    with managed_inference_engine(engine_plan) as llm:
        before, = llm.collective_rpc('trial_randomization_state', timeout=60.)
        repeated, = llm.collective_rpc('trial_randomization_state', timeout=60.)
        require(before == repeated, 'Read-only worker probe advanced RNG state')
        require(all(before[k] == args.seed for k in
                ('model_seed', 'torch_cpu_initial_seed', 'torch_cuda_initial_seed')), 'Live seed differs')
        require(before['python_hash_probe'] == hash(HASH_PROBE), 'Worker hash seed differs')
        samples, = llm.collective_rpc('sample_updated_embeddings', timeout=60., kwargs={
            'coordinates': [{k: c[k] for k in ('token_id', 'column')} for c in plan['coordinates']]})
        require(samples['values'] == [c['updated'] for c in plan['coordinates']], 'Wrong live parameters')
        write_new(root / 'request.json', {'kind': 'trial_seed_request_start', 'seed': args.seed,
            'plan_sha256': file_sha256(args.plan), 'actual_worker_state': before,
            'actual_worker_samples': samples, 'logical_calls_attempted': 8,
            'sampling': plan['sampling'], 'prompt_token_ids': plan['prompt_token_ids'],
            'returned_tokens': None})
        outputs = llm.generate([{'prompt_token_ids': plan['prompt_token_ids']} for _ in range(8)],
                               SamplingParams(**plan['sampling']), use_tqdm=False)
        after, = llm.collective_rpc('trial_randomization_state', timeout=60.)
        require(len(outputs) == 8 and all(len(o.outputs) == 1 for o in outputs), 'Wrong output coverage')
        replies = [{'request_id': o.request_id, 'token_ids': list(o.outputs[0].token_ids),
                    'text': o.outputs[0].text, 'finish_reason': o.outputs[0].finish_reason} for o in outputs]
        require(all(0 < len(r['token_ids']) <= 4 for r in replies), 'Empty or over-budget generation')
    write_new(root / 'result.json', {'kind': 'trial_seed_result', 'status': 'PASS', 'seed': args.seed,
        'plan_sha256': file_sha256(args.plan), 'job_id': os.environ['SLURM_JOB_ID'],
        'before': before, 'after': after, 'actual_worker_samples': samples,
        'new_model_calls': 8, 'generated_tokens': sum(len(r['token_ids']) for r in replies),
        'replies': replies, 'seconds': time.monotonic() - started,
        'interpretation': 'Actual seed/weight verification; output diversity is diagnostic, not a pass criterion.'})


def run_all(args, plan):
    job = os.environ['SLURM_JOB_ID']
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    require(len(devices) == 2, 'Expected exactly two visible GPUs')
    root = Path(f'results/trial-randomization-{job}')
    root.mkdir(exist_ok=False)
    write_new(root / 'start.json', {'kind': 'trial_seed_allocation_start', 'job_id': job,
        'plan_sha256': file_sha256(args.plan), 'checked_model_manifest': plan['checkpoint_manifest'],
        'assignments': plan['gpu_assignments'], 'time_utc': datetime.now(timezone.utc).isoformat()})
    def wave(index):
        for seed in plan['gpu_assignments'][index]:
            command = [sys.executable, '-m', 'ours.probe_trial_randomization', '--plan', str(args.plan),
                       '--seed', str(seed), '--output', str(root / f'seed-{seed}')]
            env = dict(os.environ, **TrialRandomization(seed).environment(), CUDA_VISIBLE_DEVICES=devices[index])
            with (root / f'seed-{seed}.log').open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(wave, index) for index in range(2)]
        for future in futures:
            future.result()
    rows = [json.loads((root / f'seed-{seed}/result.json').read_text()) for seed in plan['seeds']]
    require(all(r['status'] == 'PASS' and r['plan_sha256'] == file_sha256(args.plan) for r in rows), 'Mixed trial results')
    for key in ('model_seed', 'torch_cpu_rng_sha256', 'torch_cuda_rng_sha256', 'python_hash_probe'):
        require(len({row['before'][key] for row in rows}) == 3, f'Seed trials did not differ: {key}')
    write_new(root / 'measurement.json', {'kind': 'trial_randomization_probe', 'status': 'PASS',
        'job_id': job, 'plan_sha256': file_sha256(args.plan), 'seeds': plan['seeds'],
        'new_model_calls': sum(r['new_model_calls'] for r in rows),
        'generated_tokens': sum(r['generated_tokens'] for r in rows),
        'result_sha256': {str(seed): file_sha256(root / f'seed-{seed}/result.json') for seed in plan['seeds']},
        'limitations': plan['limitations']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--write-plan', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.write_plan:
        make_plan(args.plan)
    else:
        # The allocation controller verifies all checkpoint bytes once before
        # starting children. Child processes still verify the entire source set.
        plan = check_plan(args.plan, weights=args.seed is None)
        if args.check_only:
            print(json.dumps({'status': 'PREFLIGHT_PASS', 'seeds': plan['seeds'],
                              'plan_sha256': file_sha256(args.plan)}))
        elif args.seed is not None:
            run_seed(args, plan)
        else:
            run_all(args, plan)
