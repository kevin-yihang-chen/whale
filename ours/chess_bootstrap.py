"""Recorded on-policy success sampling for E4, using the unchanged Chess runner.

This measures whether native RSFT has a nonempty accepted set. It performs no
weight update, harness search, answer repair or external API request.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

from .local_completion import checkpoint_manifest
from .native_search import target_service, tree_hashes, write_json
from .preflight import PIN
from .visual_task import file_sha256
from .vllm_runtime import RecordedVLLMClient, VLLMConfiguration


def read_rows(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path, pre_buffer=False)
    return [row for i in range(parquet.num_row_groups)
            for row in parquet.read_row_group(i, use_threads=False).to_pylist()]


def sampling_keys(ids, seeds):
    keys = [(puzzle, seed) for seed in seeds for puzzle in ids]
    if len(keys) != len(set(keys)) or not ids or not seeds:
        raise ValueError("Empty or duplicate puzzle/seed assignment")
    return keys


def check_plan(path):
    plan = json.loads(path.read_text())
    if plan['kind'] != 'chess_bootstrap_sampling_plan' or plan['veto_mode'] != 'off':
        raise ValueError('Expected an explicit E4 bootstrap plan')
    upstream = 'upstream/WHALE'
    if subprocess.check_output(['git', '-C', upstream, 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise ValueError('Wrong upstream revision')
    if subprocess.check_output(['git', '-C', upstream, 'status', '--porcelain'], text=True).strip():
        raise ValueError('Modified upstream')
    for name, digest in plan['source_sha256'].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f'Changed source: {name}')
    if {name: version(name) for name in plan['versions']} != plan['versions']:
        raise ValueError('Changed runtime versions')
    manifest = json.loads(Path(plan['pilot_manifest']).read_text())
    source = manifest['splits']['train']
    if file_sha256(Path(source['path'])) != source['sha256']:
        raise ValueError('Changed pilot training data')
    rows = sorted(read_rows(source['path']), key=lambda r: r['extra_info']['puzzle_id'])[:32]
    if [r['extra_info']['puzzle_id'] for r in rows] != plan['puzzle_ids']:
        raise ValueError('Selection must be the first 32 training IDs')
    forbidden = set(manifest['splits']['test']['puzzle_ids'] + manifest['splits']['mh_val']['puzzle_ids'])
    if forbidden.intersection(plan['puzzle_ids']):
        raise ValueError('Training selection crosses a data-role boundary')
    for chunk, expected_rows in zip(plan['chunks'], (rows[:8], rows[8:]), strict=True):
        if read_rows(chunk['dataset']) != expected_rows:
            raise ValueError('Changed chunk rows')
        if chunk['puzzle_ids'] != [r['extra_info']['puzzle_id'] for r in expected_rows]:
            raise ValueError('Changed chunk identity')
    if plan['replica_seeds'] != [[42, 43, 44, 45], [46, 47, 48, 49]]:
        raise ValueError('Changed seed partition')
    sampling_keys(plan['puzzle_ids'], sum(plan['replica_seeds'], []))
    for replica in plan['replica_seeds']:
        VLLMConfiguration(**dict(plan['target_config'], base_url='http://127.0.0.1:1', seed=replica[0]))
    from autoharness_chess_puzzle.harness import load_harness
    load_harness(plan['harness'])
    from vllm.config import SchedulerConfig
    opts = plan['server_arguments']
    def val(k):
        return int(opts[opts.index(k) + 1])
    if '--enable-chunked-prefill' not in opts:
        raise ValueError('Required supported prefill is absent')
    SchedulerConfig(max_model_len=val('--max-model-len'), is_encoder_decoder=False,
                    max_num_batched_tokens=val('--max-num-batched-tokens'),
                    max_num_seqs=val('--max-num-seqs'), enable_chunked_prefill=True,
                    is_multimodal_model=True)
    return plan


def run_worker(args, plan):
    import torch
    from autoharness_chess_puzzle import runner
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Each replica must see exactly one allocated GPU')
    args.output.mkdir(parents=True, exist_ok=False)
    identity = checkpoint_manifest(Path(plan['target_config']['model']))
    if identity['weights_sha256'] != plan['target_config']['expected_weights_sha256']:
        raise ValueError('Checkpoint identity mismatch')
    (args.output / 'plan.json').write_bytes(args.plan.read_bytes())
    summaries, clients = [], []
    chunk = plan['chunks'][args.chunk]

    class Client(RecordedVLLMClient):
        def __init__(self, config):
            super().__init__(config)
            clients.append(self)

    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Worker received signal {signum}')

    signal.signal(signal.SIGTERM, terminate)
    started = time.perf_counter()
    try:
        with target_service(plan, args.output) as endpoint, \
                patch.object(runner, 'LLMClient', Client), patch.object(runner, 'LLMConfig', VLLMConfiguration):
            for seed in plan['replica_seeds'][args.worker]:
                output = args.output / f'seed-{seed}'
                output.mkdir()
                config = dict(plan['target_config'], seed=seed, base_url=endpoint,
                              trace_path=str(output / 'generations.jsonl'))
                summary = runner.evaluate_harness(
                    harness_path=plan['harness'], dataset_path=chunk['dataset'],
                    llm_config=config, output_dir=output / 'evaluation',
                    limit=len(chunk['puzzle_ids']), seed=seed,
                    assistant_token_budget=plan['assistant_token_budget'],
                    policy_max_tokens=plan['policy_max_tokens'])
                clients[-1].close()
                if summary['num_examples'] != len(chunk['puzzle_ids']):
                    raise ValueError('Incomplete native evaluation')
                summaries.append({'seed': seed, **summary})
                print(json.dumps({'replica': args.worker, 'seed': seed,
                                  'solved': summary['solved_examples'],
                                  'trajectories': summary['num_examples']}), flush=True)
    finally:
        for client in clients:
            client.close()
    write_json(args.output / 'result.json', {
        'kind': 'chess_bootstrap_replica', 'status': 'COMPLETED', 'role': 'training_bootstrap',
        'chunk': args.chunk, 'replica': args.worker, 'summaries': summaries,
        'plan_sha256': file_sha256(args.plan), 'model_manifest': identity,
        'gpu': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'seconds': time.perf_counter() - started, 'training_steps': 0, 'external_api_calls': 0,
        'artifact_sha256': tree_hashes(args.output)})


def run_parent(args, plan):
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Allocate exactly two GPUs for the frozen seed partition')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'plan.json').write_bytes(args.plan.read_bytes())
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError('Ambiguous GPU allocation')
    children, logs = [], []
    started = time.perf_counter()
    try:
        for replica, device in enumerate(devices):
            log = (args.output / f'replica-{replica}.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
            children.append(subprocess.Popen(
                [sys.executable, '-m', 'ours.chess_bootstrap', '--plan', str(args.plan),
                 '--chunk', str(args.chunk), '--worker', str(replica), '--output',
                 str(args.output / f'replica-{replica}')], env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('A replica failed; preserve its evidence and stop its peer')
            time.sleep(2)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError('Replica failed')
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in logs:
            log.close()
    replicas = [json.loads((args.output / f'replica-{i}/result.json').read_text()) for i in range(2)]
    keys = []
    for i, result in enumerate(replicas):
        if result['status'] != 'COMPLETED' or result['replica'] != i or result['chunk'] != args.chunk:
            raise ValueError('Wrong replica result identity')
        if result['plan_sha256'] != file_sha256(args.plan):
            raise ValueError('Replica used a different plan')
        for summary in result['summaries']:
            keys.extend(sampling_keys(plan['chunks'][args.chunk]['puzzle_ids'], [summary['seed']]))
    expected = sampling_keys(plan['chunks'][args.chunk]['puzzle_ids'], list(range(42, 50)))
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError('Missing or repeated replica coverage')
    write_json(args.output / 'result.json', {
        'kind': 'chess_bootstrap_sampling', 'status': 'COMPLETED_PENDING_INDEPENDENT_AUDIT',
        'chunk': args.chunk, 'trajectories': len(keys),
        'native_solved': sum(s['solved_examples'] for r in replicas for s in r['summaries']),
        'plan_sha256': file_sha256(args.plan), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'seconds': time.perf_counter() - started, 'training_steps': 0, 'external_api_calls': 0,
        'artifact_sha256': tree_hashes(args.output)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--chunk', type=int, choices=(0, 1), default=0)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = check_plan(args.plan)
    if args.check_only:
        print(json.dumps({'status': 'PRE_SUBMISSION_PASS', 'plan_sha256': file_sha256(args.plan)}))
    elif args.output is None:
        parser.error('--output is required for execution')
    elif args.worker is not None:
        run_worker(args, plan)
    else:
        run_parent(args, plan)
