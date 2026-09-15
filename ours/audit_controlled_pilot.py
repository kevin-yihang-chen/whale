"""Verify both fresh controlled E4 batches against native order and all requests."""
import argparse
import asyncio
from collections import Counter
import hashlib
import gzip
import json
import os
from pathlib import Path
import re

from .alternation_training import write_new
from .audit_alternation_training import replay_row
from .audit_native_training_batch import RequestPool, load_batch, require
from .controlled_pilot import CPU_NAME, check
from .visual_task import file_sha256


def runtime_configuration(plan, job):
    from omegaconf import OmegaConf
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    require(re.fullmatch(r'[0-9]+', job), 'Invalid job identity')
    name = f"controlled-weight-only-seed{plan['seed']}-{job}"
    raw = Path(plan['resolved_config']).read_text().replace(CPU_NAME, name)
    config = migrate_legacy_reward_impl(OmegaConf.create(raw[raw.index('model_engine: dp\n'):]))
    require(config.trainer.total_training_steps == plan['native_batch_steps'] == 2 and
            config.trainer.online_rsft.iterations == 0, 'Wrong native stopping rule')
    require(not config.reward.reward_model.enable_resource_pool, 'Unexpected reward resource pool')
    config.reward.reward_model.nnodes = config.trainer.nnodes
    config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
    config.actor_rollout_ref.actor.optim.total_training_steps = config.trainer.total_training_steps
    config.critic.optim.total_training_steps = config.trainer.total_training_steps
    context = {'configured_model_path': plan['model'], 'job_id': job, 'recording_only': True,
               'configuration_sha256': hashlib.sha256(json.dumps(
                   OmegaConf.to_container(config, resolve=True), sort_keys=True).encode()).hexdigest()}
    return config, context


async def audit(plan_path, directory):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from omegaconf import OmegaConf
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from autoharness_chess_puzzle.harness import load_harness
    plan = check(plan_path)
    with gzip.open(directory / 'batch-1.jsonl.gz', 'rt') as stream:
        job = json.loads(next(stream))['context']['job_id']
    config, context = runtime_configuration(plan, job)
    require(directory.resolve() == (Path(config.trainer.default_local_dir) / 'audit').resolve(), 'Wrong audit directory')
    start_path = Path(f'results/controlled-pilot-{job}/start.json')
    start = json.loads(start_path.read_text())
    require(start['status'] == 'PASS' and start['plan_sha256'] == file_sha256(plan_path), 'Different trial receipt')
    environment = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env.env_vars, resolve=True)
    os.environ.update({k: str(v) for k, v in environment.items() if k.startswith('CHESS_PUZZLE_')})
    require(environment['CHESS_PUZZLE_HARNESS_PATH'] == plan['harness'], 'Wrong training harness')
    harness = load_harness(Path(plan['harness']))
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
    data = pq.read_table(plan['dataset'], use_threads=False).to_pylist()
    by_id = {row['extra_info']['puzzle_id']: row['extra_info'] for row in data}
    require(len(by_id) == len(data) == 128, 'Wrong training dataset coverage')
    pool = RequestPool(directory, plan['model'])
    require(all(c == context for c in pool.contexts), 'Request configuration differs from frozen trial')
    prompt_width, response_width = [int(config.actor_rollout_ref.rollout[k]) for k in ('prompt_length', 'response_length')]
    width = prompt_width + response_width
    shapes = {'prompts': [prompt_width], 'responses': [response_width], 'response_mask': [response_width],
              'attention_mask': [width], 'input_ids': [width], 'position_ids': [4, width],
              'token_level_scores': [response_width]}
    reports, receipts = [], []
    for step in (1, 2):
        header, rows, receipt = load_batch(directory, step=step)
        require(header['context'] == context, 'Batch context differs')
        require(len(rows) == 64 and set(header['tensors']) == set(shapes), 'Wrong batch size or tensors')
        for name, shape in shapes.items():
            dtype = 'torch.float32' if name == 'token_level_scores' else 'torch.int64'
            require(header['tensors'][name] == {'shape': [64] + shape, 'dtype': dtype}, f'Wrong tensor layout: {name}')
        require(receipt['recorder_sha256'] == plan['source_sha256']['ours/native_training_trace.py'], 'Wrong recorder')
        expected_ids = plan['native_dataset_order']['planned_batch_ids'][step-1]
        ordered_ids = [row['metadata']['extra_info']['puzzle_id'] for row in rows]
        require(ordered_ids == [key for key in expected_ids for _ in range(8)], 'Actual sampled rows differ from native frozen order')
        checks = [await replay_row(row, pool=pool, config=config, tokenizer=tokenizer,
                                  harness=harness, by_id=by_id) for row in rows]
        require(Counter(c['puzzle_id'] for c in checks) == Counter({key: 8 for key in expected_ids}), 'Wrong trajectory multiplicities')
        reports.append({'step': step, 'batch_sha256': receipt['sha256'], 'puzzle_ids': expected_ids,
            'accepted': sum(c['accepted'] for c in checks),
            'accepted_loss_tokens': sum(c['assistant_reencoded_tokens'] for c in checks if c['accepted']),
            'trajectory_checks': checks})
        receipts.extend([directory / receipt['file'], directory / f'batch-{step}.receipt.json'])
    require(not any(pool.calls.values()), 'Some recorded requests were not matched to either batch')
    require(pool.accounting['recorded_starts'] <= plan['max_policy_calls'] and
            pool.accounting['known_completion_tokens'] <= plan['max_assistant_output_tokens'], 'Exceeded frozen generation budget')
    sources = ['ours/audit_controlled_pilot.py', 'ours/audit_alternation_training.py',
               'ours/audit_native_training_batch.py', 'ours/native_training_trace.py', 'ours/controlled_pilot.py']
    return {'kind': 'controlled_pilot_batch_audit', 'status': 'PASS', 'job_id': job,
        'plan_sha256': file_sha256(plan_path), 'condition': plan['condition'], 'seed': plan['seed'],
        'context': context, 'trajectories': 128, 'batches': reports, 'request_accounting': pool.accounting,
        'artifact_sha256': {str(p): file_sha256(p) for p in [*sorted(directory.glob('requests-*.jsonl')), *receipts, start_path]},
        'audit_source_sha256': {name: file_sha256(Path(name)) for name in sources}, 'new_model_calls': 0,
        'limitations': ['Native request/token/mask replay plus independent board checks; not heldout performance.',
            'Frozen model path does not fingerprint in-place NCCL receiver values after each update.',
            'Exact duplicate request/reply matches establish multiplicity, not a unique request-task lineage.',
            'Native reencoded response tokens are checked; original generation token IDs are unavailable.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), 'Preserve existing audit')
    report = asyncio.run(audit(args.plan, args.directory))
    write_new(args.output, report)
    print(json.dumps({'status': 'PASS', 'accepted_per_batch': [r['accepted'] for r in report['batches']]}))
