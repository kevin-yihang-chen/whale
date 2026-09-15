"""Freeze the declared training-only E4 sampling study before model execution."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

from .chess_bootstrap import read_rows, sampling_keys
from .native_search import write_json
from .visual_task import file_sha256


def prepare():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from autoharness_chess_puzzle import runner
    from autoharness_chess_puzzle.harness import load_harness

    output = Path('data/chess-bootstrap-v1')
    output.mkdir(exist_ok=False)
    pilot_path = Path('results/chess-pilot-manifest-20260909.json')
    pilot = json.loads(pilot_path.read_text())
    train = pilot['splits']['train']
    assert file_sha256(Path(train['path'])) == train['sha256']
    rows = sorted(read_rows(train['path']), key=lambda r: r['extra_info']['puzzle_id'])[:32]
    ids = [r['extra_info']['puzzle_id'] for r in rows]
    sampling_keys(ids, list(range(42, 50)))
    previous = json.loads(Path('results/chess-vllm-4b-plan-20260909.json').read_text())
    harness = 'upstream/WHALE/domains/chess_puzzles/environments/chess_puzzle/base_harness.py'
    chunks = []
    for index, selected in enumerate((rows[:8], rows[8:])):
        path = output / f'chunk-{index}.parquet'
        pq.write_table(pa.Table.from_pylist(selected), path)
        chunks.append({'dataset': str(path), 'sha256': file_sha256(path),
                       'puzzle_ids': [r['extra_info']['puzzle_id'] for r in selected],
                       'trajectories': len(selected) * 8})
    tokenizer = AutoTokenizer.from_pretrained(previous['model'], local_files_only=True)
    native = load_harness(harness)
    requests = []
    for row in rows:
        example = runner.example_from_mapping(row)
        messages = runner.messages_as_dicts(runner.policy_messages_with_harness(native, runner.initialize_state(example)))
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               enable_thinking=True, return_dict=False)
        assert 0 < len(tokens) <= 4096
        requests.append({'puzzle_id': row['extra_info']['puzzle_id'], 'prompt_tokens': len(tokens)})
    preflight = output / 'prompt-preflight.json'
    write_json(preflight, {'status': 'PASS', 'model_calls': 0, 'requests': requests})
    sources = set(previous['source_sha256'])
    sources = {p for p in sources if p.startswith(('ours/', 'upstream/'))}
    sources.update(['ours/chess_bootstrap.py', 'ours/prepare_chess_bootstrap.py',
                    'ours/run_chess_bootstrap.sh', 'ours/native_search.py', 'ours/glm_gateway.py',
                    'ours/confined_exec.py', 'ours/proposer_profiles.py', str(pilot_path),
                    train['path'], str(preflight), harness,
                    'results/chess-bootstrap-resource-check-20260909.json',
                    'upstream/WHALE/domains/chess_puzzles/autoharness_chess_puzzle/grpo_dataset.py'])
    sources.update(c['dataset'] for c in chunks)
    plan = {
        'kind': 'chess_bootstrap_sampling_plan', 'role': 'training_bootstrap', 'veto_mode': 'off',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'pilot_manifest': str(pilot_path),
        'puzzle_ids': ids, 'chunks': chunks, 'replica_seeds': [[42, 43, 44, 45], [46, 47, 48, 49]],
        'harness': harness, 'assistant_token_budget': 8129, 'policy_max_tokens': 8129,
        'training_response_region_limit': 16384,
        'target_config': {'provider': 'local-vllm', 'model': previous['model'], 'max_tokens': 8129,
                          'temperature': 1.0, 'top_p': 1.0, 'top_k': 20,
                          'batch_concurrency': 4, 'request_timeout_seconds': 420,
                          'chat_template_kwargs': {'enable_thinking': True},
                          'expected_weights_sha256': previous['weights_sha256']},
        'server_arguments': previous['server_arguments'], 'startup_timeout_seconds': 360,
        'versions': {name: version(name) for name in previous['versions']},
        'source_sha256': {name: file_sha256(Path(name)) for name in sorted(sources)},
        'resources': {'replicas': 2, 'gpu_type': 'h100', 'cpus': 16, 'ram_gib': 96,
                      'chunk_0_time_limit_seconds': 3600, 'chunk_0_max_gpu_hours': 2,
                      'mail_user': 'kevin.yihang.chan@gmail.com', 'mail_type': 'ALL', 'requeue': False},
        'limitations': [
            'Training-only pilot samples, not paper test IDs or a scientific baseline result.',
            'Request seeds are sampling replicates, not eight independent training experiments.',
            'Native MH runner samples trajectories; native AgentLoop training/export remains unverified.',
            'Native MH runner has assistant-token cap; training additionally caps response region including tool text.',
            'No training, successful update, proposer search or VETO gain is claimed by this sampler.',
            'Different request batching and service lifetime from earlier eight-case runs; no bitwise equivalence claim.',
            'Stage A stops at 256 trajectories; success-triggered training uses training data only.',
            'Two replicas do not exchange gradients; merging requires every planned puzzle/seed exactly once.']}
    plan_path = Path('results/chess-bootstrap-plan-20260909.json')
    with plan_path.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'plan': str(plan_path), 'sha256': file_sha256(plan_path),
                      'selected_training_prompts': len(ids), 'max_prompt_tokens': max(r['prompt_tokens'] for r in requests)}))


if __name__ == '__main__':
    prepare()
