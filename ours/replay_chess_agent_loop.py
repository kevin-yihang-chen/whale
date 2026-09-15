"""Replay recorded local responses through the native training AgentLoop on CPU.

E4 interface validation only: no inference, optimizer step or trained checkpoint.
Reject any changed model request instead of returning an approximate response.
"""
import argparse
import asyncio
from collections import defaultdict, deque
import faulthandler
import json
import os
from pathlib import Path
import re
import sys

from .evidence import fingerprint
from .visual_task import file_sha256


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class ArchivedResponseManager:
    def __init__(self, calls, expected_texts):
        self.calls = calls
        self.expected_texts = deque(expected_texts)
        self.used = []

    async def chat_completion(self, *, request_id, messages, sampling_params):
        text = self.expected_texts.popleft()
        key = (fingerprint(messages), text)
        if not self.calls[key]:
            raise ValueError('Native training messages differ from the archived inference request')
        call = self.calls[key].popleft()
        for field in ('max_tokens', 'temperature', 'top_p', 'top_k'):
            if sampling_params[field] != call['request'][field]:
                raise ValueError(f'Native sampling mismatch: {field}')
        self.used.append(call)
        return {'content': text, 'usage': call['raw_response']['usage']}


async def replay(args):
    print('replay: restoring native imports', file=sys.stderr, flush=True)
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.agent_loop.chess_puzzle_agent_loop import ChessPuzzleAgentLoop

    print('replay: native imports ready', file=sys.stderr, flush=True)
    config_path = Path('results/native-rsft-resolved-20260909.yaml')
    config = OmegaConf.load(config_path)
    config.data.apply_chat_template_kwargs.enable_thinking = True
    target = Path('data/models/qwen3.5-4b-851bf6e').resolve()
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    print('replay: tokenizer ready', file=sys.stderr, flush=True)
    harness = Path('upstream/WHALE/domains/chess_puzzles/environments/chess_puzzle/base_harness.py')
    os.environ.update(CHESS_PUZZLE_HARNESS_PATH=str(harness.resolve()),
                      CHESS_PUZZLE_ASSISTANT_TOKEN_BUDGET='8129', CHESS_PUZZLE_POLICY_MAX_TOKENS='8129',
                      CHESS_PUZZLE_DEFAULT_MAX_TURNS='9', CHESS_PUZZLE_MAX_TURNS_CAP='18',
                      CHESS_PUZZLE_FORMAT_RETRIES='1', CHESS_PUZZLE_ILLEGAL_RETRIES='1')
    parquet = pq.ParquetFile(args.dataset, pre_buffer=False)
    rows = [r for i in range(parquet.num_row_groups)
            for r in parquet.read_row_group(i, use_threads=False).to_pylist()]
    by_fen = {r['extra_info']['start_fen']: r['extra_info'] for r in rows}
    assert len(by_fen) == len(rows)
    calls = defaultdict(deque)
    for call in lines(args.generations):
        content = call['raw_response']['choices'][0]['message']['content']
        assert call['request']['model'] == str(target)
        assert call['request']['chat_template_kwargs'] == {'enable_thinking': True}
        calls[(fingerprint(call['request']['messages']), content)].append(call)
    reports, accepted = [], []
    for trajectory in lines(args.trajectories):
        fen = re.search(r'^FEN: (.+)$', trajectory['question'], re.M).group(1)
        info = by_fen[fen]
        print(f"replay: {info['puzzle_id']}", file=sys.stderr, flush=True)
        manager = ArchivedResponseManager(calls, [e['raw_response'] for e in trajectory['harness_trace']
                                                  if e['actor'] == 'assistant'])
        loop = ChessPuzzleAgentLoop(trainer_config=DictConfigWrap(config), server_manager=manager,
                                   tokenizer=tokenizer, processor=None, dataset_cls=None,
                                   data_config=DictConfigWrap(config.data))
        assert loop.response_length == 16384 and loop.max_assistant_tokens == loop.policy_max_tokens == 8129
        result = await loop.run({'temperature': 1.0, 'top_p': 1.0, 'top_k': 20, 'max_tokens': 8129},
                                extra_info=info)
        stop = result.extra_fields['reward_extra_info']['stop_condition']
        assert stop != 'rollout_error', result.extra_fields
        assert not manager.expected_texts
        assert len(manager.used) == trajectory['turn_count']
        assert result.prompt_ids == manager.used[0]['raw_response']['prompt_token_ids']
        assert result.reward_score == trajectory['reward']
        assert stop == trajectory['stop_condition']
        assert len(result.response_ids) == len(result.response_mask) <= 16384
        assert set(result.response_mask) <= {0, 1}
        assistant_ids = [t for t, m in zip(result.response_ids, result.response_mask, strict=True) if m]
        expected_ids = [t for c in manager.used for t in tokenizer.encode(
            c['raw_response']['choices'][0]['message']['content'], add_special_tokens=False)]
        assert assistant_ids == expected_ids
        original_generated_ids = [t for c in manager.used for t in c['raw_response']['choices'][0]['token_ids']]
        reports.append({'puzzle_id': info['puzzle_id'], 'score': result.reward_score,
                        'calls': len(manager.used), 'assistant_tokens_reencoded': len(assistant_ids),
                        'assistant_tokens_usage': sum(c['raw_response']['usage']['completion_tokens'] for c in manager.used),
                        'response_region_tokens': len(result.response_ids),
                        'nonassistant_tokens_masked': len(result.response_ids) - sum(result.response_mask),
                        'original_token_ids_preserved_by_native_reencoding': original_generated_ids == assistant_ids,
                        'stop': stop})
        if result.reward_score >= 1:
            accepted.append({'puzzle_id': info['puzzle_id'], 'prompt_ids': result.prompt_ids,
                             'response_ids': result.response_ids, 'response_mask': result.response_mask})
    assert not any(calls.values()), 'Unmatched archived calls'
    args.output.mkdir(exist_ok=False, parents=True)
    accepted_path = args.output / 'accepted-native-tokens.jsonl'
    accepted_path.write_text(''.join(json.dumps(row) + '\n' for row in accepted))
    report = {'kind': 'native_chess_agent_loop_offline_replay', 'status': 'PASS',
              'role': 'engineering', 'examples': len(reports), 'accepted': len(accepted),
              'training_steps': 0, 'new_model_calls': 0, 'puzzles': reports,
              'source_sha256': {str(p): file_sha256(p) for p in [args.dataset, args.generations,
                  args.trajectories, config_path, harness, Path(__file__),
                  Path('upstream/WHALE/domains/chess_puzzles/verl/experimental/agent_loop/chess_puzzle_agent_loop.py'),
                  Path('upstream/WHALE/domains/chess_puzzles/verl/experimental/agent_loop/agent_loop.py')]},
              'accepted_tokens_sha256': file_sha256(accepted_path),
              'limitations': ['Archived responses replace the server; not a live training rollout or weight update.',
                              'Native reward replay is not an independent verifier; run board audit separately.',
                              'The native loop reencodes content; equality with original generated IDs is measured separately.',
                              'Reencoded tokens must not silently be presented as exact original token trajectories.']}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('puzzles', 'source_sha256', 'limitations')}))


if __name__ == '__main__':
    faulthandler.dump_traceback_later(45, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--generations', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(replay(parser.parse_args()))
