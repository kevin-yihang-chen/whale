"""Audit recorded E4 batches against exact requests and native CPU replay.

Board/action checks are independent of the native verifier. Request construction
and token masks are checked by replaying the original AgentLoop without inference.
This does not establish optimizer success, parameter changes or model quality.
"""
import argparse
import asyncio
from collections import Counter, defaultdict, deque
import gzip
import json
import math
import os
from pathlib import Path

from .audit_chess_decode import parse_raw
from .evidence import fingerprint
from .native_training_trace import summarize_requests
from .visual_task import file_sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_batch(directory, step=1):
    receipt = json.loads((directory / f'batch-{step}.receipt.json').read_text())
    require(receipt['kind'] == 'native_training_batch_receipt' and receipt['status'] == 'COMPLETE',
            'A completed native batch receipt is required')
    require(receipt['step'] == step and receipt['file'] == f'batch-{step}.jsonl.gz', 'Invalid batch identity')
    path = directory / receipt['file']
    require(file_sha256(path) == receipt['sha256'], 'Batch hash mismatch')
    with gzip.open(path, 'rt') as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    require(header['kind'] == 'native_training_batch_header' and header['step'] == step, 'Invalid batch header')
    require(len(rows) == header['count'] == receipt['count'] > 0, 'Incomplete tensor batch')
    for i, row in enumerate(rows):
        require(row['kind'] == 'native_training_trajectory' and row['index'] == i, 'Batch row order changed')
        require(set(row['tensors']) == set(header['tensors']), 'Tensor fields differ from schema')
        for name, schema in header['tensors'].items():
            require(schema['shape'][0] == len(rows), 'Tensor schema count differs')
            value, dimensions = row['tensors'][name], schema['shape'][1:]
            def shape_matches(array, shape):
                if not shape:
                    return not isinstance(array, list)
                return (isinstance(array, list) and len(array) == shape[0]
                        and all(shape_matches(item, shape[1:]) for item in array))
            require(shape_matches(value, dimensions), f'Tensor shape differs: {i}/{name}')
    return header, rows, receipt


class RequestPool:
    def __init__(self, directory, model):
        accounting = summarize_requests(directory)
        require(accounting['observed_requests_fully_accounted'], 'Request journal is incomplete or contains errors')
        self.accounting = accounting
        self.calls = defaultdict(deque)
        self.used = []
        self.contexts = []
        for path in sorted(directory.glob('requests-*.jsonl')):
            starts = {}
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event['stage'] == 'started':
                    starts[event['trace_id']] = event
                elif event['stage'] == 'completed':
                    start = starts[event['trace_id']]
                    response = event['response']
                    raw = response['raw_response']
                    require(start['context']['configured_model_path'] == model == raw['model'], 'Wrong request model')
                    require(len(raw['choices']) == 1, 'Expected exactly one completion choice')
                    content = raw['choices'][0]['message'].get('content') or ''
                    require(content == response['content'] and response['usage'] == raw['usage'], 'Response wrapper differs')
                    usage = response['usage']
                    require(type(usage['completion_tokens']) is int and
                            0 < usage['completion_tokens'] <= start['sampling_params']['max_tokens'], 'Invalid usage')
                    require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'Usage totals differ')
                    self.calls[(fingerprint(start['messages']), content)].append((start, response))
                    self.contexts.append(start['context'])

    def manager(self, texts):
        pool = self
        expected = deque(texts)
        used = []
        class Manager:
            async def chat_completion(self, request_id, *, messages, sampling_params):
                require(bool(expected), 'Replay requested an unrecorded response')
                text = expected.popleft()
                key = (fingerprint(messages), text)
                candidates = pool.calls[key]
                match = next((i for i, pair in enumerate(candidates) if pair[0]['sampling_params'] == sampling_params), None)
                require(match is not None, 'Native replay request or sampling differs from the journal')
                start, response = candidates[match]
                del candidates[match]
                used.append((start, response))
                pool.used.append((start, response))
                return response
        return Manager(), expected, used


def check_board(info, events, score, stop):
    import chess
    board = chess.Board(info['start_fen'])
    require(board.is_valid(), 'Invalid initial board')
    solution = info['solution_moves']
    pointer = formats = illegals = 0
    terminal = False
    require(bool(events) and len(events) % 2 == 0, 'Incomplete or unsupported event sequence')
    for action, verdict in zip(events[::2], events[1::2], strict=True):
        require(not terminal and action['actor'] == 'assistant' and verdict['actor'] == 'verifier', 'Invalid event order')
        parsed = parse_raw(action['raw_response'])
        require(parsed == action['parsed_action'], 'Independent action parser differs')
        try:
            move = chess.Move.from_uci(parsed)
        except ValueError:
            move = None
        if move is None:
            terminal, reason = formats >= 1, 'malformed' if formats >= 1 else 'format_retry'
            formats += 1
        elif move not in board.legal_moves:
            terminal, reason = illegals >= 1, 'illegal' if illegals >= 1 else 'illegal_retry'
            illegals += 1
        elif parsed != solution[pointer]:
            terminal, reason = True, 'wrong_move'
        else:
            board.push(move)
            pointer += 1
            if pointer < len(solution):
                reply = chess.Move.from_uci(solution[pointer])
                require(reply in board.legal_moves, 'Illegal reference opponent move')
                board.push(reply)
                pointer += 1
            terminal = pointer == len(solution)
            reason = 'solved' if terminal else 'continue'
        require(verdict['terminal'] == terminal and verdict['stop_condition'] == reason, 'Independent board verdict differs')
    require(float(pointer == len(solution)) == score, 'Independent board reward differs')
    require(stop == reason if terminal else stop in ('assistant_token_budget', 'max_policy_calls'),
            'Unsupported or inconsistent final stop condition')


def check_tensors(tensors, replayed, pad_id):
    prompt_length, response_length = len(tensors['prompts']), len(tensors['responses'])
    prompt = replayed.prompt_ids[-prompt_length:]
    response = replayed.response_ids
    require(0 < len(response) <= response_length, 'Replay response is empty or exceeds batch width')
    expected_prompt = [pad_id] * (prompt_length - len(prompt)) + prompt
    expected_response = response + [pad_id] * (response_length - len(response))
    expected_mask = replayed.response_mask + [0] * (response_length - len(response))
    attention = [0] * (prompt_length - len(prompt)) + [1] * (len(prompt) + len(response)) + [0] * (response_length - len(response))
    for name, expected in [('prompts', expected_prompt), ('responses', expected_response),
                           ('response_mask', expected_mask), ('attention_mask', attention),
                           ('input_ids', expected_prompt + expected_response)]:
        require(tensors[name] == expected, f'Native replay tensor differs: {name}')
        require(all(type(x) is int for x in tensors[name]), f'Invalid integer tensor: {name}')
    expected_scores = [0.] * response_length
    expected_scores[len(response) - 1] = replayed.reward_score
    require(tensors['token_level_scores'] == expected_scores, 'Reward tensor differs from replay')
    require(all(math.isfinite(x) for x in tensors['token_level_scores']), 'Nonfinite reward tensor')
    positions = tensors['position_ids']
    require(len(positions) == 4, 'Expected native Qwen3.5 text and three rotary position axes')
    for axis in positions:
        require(len(axis) == len(attention) and all(type(x) is int for x in axis), 'Invalid position tensor')
        require([p for p, attended in zip(axis, attention) if attended] == list(range(sum(attention))),
                'Attended text positions differ')


async def audit(args):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.agent_loop.chess_puzzle_agent_loop import ChessPuzzleAgentLoop

    plan = json.loads(args.plan.read_text())
    for path, expected in plan['source_sha256'].items():
        require(file_sha256(Path(path)) == expected, f'Frozen source changed: {path}')
    config = OmegaConf.load(plan['resolved_config'])
    env = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env.env_vars, resolve=True)
    os.environ.update({key: str(value) for key, value in env.items() if key.startswith('CHESS_PUZZLE_')})
    require(file_sha256(Path(env['CHESS_PUZZLE_HARNESS_PATH'])) ==
            '2ecc19b7475cb2938e6a7db7e451257802bc55fe6fe1927e653d094c3ee0db6a', 'Requires original h0')
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
    data = pq.read_table(plan['train_dataset'], use_threads=False).to_pylist()
    by_id = {row['extra_info']['puzzle_id']: row['extra_info'] for row in data}
    header, rows, receipt = load_batch(args.directory)
    require(len(rows) == plan['fresh_trajectories'], 'Wrong trajectory count')
    prompt_width = int(config.actor_rollout_ref.rollout.prompt_length)
    response_width = int(config.actor_rollout_ref.rollout.response_length)
    total_width = prompt_width + response_width
    shapes = {'prompts': [prompt_width], 'responses': [response_width],
              'response_mask': [response_width], 'attention_mask': [total_width],
              'input_ids': [total_width], 'position_ids': [4, total_width],
              'token_level_scores': [response_width]}
    require(set(header['tensors']) == set(shapes), 'Tensor fields differ from the frozen training interface')
    for name, shape in shapes.items():
        expected_dtype = 'torch.float32' if name == 'token_level_scores' else 'torch.int64'
        require(header['tensors'][name] == {'shape': [len(rows)] + shape, 'dtype': expected_dtype},
                f'Tensor schema differs from frozen configuration: {name}')
    require(receipt['recorder_sha256'] == plan['source_sha256']['ours/native_training_trace.py'], 'Wrong recorder identity')
    pool = RequestPool(args.directory, plan['model'])
    require(all(context == header['context'] for context in pool.contexts), 'Request and batch context differs')
    reports = []
    for row in rows:
        metadata = row['metadata']
        info = metadata['extra_info']
        require(info == by_id[info['puzzle_id']], 'Training metadata differs from the frozen dataset')
        events = metadata['extras']['events']
        manager, pending, used = pool.manager([e['raw_response'] for e in events if e['actor'] == 'assistant'])
        loop = ChessPuzzleAgentLoop(trainer_config=DictConfigWrap(config), server_manager=manager,
                                   tokenizer=tokenizer, processor=None, dataset_cls=None,
                                   data_config=DictConfigWrap(config.data))
        result = await loop.run({'temperature': 1., 'top_p': 1., 'top_k': 20,
                                 'repetition_penalty': 1., 'logprobs': False}, extra_info=info)
        require(not pending, 'Unused expected replies')
        require(result.extra_fields['extras']['events'] == events, 'Replayed event sequence differs')
        require(result.extra_fields['reward_extra_info'] == metadata['reward_extra_info'], 'Replayed outcome metadata differs')
        check_tensors(row['tensors'], result, tokenizer.pad_token_id)
        stop = result.extra_fields['reward_extra_info']['stop_condition']
        check_board(info, events, result.reward_score, stop)
        for request, response in used:
            prompt_ids = tokenizer.apply_chat_template(request['messages'], tokenize=True,
                add_generation_prompt=True, return_dict=False, enable_thinking=True)
            require(len(prompt_ids) == response['usage']['prompt_tokens'], 'Prompt token count differs')
        reports.append({'row': row['index'], 'puzzle_id': info['puzzle_id'], 'score': result.reward_score,
                        'stop': stop, 'calls': len(used), 'response_tokens': len(result.response_ids),
                        'assistant_reencoded_tokens': sum(result.response_mask)})
    require(not any(pool.calls.values()), 'Unmatched journal calls')
    coverage = Counter(row['puzzle_id'] for row in reports)
    require(set(coverage) == set(plan['puzzle_ids']) and set(coverage.values()) == {8}, 'Wrong puzzle coverage')
    report = {'kind': 'native_training_batch_audit', 'status': 'PASS', 'job_id': header['context']['job_id'],
              'trajectories': len(rows), 'accepted': sum(r['score'] > .5 for r in reports),
              'request_accounting': pool.accounting, 'trajectory_checks': reports,
              'plan_sha256': file_sha256(args.plan), 'batch_sha256': receipt['sha256'],
              'audit_source_sha256': file_sha256(Path(__file__)), 'new_model_calls': 0, 'optimizer_updates': 0,
              'limitations': ['Native request/token replay and independent board checks are distinct checks.',
                              'Response tensors contain native reencoded content; original generation token IDs are not established here.',
                              'Unfinished reasoning may satisfy the original move parser; no finalized-answer guarantee.',
                              'This audit does not prove optimizer success, updated weights or held-out improvement.']}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ['status', 'job_id', 'trajectories', 'accepted', 'new_model_calls']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(audit(parser.parse_args()))
