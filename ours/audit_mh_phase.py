"""Audit a completed MH evaluation using exact replies and independent boards.

The native replay checks input/format/parser behavior; a separate python-chess
walk checks legal reference continuations. Neither makes a new model/API call.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict
import json
from pathlib import Path
import re

from .audit_native_training_batch import require
from .local_completion import CompletionResponse
from .mh_phase import check_plan, merge_outputs, verify_receipt
from .native_search import tree_hashes, write_json
from .visual_task import file_sha256


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def board_review(example, output):
    """Check recorded actions against board rules/reference without native verifier calls."""
    import chess
    board, pointer = chess.Board(example.start_fen), 0
    solution = list(example.solution_moves)
    events = [e for e in output['harness_trace'] if e.get('actor') in {'assistant', 'verifier'}]
    require(len(events) % 2 == 0, 'Unpaired native events')
    steps, terminal = [], False
    for assistant, verdict in zip(events[::2], events[1::2], strict=True):
        require(not terminal and assistant['actor'] == 'assistant' and verdict['actor'] == 'verifier', 'Invalid event ordering')
        action, reason = assistant['parsed_action'], verdict['stop_condition']
        move = None
        if re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?', action):
            try:
                move = chess.Move.from_uci(action)
            except ValueError:
                pass
        legal = move is not None and move in board.legal_moves
        correct = legal and pointer < len(solution) and action == solution[pointer]
        if reason in {'continue', 'solved'}:
            require(correct, 'Accepted move is not a legal reference move')
            board.push(move)
            pointer += 1
            if pointer < len(solution):
                reply = chess.Move.from_uci(solution[pointer])
                require(reply in board.legal_moves, 'Invalid stored opponent reply')
                board.push(reply)
                pointer += 1
            terminal = pointer == len(solution)
            require((reason == 'solved') == terminal and verdict['terminal'] == terminal, 'Wrong solved boundary')
        elif reason == 'wrong_move':
            require(legal and not correct and verdict['terminal'], 'Wrong-move verdict contradicts board/reference')
            terminal = True
        elif reason in {'format_retry', 'malformed'}:
            require(move is None, 'Malformed verdict for a syntactically valid move')
            terminal = reason == 'malformed'
            require(verdict['terminal'] == terminal, 'Wrong format retry boundary')
        elif reason in {'illegal_retry', 'illegal'}:
            require(move is not None, 'Illegal verdict for unparseable move')
            info = verdict.get('info', {})
            require(info.get('legal_by_board') == legal, 'Native board-legality flag differs')
            require(not legal or info.get('legal_by_harness') is False, 'Unsupported illegal rejection')
            terminal = reason == 'illegal'
            require(verdict['terminal'] == terminal, 'Wrong illegal retry boundary')
        else:
            raise ValueError(f'Unreviewed native stop reason: {reason}')
        steps.append({'action': action, 'legal_by_board': legal, 'correct_reference_move': correct, 'reason': reason})
    solved = pointer == len(solution)
    require(output['reward'] == float(solved), 'Recorded reward differs from complete independent continuation')
    return {'solved': solved, 'steps': steps, 'stop': output['stop_condition'],
            'turns': output['turn_count'], 'assistant_tokens': output['metrics']['assistant_tokens_used']}


def audit(plan_path, harness):
    from autoharness_chess_puzzle import runner
    from autoharness_chess_puzzle.harness import load_harness
    from transformers import AutoTokenizer
    plan = check_plan(plan_path)
    root = Path(plan['phase_root'])
    boundary = 'baseline-ready.json' if harness == 'h0' else 'search-complete.json'
    verify_receipt(root, boundary, plan_path)
    private = root / 'evaluations' / harness
    request = json.loads((private / 'request.json').read_text())
    harness_path = Path(request['harness_path'])
    require(file_sha256(harness_path) == request['harness_sha256'], 'Evaluated harness changed')
    require(request['plan_sha256'] == file_sha256(plan_path), 'Wrong evaluation identity')
    tokenizer = AutoTokenizer.from_pretrained(plan['target_config']['model'], local_files_only=True)
    examples = {e.example_id: e for e in runner._read_examples(plan['dataset'], limit=32, seed=42)}
    records, shard_outputs = [], []
    calls_total = tokens_total = length_stops = missing_think_end = 0
    artifact_hashes = {str(p.relative_to(root)): file_sha256(p) for p in private.rglob('*') if p.is_file()}
    for replica in range(2):
        directory = private / f'replica-{replica}'
        receipt = json.loads((directory / 'result.json').read_text())
        require(receipt['status'] == 'COMPLETED' and receipt['plan_sha256'] == file_sha256(plan_path), 'Incomplete shard')
        require(receipt['replica'] == replica and receipt['harness_sha256'] == request['harness_sha256'], 'Mixed shard identity')
        for name, digest in receipt['artifact_sha256'].items():
            require(file_sha256(directory / name) == digest, f'Changed completed shard: {name}')
        proof = json.loads((directory / 'worker-weights.json').read_text())
        require(proof['status'] == 'PASS' and proof['plan_sha256'] == file_sha256(plan_path), 'Missing worker weight proof')
        require(proof['actual_worker']['values'] == [p['updated'] for p in plan['worker_probe_coordinates']], 'Wrong actual worker values')
        outputs = json.loads((directory / 'native-outputs.json').read_text())
        ids = json.loads((directory / 'ordered-ids.json').read_text())
        expected_ids = [key for key in examples if key in plan['shard_ids'][replica]]
        require(ids == expected_ids and len(outputs) == len(ids) == 16, 'Wrong private row order or coverage')
        shard_outputs.append({'ids': ids, 'outputs': outputs})
        calls = read_lines(directory / 'generations.jsonl')
        events = read_lines(directory / 'request-events.jsonl')
        starts, ends = {}, {}
        for event in events:
            require(event['event'] in {'start', 'complete'}, 'Incomplete/error request journal')
            target = starts if event['event'] == 'start' else ends
            require(event['id'] not in target, 'Duplicate request journal event')
            target[event['id']] = event
        require(set(starts) == set(ends) and len(starts) == len(calls), 'Unsettled or unrecorded requests')
        outer = Counter(canonical([e['messages'], e['max_tokens'], ends[key]['usage']]) for key, e in starts.items())
        inner = Counter(canonical([c['request']['messages'], c['request']['max_tokens'], c['raw_response']['usage']]) for c in calls)
        require(outer == inner, 'Request starts/completions differ from the actual response ledger')
        require([c['call'] for c in calls] == list(range(1, len(calls) + 1)), 'Broken completion numbering')
        available = defaultdict(deque)
        for call in calls:
            payload, response = call['request'], call['raw_response']
            choice, = response['choices']
            content, usage = choice['message']['content'], response['usage']
            require(call['weights_sha256'] == plan['target_config']['expected_weights_sha256'], 'Wrong call checkpoint identity')
            for key in ('model', 'temperature', 'top_p', 'top_k', 'seed', 'chat_template_kwargs'):
                require(payload[key] == plan['target_config'][key], f'Changed sampling field: {key}')
            require(response['model'] == payload['model'], 'Wrong served model')
            require(len(choice['token_ids']) == usage['completion_tokens'] <= payload['max_tokens'] <= 8129, 'Output token accounting differs')
            require(len(response['prompt_token_ids']) == usage['prompt_tokens'] and
                    usage['total_tokens'] == usage['completion_tokens'] + usage['prompt_tokens'], 'Prompt token accounting differs')
            require(tokenizer.decode(choice['token_ids'], skip_special_tokens=True) == content, 'Decoded token evidence differs')
            require(tokenizer.apply_chat_template(payload['messages'], tokenize=True, return_dict=False,
                    add_generation_prompt=True, **payload['chat_template_kwargs']) == response['prompt_token_ids'], 'Prompt rendering differs')
            key = canonical([payload['messages'], payload['max_tokens'], content])
            available[key].append(call)
            calls_total += 1
            tokens_total += usage['completion_tokens']
            length_stops += choice['finish_reason'] == 'length'
            missing_think_end += 248069 not in choice['token_ids']
        loaded_harness = load_harness(harness_path)
        class ReplayClient:
            def __init__(self, expected):
                self.expected = deque(e for e in expected['harness_trace'] if e.get('actor') == 'assistant')
            def complete_response(self, messages, *, max_tokens=None):
                require(bool(self.expected), 'Native replay requested an extra response')
                event = self.expected.popleft()
                key = canonical([[asdict(m) for m in messages], max_tokens, event['raw_response']])
                require(bool(available[key]), 'Native replay request/content missing from actual calls')
                call = available[key].popleft()
                return CompletionResponse(event['raw_response'], call['raw_response']['usage'])
        for key, actual in zip(ids, outputs, strict=True):
            client = ReplayClient(actual)
            replayed = runner.run_puzzle_rollout(harness=loaded_harness, example=examples[key], llm=client,
                       assistant_token_budget=8129, policy_max_tokens=8129)
            require(not client.expected and replayed == actual, 'Native replay differs from recorded output')
            review = board_review(examples[key], actual)
            require(0 < review['assistant_tokens'] <= 8129, 'Per-trajectory assistant budget differs')
            records.append({'puzzle_id': key, 'replica': replica, **review})
        require(not any(available.values()), 'Unused actual responses after replay')
    combined = merge_outputs(plan, shard_outputs)
    val_path, = (root / 'search/logs').glob(f'*/{harness}/*/val.json')
    summary = json.loads(val_path.read_text())
    require(runner.summarize_outputs(combined, summary['metadata']) == summary, 'Merged native summary differs')
    require(len(records) == 32 and len({r['puzzle_id'] for r in records}) == 32, 'Wrong overall coverage')
    require(sum(r['solved'] for r in records) == summary['solved_examples'], 'Independent solved count differs')
    require(sum(r['assistant_tokens'] for r in records) == tokens_total, 'Trajectory and call token totals differ')
    artifact_hashes[str(harness_path.relative_to(root))] = file_sha256(harness_path)
    for path in val_path.parent.iterdir():
        if path.is_file():
            artifact_hashes[str(path.relative_to(root))] = file_sha256(path)
    return {'kind': 'mh_phase_audit', 'status': 'PASS', 'harness': harness,
            'plan_sha256': file_sha256(plan_path), 'baseline_receipt_sha256': file_sha256(root / 'baseline-ready.json'),
            'examples': 32, 'solved': sum(r['solved'] for r in records), 'calls': calls_total,
            'generated_tokens': tokens_total, 'length_stop_calls': length_stops,
            'calls_without_thinking_end': missing_think_end, 'records': records,
            'actual_worker_weight_proofs': 2, 'new_model_calls': 0, 'artifact_sha256': artifact_hashes,
            'audit_source_sha256': file_sha256(Path(__file__)),
            'limitations': ['Native runner replay verifies implementation consistency; the board walk separately checks accepted legal reference continuations.',
                            'A native success can be parsed from unfinished reasoning; no finalized-answer claim.',
                            'One seed on MH optimization data, not final test accuracy or VETO gain.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--harness', choices=('h0', 'h1'), required=True)
    args = parser.parse_args()
    result = audit(args.plan, args.harness)
    root = Path(json.loads(args.plan.read_text())['phase_root'])
    path = root / ('baseline-audit.json' if args.harness == 'h0' else 'candidate-audit.json')
    require(not path.exists(), 'Preserve existing audit evidence')
    write_json(path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {'records', 'artifact_sha256', 'limitations'}}))
