"""Independent board, action, token and replica-coverage audit for E4 sampling.

Does not execute WHALE's runner/verifier or the target harness. Model performance
is measured from preserved outputs, never inferred from execution completion.
"""
from collections import Counter, defaultdict, deque
import argparse
import json
from pathlib import Path
import random
import re

from .audit_chess_decode import parse_raw
from .visual_task import file_sha256


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def action_properties(board, parsed):
    import chess
    if not re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?', parsed):
        return False, False
    try:
        move = chess.Move.from_uci(parsed)
    except chess.InvalidMoveError:
        return False, False
    return True, move in board.legal_moves


def audit(output):
    import chess
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    result = json.loads((output / 'result.json').read_text())
    plan = json.loads((output / 'plan.json').read_text())
    assert result['status'] == 'COMPLETED_PENDING_INDEPENDENT_AUDIT'
    assert result['plan_sha256'] == file_sha256(output / 'plan.json')
    for name, expected in result['artifact_sha256'].items():
        assert file_sha256(output / name) == expected, name
    for name, expected in plan['source_sha256'].items():
        assert file_sha256(Path(name)) == expected, name
    # Replay below is explicitly scoped to the frozen original h0 semantics.
    assert file_sha256(Path(plan['harness'])) == '2ecc19b7475cb2938e6a7db7e451257802bc55fe6fe1927e653d094c3ee0db6a'
    target = plan['target_config']
    tokenizer = AutoTokenizer.from_pretrained(target['model'], local_files_only=True)
    think_end = tokenizer.convert_tokens_to_ids('</think>')
    chunk = plan['chunks'][result['chunk']]
    parquet = pq.ParquetFile(chunk['dataset'], pre_buffer=False)
    rows = [r for i in range(parquet.num_row_groups)
            for r in parquet.read_row_group(i, use_threads=False).to_pylist()]
    assert [r['extra_info']['puzzle_id'] for r in rows] == chunk['puzzle_ids']
    puzzles, all_calls, observed = [], [], set()
    for replica, seeds in enumerate(plan['replica_seeds']):
        parent = output / f'replica-{replica}'
        replica_result = json.loads((parent / 'result.json').read_text())
        assert replica_result['status'] == 'COMPLETED'
        assert replica_result['model_manifest']['weights_sha256'] == target['expected_weights_sha256']
        assert [s['seed'] for s in replica_result['summaries']] == seeds
        for seed in seeds:
            directory = parent / f'seed-{seed}'
            data = rows.copy()
            random.Random(seed).shuffle(data)
            traces = read_lines(directory / 'evaluation/trajectories.jsonl')
            calls = read_lines(directory / 'generations.jsonl')
            val = json.loads((directory / 'evaluation/val.json').read_text())
            assert len(data) == len(traces) == val['num_examples']
            by_content = defaultdict(deque)
            assert [c['call'] for c in calls] == list(range(1, len(calls) + 1))
            for call in calls:
                request, response = call['request'], call['raw_response']
                choice, = response['choices']
                content = choice['message']['content']
                assert call['weights_sha256'] == target['expected_weights_sha256']
                assert response['model'] == request['model'] == target['model']
                for k in ('temperature', 'top_p', 'top_k', 'chat_template_kwargs'):
                    assert request[k] == target[k]
                assert request['seed'] == seed
                usage = response['usage']
                assert 0 < len(choice['token_ids']) == usage['completion_tokens'] <= request['max_tokens']
                assert len(response['prompt_token_ids']) == usage['prompt_tokens']
                assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
                assert tokenizer.decode(choice['token_ids'], skip_special_tokens=True) == content
                local_ids = tokenizer.apply_chat_template(request['messages'], tokenize=True,
                    add_generation_prompt=True, return_dict=False, **request['chat_template_kwargs'])
                assert local_ids == response['prompt_token_ids'], 'Prompt tokenization mismatch'
                fen = re.search(r'^FEN: (.+)$', request['messages'][-1]['content'], re.M).group(1)
                by_content[(fen, content)].append(call)
            solved_in_seed = 0
            for row, trajectory in zip(data, traces, strict=True):
                info = row['extra_info']
                key = (info['puzzle_id'], seed)
                assert key not in observed, 'Repeated puzzle/seed'
                observed.add(key)
                assert f"FEN: {info['start_fen']}\n" in trajectory['question']
                board, solution = chess.Board(info['start_fen']), list(info['solution_moves'])
                pointer, tokens, formats, illegals = 0, 0, 0, 0
                steps, terminal = [], False
                events = trajectory['harness_trace']
                assert events and len(events) % 2 == 0
                for assistant, verifier in zip(events[::2], events[1::2], strict=True):
                    assert not terminal
                    assert assistant['actor'] == 'assistant' and verifier['actor'] == 'verifier'
                    call = by_content[(board.fen(), assistant['raw_response'])].popleft()
                    request, response = call['request'], call['raw_response']
                    assert request['max_tokens'] == min(plan['policy_max_tokens'], plan['assistant_token_budget'] - tokens)
                    parsed = parse_raw(assistant['raw_response'])
                    assert parsed == assistant['parsed_action']
                    tokens += response['usage']['completion_tokens']
                    assert tokens == assistant['assistant_tokens_used'] <= plan['assistant_token_budget']
                    valid, legal = action_properties(board, parsed)
                    correct = legal and parsed == solution[pointer]
                    if not valid:
                        terminal = formats >= 1
                        reason = 'malformed' if terminal else 'format_retry'
                        formats += 1
                    elif not legal:
                        terminal = illegals >= 1
                        reason = 'illegal' if terminal else 'illegal_retry'
                        illegals += 1
                    elif not correct:
                        reason, terminal = 'wrong_move', True
                    else:
                        board.push_uci(parsed)
                        pointer += 1
                        if pointer < len(solution):
                            reply = chess.Move.from_uci(solution[pointer])
                            assert reply in board.legal_moves
                            board.push(reply)
                            pointer += 1
                        terminal = pointer >= len(solution)
                        reason = 'solved' if terminal else 'continue'
                    assert verifier['terminal'] == terminal and verifier['stop_condition'] == reason
                    steps.append({'action': parsed, 'legal': legal, 'correct': correct, 'reason': reason})
                solved = pointer >= len(solution)
                expected_stop = steps[-1]['reason'] if terminal else ('assistant_token_budget'
                    if tokens >= plan['assistant_token_budget'] else 'max_policy_calls')
                assert trajectory['stop_condition'] == expected_stop
                assert trajectory['correct'] == solved and trajectory['reward'] == float(solved)
                assert trajectory['turn_count'] == len(steps) <= 9
                puzzles.append({'puzzle_id': info['puzzle_id'], 'seed': seed, 'solved': solved,
                                'tokens': tokens, 'steps': steps, 'stop': expected_stop})
                solved_in_seed += solved
            assert not any(by_content.values()), 'Unmatched generation calls'
            assert val['solved_examples'] == solved_in_seed
            assert val['success_rate'] == solved_in_seed / len(data)
            assert val['mean_turn_count'] == len(calls) / len(data)
            all_calls.extend(calls)
    expected = {(p, s) for p in chunk['puzzle_ids'] for s in range(42, 50)}
    assert observed == expected and len(puzzles) == result['trajectories'] == chunk['trajectories']
    solved = sum(p['solved'] for p in puzzles)
    assert solved == result['native_solved']
    return {'kind': 'independent_chess_bootstrap_audit', 'status': 'PASS', 'role': 'training_bootstrap',
            'job_id': result['slurm_job_id'], 'chunk': result['chunk'], 'trajectories': len(puzzles),
            'unique_prompts': len(chunk['puzzle_ids']), 'solved': solved,
            'unique_prompts_solved': len({p['puzzle_id'] for p in puzzles if p['solved']}),
            'correct_first_moves': sum(p['steps'][0]['correct'] for p in puzzles),
            'legal_first_moves': sum(p['steps'][0]['legal'] for p in puzzles),
            'model_calls': len(all_calls), 'output_tokens': sum(p['tokens'] for p in puzzles),
            'length_stops': sum(c['raw_response']['choices'][0]['finish_reason'] == 'length' for c in all_calls),
            'calls_with_thinking_end': sum(think_end in c['raw_response']['choices'][0]['token_ids'] for c in all_calls),
            'terminal_reasons': dict(Counter(p['stop'] for p in puzzles)), 'puzzles': puzzles,
            'training_steps': 0, 'external_api_calls': 0, 'result_sha256': file_sha256(output / 'result.json'),
            'audit_source_sha256': file_sha256(Path(__file__)), 'limitations': plan['limitations']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.directory)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('puzzles', 'limitations')}))
