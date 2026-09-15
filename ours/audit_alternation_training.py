"""Verify the selected-harness E4 batch against frozen inputs and native replay.

Replay uses recorded replies only. The independent board walk checks the legal
reference continuation, while native replay checks parser, prompts and masks.
This is evidence for the E3-to-E4 handoff, not an additional training mechanism.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from .audit_native_training_batch import RequestPool, check_tensors, load_batch, require
from .visual_task import file_sha256


def runtime_configuration(plan, job_id):
    from omegaconf import OmegaConf
    from .recovery_gate import native_runtime_config
    require(re.fullmatch(r'[0-9]+', job_id), 'Invalid recorded Slurm job identity')
    raw = Path(plan['resolved_config']).read_text().replace(
        'alternation-cpu-preflight', f'alternation-{job_id}')
    config = native_runtime_config(OmegaConf.create(raw[raw.index('model_engine: dp\n'):]))
    context = {'configured_model_path': plan['model'], 'job_id': job_id, 'recording_only': True,
               'configuration_sha256': hashlib.sha256(json.dumps(
                   OmegaConf.to_container(config, resolve=True), sort_keys=True).encode()).hexdigest()}
    return config, context


def board_continuation(info, result, harness):
    """Check board/reference and retry boundaries without the native verifier."""
    import chess
    board = chess.Board(info['start_fen'])
    solution = info['solution_moves']
    require(board.is_valid() and bool(solution), 'Invalid board or empty reference')
    events = result.extra_fields['extras']['events']
    require(len(events) % 2 == 0, 'Incomplete native event pairs')
    pointer = formats = illegals = assistant_tokens = 0
    terminal, reason, steps = False, None, []
    for action, verdict in zip(events[::2], events[1::2], strict=True):
        require(not terminal and action['actor'] == 'assistant' and verdict['actor'] == 'verifier',
                'Invalid action/verifier ordering')
        parsed = action['parsed_action']
        move = None
        if re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?', parsed):
            try:
                move = chess.Move.from_uci(parsed)
            except ValueError:
                pass
        legal = move is not None and move in board.legal_moves
        if move is None:
            terminal = formats >= harness.format_retry_budget
            reason = 'malformed' if terminal else 'format_retry'
            formats += not terminal
        elif not legal:
            terminal = illegals >= harness.illegal_move_retry_budget
            reason = 'illegal' if terminal else 'illegal_retry'
            illegals += not terminal
            require(verdict['info']['legal_by_board'] is False, 'Incorrect board-legality flag')
        elif parsed != solution[pointer]:
            terminal, reason = True, 'wrong_move'
        else:
            board.push(move)
            pointer += 1
            if pointer < len(solution):
                reply = chess.Move.from_uci(solution[pointer])
                require(reply in board.legal_moves, 'Illegal reference opponent reply')
                board.push(reply)
                pointer += 1
            terminal = pointer == len(solution)
            reason = 'solved' if terminal else 'continue'
        require(verdict['terminal'] == terminal and verdict['stop_condition'] == reason,
                'Independent board verdict differs')
        require(type(action['completion_tokens']) is int and action['completion_tokens'] > 0,
                'Invalid event token accounting')
        assistant_tokens += action['completion_tokens']
        require(action['assistant_tokens_used'] == assistant_tokens, 'Event cumulative usage differs')
        steps.append({'action': parsed, 'legal': legal, 'reason': reason})
    outcome = result.extra_fields['reward_extra_info']
    stop = outcome['stop_condition']
    require(outcome['assistant_tokens_used'] == assistant_tokens and outcome['policy_calls'] == len(steps),
            'Final event accounting differs')
    solved = pointer == len(solution)
    if stop == 'response_length':
        # Native appends feedback before awarding reward: even a solved board can
        # have zero training reward if feedback no longer fits. Replay proves the cap.
        expected_reward = 0.
    elif terminal:
        require(stop == reason, 'Terminal board reason differs')
        expected_reward = float(solved)
    else:
        require(stop in {'assistant_token_budget', 'max_assistant_turns'}, 'Unsupported nonterminal stop')
        expected_reward = 0.
    require(result.reward_score == expected_reward == outcome['score'], 'Independent training reward differs')
    require(outcome['outcome'] == ('solved' if expected_reward else 'loss'), 'Unexpected outcome category')
    return {'board_solved': solved, 'accepted': expected_reward > .5,
            'solved_but_feedback_truncated': solved and stop == 'response_length', 'steps': steps}


async def replay_row(row, *, pool, config, tokenizer, harness, by_id):
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.agent_loop.chess_puzzle_agent_loop import ChessPuzzleAgentLoop
    metadata, tensors = row['metadata'], row['tensors']
    info = metadata['extra_info']
    require(info == by_id[info['puzzle_id']], 'Training metadata differs from frozen dataset')
    events = metadata['extras']['events']
    require(all(e['actor'] in {'assistant', 'verifier'} for e in events), 'Rollout contains error events')
    manager, pending, used = pool.manager([e['raw_response'] for e in events if e['actor'] == 'assistant'])
    loop = ChessPuzzleAgentLoop(trainer_config=DictConfigWrap(config), server_manager=manager,
                               tokenizer=tokenizer, processor=None, dataset_cls=None,
                               data_config=DictConfigWrap(config.data))
    sampling = {key: config.actor_rollout_ref.rollout[key] for key in ('temperature', 'top_p', 'top_k')}
    sampling.update(repetition_penalty=1., logprobs=False)
    replayed = await loop.run(sampling, extra_info=info)
    require(not pending, 'Unused expected replies')
    require(replayed.extra_fields['extras'] == metadata['extras'], 'Replayed events differ')
    require(replayed.extra_fields['reward_extra_info'] == metadata['reward_extra_info'], 'Replayed outcome differs')
    check_tensors(tensors, replayed, tokenizer.pad_token_id)
    board = board_continuation(info, replayed, harness)
    for request, response in used:
        prompt_ids = tokenizer.apply_chat_template(request['messages'], tokenize=True,
            add_generation_prompt=True, return_dict=False, **dict(config.data.apply_chat_template_kwargs))
        require(len(prompt_ids) == response['usage']['prompt_tokens'], 'Request prompt token count differs')
    require(sum(response['usage']['completion_tokens'] for _, response in used) ==
            metadata['reward_extra_info']['assistant_tokens_used'], 'Trajectory response usage differs')
    return {'row': row['index'], 'puzzle_id': info['puzzle_id'], 'score': replayed.reward_score,
            'stop': metadata['reward_extra_info']['stop_condition'], 'calls': len(used),
            'request_trace_ids': [start['trace_id'] for start, _ in used],
            'response_tokens': len(replayed.response_ids),
            'assistant_reencoded_tokens': sum(replayed.response_mask), **board}


async def audit(plan_path, directory):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from autoharness_chess_puzzle.harness import load_harness
    plan = json.loads(plan_path.read_text())
    require(plan['kind'] == 'native_alternation_training_plan' and plan['veto_mode'] == 'off',
            'Wrong frozen training phase')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(Path(name)) == digest, f'Frozen source changed: {name}')
    header, rows, receipt = load_batch(directory)
    require(len(rows) == plan['fresh_trajectories'], 'Wrong trajectory count')
    config, context = runtime_configuration(plan, header['context']['job_id'])
    require(header['context'] == context, 'Batch runtime identity differs from frozen config')
    require(directory.resolve() == (Path(config.trainer.default_local_dir) / 'audit').resolve(),
            'Batch directory differs from frozen runtime')
    start_path = Path(f"results/alternation-rsft-start-{context['job_id']}.json")
    start = json.loads(start_path.read_text())
    require(start['status'] == 'PASS' and start['plan_sha256'] == file_sha256(plan_path) and
            start['run_name'] == f"alternation-{context['job_id']}", 'Wrong runtime start receipt')
    env = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env.env_vars, resolve=True)
    require(env['CHESS_PUZZLE_HARNESS_PATH'] == plan['harness'], 'Frozen selected harness differs')
    os.environ.update({k: str(v) for k, v in env.items() if k.startswith('CHESS_PUZZLE_')})
    harness = load_harness(Path(plan['harness']))
    # Reviewed selected harness uses ordinary legal-move admission and explicit retries.
    require(harness.format_retry_budget == harness.illegal_move_retry_budget == 1,
            'Unreviewed retry configuration')
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
    data = pq.read_table(plan['dataset'], use_threads=False).to_pylist()
    by_id = {row['extra_info']['puzzle_id']: row['extra_info'] for row in data}
    data_report = json.loads(Path(plan['data_report']).read_text())
    require(list(by_id) == data_report['puzzle_ids'] and len(by_id) == len(data) == 8, 'Wrong frozen row coverage')
    prompt_width, response_width = (int(config.actor_rollout_ref.rollout[key])
                                    for key in ('prompt_length', 'response_length'))
    total_width = prompt_width + response_width
    shapes = {'prompts': [prompt_width], 'responses': [response_width], 'response_mask': [response_width],
              'attention_mask': [total_width], 'input_ids': [total_width],
              'position_ids': [4, total_width], 'token_level_scores': [response_width]}
    require(set(header['tensors']) == set(shapes), 'Wrong native tensor interface')
    for name, shape in shapes.items():
        dtype = 'torch.float32' if name == 'token_level_scores' else 'torch.int64'
        require(header['tensors'][name] == {'shape': [len(rows)] + shape, 'dtype': dtype},
                f'Tensor schema differs: {name}')
    require(receipt['recorder_sha256'] == plan['source_sha256']['ours/native_training_trace.py'],
            'Wrong recorder identity')
    pool = RequestPool(directory, plan['model'])
    require(all(c == context for c in pool.contexts), 'Request runtime identity differs')
    reports = []
    for row in rows:
        reports.append(await replay_row(row, pool=pool, config=config, tokenizer=tokenizer,
                                        harness=harness, by_id=by_id))
    require(not any(pool.calls.values()), 'Unmatched recorded requests')
    coverage = Counter(row['puzzle_id'] for row in reports)
    require(set(coverage) == set(by_id) and set(coverage.values()) == {config.actor_rollout_ref.rollout.n},
            'Incorrect per-puzzle trajectory count')
    artifacts = {str(p): file_sha256(p) for p in directory.glob('requests-*.jsonl')}
    for p in (directory / receipt['file'], directory / 'batch-1.receipt.json', start_path):
        artifacts[str(p)] = file_sha256(p)
    return {'kind': 'native_alternation_batch_audit', 'status': 'PASS', 'job_id': context['job_id'],
            'plan_sha256': file_sha256(plan_path), 'context': context, 'batch_sha256': receipt['sha256'],
            'harness_sha256': file_sha256(Path(plan['harness'])), 'trajectories': len(rows),
            'accepted': sum(r['accepted'] for r in reports),
            'accepted_loss_tokens': sum(r['assistant_reencoded_tokens'] for r in reports if r['accepted']),
            'board_solved': sum(r['board_solved'] for r in reports),
            'request_accounting': pool.accounting, 'trajectory_checks': reports,
            'artifact_sha256': artifacts, 'audit_source_sha256': file_sha256(Path(__file__)),
            'audit_dependency_sha256': {name: file_sha256(Path(name)) for name in
                ('ours/audit_native_training_batch.py', 'ours/recovery_gate.py',
                 'ours/native_training_trace.py', 'ours/evidence.py', 'ours/visual_task.py')},
            'new_model_calls': 0, 'optimizer_updates': 0,
            'limitations': ['Native replay and independent board continuation are distinct checks.',
                'Response tensors contain native reencoded content; original generation token IDs are not established.',
                'This audit fails closed on replies discarded before event recording; such calls cannot be silently omitted.',
                'Duplicate identical requests/replies are matched by multiplicity; exact task lineage is not established.',
                'This is a batch audit, not numerical proof of served weights, optimizer success or generalization.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing audit evidence')
    report = asyncio.run(audit(args.plan, args.directory))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in
          {'trajectory_checks', 'artifact_sha256', 'limitations'}}))
