"""Independently replay recorded Chess decisions and check token evidence.

No import of WHALE's parser, verifier or runner. Engineering outputs remain
excluded from scientific evaluation even when their traces pass this audit.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import random
import re

from .audit_chess_decode import parse_raw
from .evidence import fingerprint
from .visual_task import file_sha256


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(output):
    import chess
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    result = json.loads((output / "result.json").read_text())
    plan = json.loads((output / "plan.json").read_text())
    assert result["status"] == "COMPLETED" and result["role"] == plan["role"] == "engineering"
    assert file_sha256(output / "plan.json") == result["plan_sha256"]
    for name, expected in result["artifact_sha256"].items():
        assert file_sha256(output / name) == expected, name
    for name, expected in plan["source_sha256"].items():
        assert file_sha256(Path(name)) == expected, name
    tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True)
    puzzles, all_calls = [], []
    for index, shard_path in enumerate(plan["shard_plans"]):
        shard = json.loads(Path(shard_path).read_text())
        parquet = pq.ParquetFile(shard["dataset"], pre_buffer=False)
        data = [row for i in range(parquet.num_row_groups)
                for row in parquet.read_row_group(i, use_threads=False).to_pylist()]
        random.Random(shard["seed"]).shuffle(data)
        trajectories = read_lines(output / f"shard-{index}/evaluation/trajectories.jsonl")
        calls = read_lines(output / f"shard-{index}/generations.jsonl")
        assert len(data) == len(trajectories) == shard["examples"]
        by_content = defaultdict(deque)
        for call in calls:
            request, response = call["request"], call["raw_response"]
            choice, = response["choices"]
            content = choice["message"]["content"]
            assert call["weights_sha256"] == result["model_manifest"]["weights_sha256"] == plan["weights_sha256"]
            assert response["model"] == request["model"] == plan["model"]
            usage = response["usage"]
            assert len(choice["token_ids"]) == usage["completion_tokens"] <= request["max_tokens"]
            assert len(response["prompt_token_ids"]) == usage["prompt_tokens"]
            assert usage["total_tokens"] == usage["completion_tokens"] + usage["prompt_tokens"]
            assert tokenizer.decode(choice["token_ids"], skip_special_tokens=True) == content
            local_ids = tokenizer.apply_chat_template(request["messages"], tokenize=True,
                                                       add_generation_prompt=True,
                                                       return_dict=False,
                                                       **request["chat_template_kwargs"])
            assert local_ids == response["prompt_token_ids"], "Server and local prompt tokenization differ"
            first_user = next(m["content"] for m in request["messages"] if m["role"] == "user")
            initial_fen = re.search(r"^FEN: (.+)$", first_user, re.M).group(1)
            by_content[(initial_fen, content)].append(call)
        shard_solved = 0
        for row, trajectory in zip(data, trajectories, strict=True):
            info = row["extra_info"]
            assert f"FEN: {info['start_fen']}\n" in trajectory["question"]
            board = chess.Board(info["start_fen"])
            solution = list(info["solution_moves"])
            pointer, format_retries, illegal_retries, tokens = 0, 0, 0, 0
            steps = []
            events = trajectory["harness_trace"]
            assert len(events) % 2 == 0
            terminal = False
            for assistant, verifier in zip(events[::2], events[1::2], strict=True):
                assert not terminal
                assert assistant["actor"] == "assistant" and verifier["actor"] == "verifier"
                call = by_content[(info["start_fen"], assistant["raw_response"])].popleft()
                request, response = call["request"], call["raw_response"]
                assert f"FEN: {board.fen()}\n" in request["messages"][-1]["content"]
                assert request["max_tokens"] == min(shard["policy_max_tokens"], shard["assistant_token_budget"] - tokens)
                parsed = parse_raw(assistant["raw_response"])
                assert parsed == assistant["parsed_action"]
                tokens += response["usage"]["completion_tokens"]
                assert tokens == assistant["assistant_tokens_used"]
                valid = bool(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", parsed))
                legal = valid and chess.Move.from_uci(parsed) in board.legal_moves
                correct = legal and parsed == solution[pointer]
                if not valid:
                    reason = "format_retry" if format_retries == 0 else "malformed"
                    terminal = format_retries > 0
                    format_retries += 1
                elif not legal:
                    reason = "illegal_retry" if illegal_retries == 0 else "illegal"
                    terminal = illegal_retries > 0
                    illegal_retries += 1
                elif not correct:
                    reason, terminal = "wrong_move", True
                else:
                    board.push_uci(parsed)
                    pointer += 1
                    if pointer < len(solution):
                        reply = chess.Move.from_uci(solution[pointer])
                        assert reply in board.legal_moves
                        board.push(reply)
                        pointer += 1
                    terminal = pointer >= len(solution)
                    reason = "solved" if terminal else "continue"
                assert verifier["stop_condition"] == reason, (verifier, reason)
                assert verifier["terminal"] == terminal
                choice = response["choices"][0]
                steps.append({"parsed_action": parsed, "legal_by_board": legal, "correct_move": correct,
                              "verifier_reason": reason, "output_tokens": response["usage"]["completion_tokens"],
                              "finish_reason": choice["finish_reason"],
                              "prompt_ids_sha256": fingerprint([response["prompt_token_ids"]])})
            solved = pointer >= len(solution)
            assert trajectory["correct"] == solved and trajectory["reward"] == float(solved)
            if not terminal:
                expected_stop = "assistant_token_budget" if tokens >= shard["assistant_token_budget"] else "max_turns"
                assert trajectory["stop_condition"] == expected_stop
            else:
                assert trajectory["stop_condition"] == steps[-1]["verifier_reason"]
            assert len(steps) == trajectory["turn_count"]
            puzzles.append({"puzzle_id": info["puzzle_id"], "shard": index, "solved": solved,
                            "output_tokens": tokens, "steps": steps, "stop_condition": trajectory["stop_condition"]})
            shard_solved += int(solved)
        assert not any(by_content.values()), "Unmatched model calls"
        assert result["summaries"][index]["solved_examples"] == shard_solved
        all_calls.extend(calls)
    ids = [p["puzzle_id"] for p in puzzles]
    assert len(ids) == len(set(ids)) == 8
    return {"kind": "independent_vllm_chess_audit", "role": "engineering", "status": "PASS",
            "job_id": result["slurm_job_id"], "examples": len(puzzles),
            "solved": sum(p["solved"] for p in puzzles), "model_calls": len(all_calls),
            "output_tokens": sum(p["output_tokens"] for p in puzzles),
            "length_stops": sum(c["raw_response"]["choices"][0]["finish_reason"] == "length" for c in all_calls),
            "terminal_reasons": dict(Counter(p["stop_condition"] for p in puzzles)), "puzzles": puzzles,
            "result_sha256": file_sha256(output / "result.json"),
            "audit_source_sha256": file_sha256(Path(__file__)),
            "training_steps": 0, "external_api_calls": 0,
            "limitations": plan["limitations"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_directory)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"puzzles", "limitations"}}))
