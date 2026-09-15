"""Audit native search accounting and independently replay its Chess verdicts.

Requires the baseline parsing/legality functions to remain unchanged. A new
parser is rejected until a separately reviewed replay rule is implemented.
No native runner, parser, verifier or generated harness is executed here.
"""
import argparse
import ast
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import random
import re

from .audit_chess_decode import parse_raw
from .visual_task import file_sha256


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def baseline_semantics(source, baseline):
    """Scope this independent replay to reviewed native action semantics."""
    def parts(text):
        tree = ast.parse(text)
        functions = {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id in {"FORMAT_RETRY_BUDGET", "ILLEGAL_MOVE_RETRY_BUDGET", "MAX_TURNS"}}
        return functions, constants
    candidate, caps = parts(source)
    original, _ = parts(baseline)
    for name in ("parse_action", "_uci_candidates", "is_legal_action", "_legal_moves_from_observation", "propose_action"):
        if candidate.get(name) != original[name]:
            raise ValueError(f"Independent replay requires review of changed function: {name}")
    return caps


def audit(output):
    import chess
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    result = json.loads((output / "result.json").read_text())
    plan = json.loads((output / "plan.json").read_text())
    assert result["status"] == "COMPLETED" and result["completed_search_iterations"] == 1
    assert result["role"] == plan["role"] == "engineering" and result["veto_mode"] == "off"
    assert file_sha256(output / "plan.json") == result["plan_sha256"]
    for name, expected in plan["source_sha256"].items():
        assert file_sha256(Path(name)) == expected, name
    target = plan["target_config"]
    tokenizer = AutoTokenizer.from_pretrained(target["model"], local_files_only=True)
    parquet = pq.ParquetFile(plan["dataset"], pre_buffer=False)
    data = [row for i in range(parquet.num_row_groups)
            for row in parquet.read_row_group(i, use_threads=False).to_pylist()]
    random.Random(plan["seed"]).shuffle(data)
    baseline = (output / "search/harnesses/h0/harness.py").read_text()
    arms = []
    for name in ("h0", "h1"):
        harness_path = output / f"search/harnesses/{name}/harness.py"
        caps = baseline_semantics(harness_path.read_text(), baseline)
        val_path, = (output / "search/logs").glob(f"*/{name}/*/val.json")
        val = json.loads(val_path.read_text())
        trajectories = lines(val_path.with_name("trajectories.jsonl"))
        calls = lines(output / f"generations-{name}.jsonl")
        assert len(data) == len(trajectories) == val["num_examples"] == 8
        assert [c["call"] for c in calls] == list(range(1, len(calls) + 1))
        by_response = defaultdict(deque)
        for call in calls:
            request, response = call["request"], call["raw_response"]
            choice, = response["choices"]
            content = choice["message"]["content"]
            usage = response["usage"]
            assert call["weights_sha256"] == result["model_manifest"]["weights_sha256"] == target["expected_weights_sha256"]
            assert response["model"] == request["model"] == target["model"]
            for setting in ("temperature", "top_p", "top_k", "seed", "chat_template_kwargs"):
                assert request[setting] == target[setting]
            assert len(choice["token_ids"]) == usage["completion_tokens"] <= request["max_tokens"]
            assert len(response["prompt_token_ids"]) == usage["prompt_tokens"]
            assert usage["total_tokens"] == usage["completion_tokens"] + usage["prompt_tokens"]
            assert tokenizer.decode(choice["token_ids"], skip_special_tokens=True) == content
            assert tokenizer.apply_chat_template(request["messages"], tokenize=True, return_dict=False,
                       add_generation_prompt=True, **request["chat_template_kwargs"]) == response["prompt_token_ids"]
            current_fen = re.search(r"^FEN: (.+)$", request["messages"][-1]["content"], re.M).group(1)
            by_response[(current_fen, content)].append(call)
        puzzles = []
        for row, trajectory in zip(data, trajectories, strict=True):
            info = row["extra_info"]
            assert trajectory["answer_redacted"] and trajectory["answer"] == ""
            assert f"FEN: {info['start_fen']}\n" in trajectory["question"]
            board, solution = chess.Board(info["start_fen"]), list(info["solution_moves"])
            pointer = formats = illegals = tokens = 0
            terminal, steps = False, []
            events = trajectory["harness_trace"]
            assert len(events) % 2 == 0
            for assistant, verifier in zip(events[::2], events[1::2], strict=True):
                assert not terminal
                assert assistant["actor"] == "assistant" and verifier["actor"] == "verifier"
                call = by_response[(board.fen(), assistant["raw_response"])].popleft()
                request, response = call["request"], call["raw_response"]
                assert f"FEN: {board.fen()}\n" in request["messages"][-1]["content"]
                assert request["max_tokens"] == 8129 - tokens
                parsed = parse_raw(assistant["raw_response"])
                assert parsed == assistant["parsed_action"]
                tokens += response["usage"]["completion_tokens"]
                assert tokens == assistant["assistant_tokens_used"] <= 8129
                valid = bool(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", parsed))
                legal = valid and chess.Move.from_uci(parsed) in board.legal_moves
                correct = legal and parsed == solution[pointer]
                if not valid:
                    terminal = formats >= min(10, max(0, caps.get("FORMAT_RETRY_BUDGET", 1)))
                    reason = "malformed" if terminal else "format_retry"
                    formats += 1
                elif not legal:
                    terminal = illegals >= min(10, max(0, caps.get("ILLEGAL_MOVE_RETRY_BUDGET", 1)))
                    reason = "illegal" if terminal else "illegal_retry"
                    illegals += 1
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
                assert verifier["stop_condition"] == reason and verifier["terminal"] == terminal
                steps.append({"action": parsed, "legal": legal, "correct": correct, "reason": reason})
            solved = pointer >= len(solution)
            assert trajectory["correct"] == solved and trajectory["reward"] == float(solved)
            stop = steps[-1]["reason"] if terminal else "assistant_token_budget" if tokens >= 8129 else "max_policy_calls"
            assert trajectory["stop_condition"] == stop
            assert trajectory["turn_count"] == len(steps)
            puzzles.append({"puzzle_id": info["puzzle_id"], "solved": solved, "tokens": tokens,
                            "steps": steps, "stop": stop})
        assert not any(by_response.values()), "Unmatched model calls"
        solved = sum(p["solved"] for p in puzzles)
        assert val["solved_examples"] == solved and val["success_rate"] == solved / 8
        assert val["mean_turn_count"] == len(calls) / 8
        arms.append({"harness": name, "harness_sha256": file_sha256(harness_path), "solved": solved,
                     "examples": 8, "calls": len(calls), "output_tokens": sum(p["tokens"] for p in puzzles),
                     "length_stops": sum(c["raw_response"]["choices"][0]["finish_reason"] == "length" for c in calls),
                     "terminal_reasons": dict(Counter(p["stop"] for p in puzzles)), "puzzles": puzzles})
    best_objective = min((-a["solved"], a["calls"]) for a in arms)
    tied_best = [a["harness"] for a in arms if (-a["solved"], a["calls"]) == best_objective]
    accepted = (output / "search/logs/accepted_harness.txt").read_text().strip()
    frontier = json.loads((output / "search/logs/frontier_val.json").read_text())
    assert accepted == result["accepted"] == frontier["_best"]["harness"] and accepted in tied_best
    return {"kind": "independent_native_search_audit", "role": "engineering", "status": "PASS",
            "job_id": result["slurm_job_id"], "accepted": accepted, "arms": arms,
            "tied_best": tied_best, "strict_score_improvement": arms[1]["solved"] > arms[0]["solved"],
            "selection_audit_scope": "Objective optimality and membership in the tied best set; native stable sorting inherits filesystem enumeration order, which was not frozen",
            "result_sha256": file_sha256(output / "result.json"), "audit_source_sha256": file_sha256(Path(__file__)),
            "artifact_sha256": {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*"))
                               if p.is_file() and ".cli-state" not in p.parts},
            "limitations": plan["limitations"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_directory)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"arms", "artifact_sha256", "limitations"}}))
