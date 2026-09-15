"""Independently audit the completed, single-call Chess decoding diagnostic.

The audit deliberately rejects multi-call runs: no multi-turn verification is
claimed by this diagnostic. It does not import the upstream parser or verifier.
"""
from collections import Counter
import json
from pathlib import Path
import random
import re

from .visual_task import file_sha256


def parse_raw(text):
    move = r"([a-h][1-8][a-h][1-8][qrbn]?)"
    for pattern in (rf"<move>\s*{move}\s*</move>", rf"\[\s*{move}\s*\]", rf"\b{move}\b"):
        found = {m.lower() for m in re.findall(pattern, text, re.I)}
        if len(found) == 1:
            return found.pop()
        if found:
            return "__ambiguous_multiple_moves__"
    return "__no_move__"


def audit():
    import chess
    import pandas as pd
    from transformers import AutoTokenizer

    study_path = Path("results/chess-decode-study-plan-20260909.json")
    study = json.loads(study_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained("data/models/qwen3.5-2b-15852e8", local_files_only=True)
    by_puzzle = []
    artifact_hashes = {}
    for index, shard in enumerate(study["shards"]):
        output = Path(f"data/chess-decode-221597/shard-{index}")
        plan = json.loads((output / "plan.json").read_text())
        result = json.loads((output / "result.json").read_text())
        assert file_sha256(output / "plan.json") == shard["sha256"] == result["plan_sha256"]
        assert result["status"] == "COMPLETED"
        assert result["source_sha256"] == plan["source_sha256"]
        for path, expected in result["artifacts_sha256"].items():
            assert file_sha256(output / path) == expected
        for path, expected in plan["source_sha256"].items():
            assert file_sha256(Path(path)) == expected
        manifest = result["dataset_manifest"]
        assert manifest["parent_manifest_sha256"] == study["parent_manifest_sha256"]
        assert file_sha256(Path(plan["dataset"])) == manifest["parquet_sha256"]
        data = pd.read_parquet(plan["dataset"]).to_dict("records")
        random.Random(plan["seed"]).shuffle(data)
        traces = [json.loads(line) for line in (output / "generations.jsonl").read_text().splitlines()]
        native = [json.loads(line) for line in (output / "evaluation/trajectories.jsonl").read_text().splitlines()]
        preflight = json.loads(Path(plan["processor_preflight"]).read_text())["requests"]
        assert len(traces) == len(native) == len(data) == len(preflight) == shard["examples"]
        for row, trace, trajectory, cpu in zip(data, traces, native, preflight, strict=True):
            info = row["extra_info"]
            assert cpu["example_id"] == info["puzzle_id"]
            assert f"FEN: {info['start_fen']}\n" in trajectory["question"]
            assert trajectory["turn_count"] == 1
            assert trace["input_ids_sha256"] == cpu["input_ids_sha256"]
            assert trace["weights_sha256"] == result["model_manifest"]["weights_sha256"]
            assert tokenizer.decode(trace["output_ids"], skip_special_tokens=True) == trace["raw_answer"]
            assert len(trace["output_ids"]) == trace["usage"]["completion_tokens"] == 8129
            assert trace["finish"] == "length"
            parsed = parse_raw(trace["raw_answer"])
            board = chess.Board(info["start_fen"])
            legal = bool(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", parsed)) and chess.Move.from_uci(parsed) in board.legal_moves
            correct_first = legal and parsed == info["solution_moves"][0]
            assert not correct_first  # This diagnosis has no successful continuation to replay.
            kind = "wrong_move" if legal else "illegal_retry" if not parsed.startswith("__") else "format_retry"
            events = trajectory["harness_trace"]
            assert len(events) == 2 and events[0]["actor"] == "assistant" and events[1]["actor"] == "verifier"
            assert events[0]["raw_response"] == trace["raw_answer"]
            assert events[0]["parsed_action"] == parsed and events[1]["stop_condition"] == kind
            assert trajectory["stop_condition"] == ("wrong_move" if legal else "assistant_token_budget")
            assert trajectory["reward"] == 0 and trajectory["correct"] is False
            by_puzzle.append({"puzzle_id": info["puzzle_id"], "shard": index, "parsed_action": parsed,
                              "legal_by_board": legal, "correct_first_move": correct_first,
                              "verifier_step": kind, "terminal_reason": trajectory["stop_condition"],
                              "output_tokens": len(trace["output_ids"]), "finish": trace["finish"]})
        artifact_hashes.update({str(p): file_sha256(p) for p in output.rglob("*") if p.is_file()})
    planned_ids = [p for shard in study["shards"] for p in shard["puzzle_ids"]]
    actual_ids = [r["puzzle_id"] for r in by_puzzle]
    assert len(actual_ids) == len(set(actual_ids)) == 8 and sorted(actual_ids) == sorted(planned_ids)
    return {"kind": "independent_chess_decoding_audit", "status": "PASS", "role": "engineering",
            "job_id": "221597", "examples": 8, "solved": 0, "correct_first_moves": 0,
            "legal_first_moves": sum(r["legal_by_board"] for r in by_puzzle),
            "output_tokens": sum(r["output_tokens"] for r in by_puzzle), "length_stops": 8,
            "verifier_steps": dict(Counter(r["verifier_step"] for r in by_puzzle)),
            "puzzles": by_puzzle, "artifact_sha256": artifact_hashes,
            "study_plan_sha256": file_sha256(study_path), "audit_source_sha256": file_sha256(Path(__file__)),
            "training_steps": 0, "proposer_calls_in_this_job": 0,
            "limitations": ["All outputs exhausted their generation budget; no successful RSFT samples",
                            "Same 8 engineering inputs, two disjoint shards; not two research seeds",
                            "HF local 2B/backend variant; not the released 4B/served-model baseline",
                            "Native legal_rate checks terminal illegal only and reports 1.0 despite malformed/illegal-retry outputs",
                            "Not a controlled comparison of thinking, no VETO result, no multi-turn audit"]}


if __name__ == "__main__":
    result = audit()
    Path("results/chess-decode-221597-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"artifact_sha256", "puzzles", "limitations"}}))
