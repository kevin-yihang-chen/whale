"""Run the pinned WHALE Chess evaluator with the declared local HF client.

Shared baseline runtime validation, not an optimized baseline or Method result.
An explicit pre-run plan binds source, dataset, weights and reduced decode budget.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time

from .local_completion import LLMClient
from .preflight import PIN
from .visual_task import file_sha256


def run(args):
    plan = json.loads(args.plan.read_text())
    if plan.get("kind") != "chess_runtime_prerun_plan" or plan.get("role") != "engineering":
        raise ValueError("Expected a frozen Chess engineering run plan")
    if type(plan["examples"]) is not int or not 1 <= plan["examples"] <= 8:
        raise ValueError("This engineering run accepts one to eight examples")
    upstream = Path("upstream/WHALE")
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if head != PIN or dirty:
        raise ValueError("Pinned upstream source must be unchanged")
    for path, sha in plan["source_sha256"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError(f"Frozen source changed: {path}")
    dataset_manifest = json.loads(Path(plan["dataset_manifest"]).read_text())
    if dataset_manifest["role"] != "engineering" or dataset_manifest["examples"] != plan["examples"]:
        raise ValueError("Prepared engineering subset size differs from the run plan")
    if file_sha256(Path(plan["dataset"])) != dataset_manifest["parquet_sha256"]:
        raise ValueError("Dataset bytes changed after preparation")
    processor_check = json.loads(Path(plan["processor_preflight"]).read_text())
    if processor_check.get("status") != "PASS" or len(processor_check.get("requests", [])) != plan["examples"]:
        raise ValueError("A successful processor preflight covering the entire shard is required")
    if args.check_only:
        print(json.dumps({"status": "PRE_SUBMISSION_PASS", "plan_sha256": file_sha256(args.plan)}))
        return
    if args.output is None:
        raise ValueError("--output is required for GPU execution")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_bytes(args.plan.read_bytes())
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        os.environ.pop(key, None)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      CHESS_PUZZLE_DEFAULT_MAX_TURNS="9", CHESS_PUZZLE_MAX_TURNS_CAP="18",
                      CHESS_PUZZLE_FORMAT_RETRIES="1", CHESS_PUZZLE_ILLEGAL_RETRIES="1")
    import torch
    from autoharness_chess_puzzle import runner

    torch.set_num_threads(4)
    if not torch.cuda.is_available() or runner.LLMClient is not LLMClient:
        raise RuntimeError("Expected an allocated GPU and the explicit local compatibility namespace")
    clients = []

    class RecordedLocalClient(LLMClient):
        def __init__(self, config):
            super().__init__(config)
            clients.append(self)

    llm_config = dict(plan["llm_config"])
    llm_config["trace_path"] = str(args.output / "generations.jsonl")
    start = time.perf_counter()
    original_client = runner.LLMClient
    runner.LLMClient = RecordedLocalClient
    try:
        summary = runner.evaluate_harness(
            harness_path=plan["harness"], dataset_path=plan["dataset"], llm_config=llm_config,
            output_dir=args.output / "evaluation", limit=plan["examples"], seed=plan["seed"],
            assistant_token_budget=plan["assistant_token_budget"],
            policy_max_tokens=plan["policy_max_tokens"])
    finally:
        runner.LLMClient = original_client
        for client in clients:
            client.close()
    if len(clients) != 1:
        raise RuntimeError("Expected exactly one local target instance")
    rows = [json.loads(line) for line in (args.output / "generations.jsonl").read_text().splitlines()]
    result = {"kind": "original_chess_runner_local_client_smoke", "role": "engineering",
              "status": "COMPLETED", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "upstream_commit": head,
              "plan_sha256": file_sha256(args.plan), "source_sha256": plan["source_sha256"],
              "model_manifest": clients[0].manifest, "llm_config": llm_config,
              "dataset_manifest": dataset_manifest, "summary": summary, "generations": len(rows),
              "output_tokens": sum(row["usage"]["completion_tokens"] for row in rows),
              "generation_seconds": sum(row["seconds"] for row in rows),
              "length_stops": sum(row["finish"] == "length" for row in rows),
              "total_seconds": time.perf_counter() - start, "gpu": torch.cuda.get_device_name(),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "versions": {name: version(name) for name in ("torch", "transformers", "chess", "pandas", "pyarrow")},
              "artifacts_sha256": {str(path.relative_to(args.output)): file_sha256(path)
                                   for path in args.output.rglob("*") if path.is_file()},
              "limitations": plan["limitations"], "training_steps": 0, "proposer_calls": 0}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "generations": len(rows), "result": str(args.output / "result.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args())
