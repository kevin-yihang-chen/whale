"""One recorded native Meta-Harness iteration with GLM and a local target.

Shared candidate-generation/evaluation infrastructure, with VETO disabled.
The pinned search loop, verifier, frontier and acceptance rule remain native.
The proposer sees a disposable copy of the answer-redacted search archive;
only the declared candidate, metadata and report are imported back.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import patch

from .glm_gateway import BudgetJournal, MODEL, relay
from .local_completion import checkpoint_manifest
from .preflight import PIN
from .visual_task import file_sha256
from .vllm_runtime import RecordedVLLMClient, VLLMConfiguration, json_request


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def tree_hashes(root, *, ignore_cli_state=False):
    result = {}
    for path in sorted(root.rglob("*")):
        if ignore_cli_state and path.relative_to(root).parts[0] == ".cli-state":
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not accepted in the search archive: {path.name}")
        if path.is_file():
            result[str(path.relative_to(root))] = file_sha256(path)
    return result


def check_proposal(view, before):
    """Reject archive edits, traversal and extra candidates before native loading."""
    after = tree_hashes(view, ignore_cli_state=True)
    allowed = {"harnesses/h1/harness.py", "pending_eval.json", "logs/iteration_001/report.md"}
    for name, digest in before.items():
        if after.get(name) != digest:
            raise ValueError(f"Proposer changed its evidence archive: {name}")
    for name in after.keys() - before.keys():
        if name not in allowed and not name.startswith((".cli-state/", "logs/claude_sessions/")):
            raise ValueError(f"Unexpected proposer artifact: {name}")
    candidate = view / "harnesses/h1/harness.py"
    if not candidate.is_file() or candidate.stat().st_size > 65536:
        raise ValueError("Expected one bounded h1 candidate")
    pending = view / "pending_eval.json"
    if pending.exists():
        if pending.stat().st_size > 16384:
            raise ValueError("Oversized proposer metadata")
        normalize_metadata(json.loads(pending.read_text()))
    # Missing metadata remains recoverable by the unchanged native loop.
    return [name for name in allowed if (view / name).is_file()]


def normalize_metadata(payload):
    """Accept the unambiguous slot alias; never change code, scores or scope."""
    entries = payload.get("candidates")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ValueError("Exactly the allocated h1 slot is required")
    entry = entries[0]
    names = [entry[k] for k in ("name", "slot") if k in entry]
    if not names or any(name != "h1" for name in names):
        raise ValueError("Exactly the allocated h1 slot is required")
    if entry.get("path", "harnesses/h1/harness.py") != "harnesses/h1/harness.py":
        raise ValueError("Unexpected candidate path")
    if entry.get("parent", "h0") != "h0":
        raise ValueError("Unexpected candidate parent")
    return dict(payload, candidates=[dict(entry, name="h1")])


def require_complete_sweep(rows, harnesses, logs_dir):
    """Native run_sweep catches exceptions; never let those become scored zeros."""
    if len(rows) != len(harnesses) or not all(ok for _, ok in rows):
        raise RuntimeError("Native evaluation failed; inspect preserved error.txt")
    for name, _ in harnesses:
        files = list(logs_dir.glob(f"*/{name}/*/val.json"))
        if len(files) != 1:
            raise RuntimeError("Expected exactly one completed evaluation per candidate")
        val = json.loads(files[0].read_text())
        trajectories = files[0].with_name("trajectories.jsonl").read_text().splitlines()
        if val["num_examples"] != 8 or len(trajectories) != 8:
            raise RuntimeError("Incomplete engineering coverage")
    return rows


def check_plan(path):
    plan = json.loads(path.read_text())
    if plan["kind"] != "native_meta_harness_plan" or plan["role"] != "engineering":
        raise ValueError("Expected a frozen native engineering search plan")
    upstream = Path("upstream/WHALE")
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("Wrong upstream revision")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("Modified upstream source")
    for name, expected in plan["source_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Frozen source differs: {name}")
    if {name: version(name) for name in plan["versions"]} != plan["versions"]:
        raise ValueError("Runtime versions differ")
    if plan["proposer_model"] != MODEL or plan["iterations"] != 1 or plan["proposals_per_iter"] != 1:
        raise ValueError("This driver permits one GLM candidate and one iteration")
    if plan["veto_mode"] != "off" or plan["examples"] != 8 or plan["seed"] != 42:
        raise ValueError("Unexpected engineering scope")
    if not 1 <= plan["max_api_requests"] <= 12 or not 1 <= plan["max_cli_turns"] <= 12:
        raise ValueError("Invalid proposer call cap")
    if not 1 <= plan["propose_timeout_seconds"] <= 300:
        raise ValueError("Invalid proposer deadline")
    from vllm.config import SchedulerConfig
    options = plan["server_arguments"]
    def option(name):
        return int(options[options.index(name) + 1])
    if "--enable-chunked-prefill" not in options:
        raise ValueError("Qwen3.5 requires chunked prefill in this runtime")
    SchedulerConfig(max_model_len=option("--max-model-len"), is_encoder_decoder=False,
                    max_num_batched_tokens=option("--max-num-batched-tokens"),
                    max_num_seqs=option("--max-num-seqs"), enable_chunked_prefill=True,
                    is_multimodal_model=True)
    VLLMConfiguration(**plan["target_config"], base_url="http://127.0.0.1:1")
    if plan.get("baseline_cache"):
        validate_baseline_cache(plan)
    if plan.get("prepared_proposal"):
        validate_prepared_proposal(plan)
    return plan


def validate_baseline_cache(plan):
    """Bind reuse to complete h0 evidence, identical data/sampling and weights."""
    cache = plan["baseline_cache"]
    root = Path(cache["directory"])
    for name, digest in json.loads(Path(cache["artifact_manifest"]).read_text()).items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or file_sha256(path) != digest:
            raise ValueError("Changed cached baseline evidence")
    previous = json.loads((root / "plan.json").read_text())
    for key in ("dataset", "examples", "seed", "target_config", "server_arguments", "versions", "veto_mode"):
        if previous[key] != plan[key]:
            raise ValueError(f"Cached h0 configuration differs: {key}")
    baseline = root / "search/harnesses/h0/harness.py"
    original = Path("upstream/WHALE/domains/chess_puzzles/environments/chess_puzzle/base_harness.py")
    if file_sha256(baseline) != file_sha256(original):
        raise ValueError("Cached h0 is not the unchanged baseline")
    value, = (root / "search/logs").glob("*/h0/*/val.json")
    summary = json.loads(value.read_text())
    if summary["metadata"]["seed"] != 42 or summary["metadata"]["assistant_token_budget"] != 8129:
        raise ValueError("Cached h0 budget/seed mismatch")
    require_complete_sweep([("cached h0", True)], [("h0", baseline)], root / "search/logs")
    return root, summary


def restore_baseline(plan, output):
    root, summary = validate_baseline_cache(plan)
    for path in (root / "search/logs").glob("*/h0"):
        shutil.copytree(path, output / "search/logs" / path.relative_to(root / "search/logs"))
    shutil.copytree(root / "search/harnesses/h0", output / "search/harnesses/h0")
    shutil.copyfile(root / "generations-h0.jsonl", output / "generations-h0.jsonl")
    return {"harness": "h0", "summary": summary, "reused_from": str(root)}


def validate_prepared_proposal(plan):
    prepared = plan["prepared_proposal"]
    root = Path(prepared["directory"])
    for name, digest in json.loads(Path(prepared["artifact_manifest"]).read_text()).items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or file_sha256(path) != digest:
            raise ValueError("Changed prepared proposal evidence")
    previous = json.loads((root / "plan.json").read_text())
    for key in ("dataset", "examples", "seed", "target_config", "server_arguments", "versions", "veto_mode"):
        if previous[key] != plan[key]:
            raise ValueError(f"Prepared proposal context differs: {key}")
    if json.loads((root / "proposer-result.json").read_text())["exit_code"] != 0:
        raise ValueError("Prepared proposer did not complete")
    view = root / "proposer-workspace"
    allowed = check_proposal(view, tree_hashes(root / "search"))
    return root, view, allowed


def import_prepared_proposal(native, plan, output, **kwargs):
    if kwargs["next_names"] != ["h1"] or kwargs["iteration"] != 1:
        raise ValueError("Unexpected prepared proposal allocation")
    root, view, allowed = validate_prepared_proposal(plan)
    prompt = root / "search/logs/iteration_001/proposer_prompt.txt"
    if prompt.read_text() != kwargs["task_prompt"]:
        raise ValueError("Prepared native task prompt differs")
    for name in allowed:
        destination = kwargs["run_dir"] / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == "pending_eval.json":
            write_json(destination, normalize_metadata(json.loads((view / name).read_text())))
        else:
            destination.write_bytes((view / name).read_bytes())
    sessions = view / "logs/claude_sessions"
    shutil.copytree(sessions, kwargs["run_dir"] / "logs/claude_sessions", dirs_exist_ok=True)
    session, = sessions.iterdir()
    meta = json.loads((session / "meta.json").read_text())
    result = native.claude_wrapper.parse_stream_events((session / "events.jsonl").read_text(),
        kwargs["task_prompt"], MODEL, meta["duration_seconds"], meta["exit_code"], cwd=meta["cwd"])
    write_json(output / "proposal-integrity.json", {"status": "PASS", "imported": allowed,
               "prepared_from": str(root), "new_api_calls": 0, "normalization": "slot h1 -> name h1 only",
               "candidate_sha256": file_sha256(view / "harnesses/h1/harness.py")})
    return result


@contextmanager
def target_service(plan, output):
    """Keep one frozen checkpoint loaded throughout h0 and h1 evaluation."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model",
               plan["target_config"]["model"], "--host", "127.0.0.1", "--port", str(port),
               *plan["server_arguments"]]
    write_json(output / "server-command.json", command)
    with (output / "server.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + plan["startup_timeout_seconds"]
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM startup exited: {process.returncode}")
                try:
                    models = json_request(endpoint + "/v1/models", timeout=2)
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("vLLM startup limit exceeded")
                    time.sleep(2)
            if [m["id"] for m in models["data"]] != [plan["target_config"]["model"]]:
                raise ValueError("Unexpected served model")
            write_json(output / "served-models.json", models)
            yield endpoint
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def isolated_proposal(native, plan, output, journal, **kwargs):
    if kwargs.pop("next_names") != ["h1"] or kwargs["iteration"] != 1:
        raise ValueError("Unexpected proposer allocation")
    original_run = kwargs["run_dir"]
    view = output / "proposer-workspace"
    shutil.copytree(original_run, view)
    before = tree_hashes(view)
    state = view / ".cli-state"
    for name in ("config", "tmp", "empty_plugins"):
        (state / name).mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    binary = root / "data/claude-runtime-2.1.236/claude"
    wrapper = native.claude_wrapper
    original_build = wrapper.build_command

    def bounded_command(*args, **options):
        command = original_build(*args, **options)
        command.remove("--dangerously-skip-permissions")
        command[0] = str(binary)
        command += ["--bare", "--no-session-persistence", "--max-turns", str(plan["max_cli_turns"]),
                    "--permission-mode", "dontAsk", "--debug-file", str(state / "debug.log")]
        return [sys.executable, str(root / "ours/confined_exec.py"),
                "--workspace", str(view), "--", *command]

    capture = output / "raw-sse"
    capture.mkdir()
    with relay(journal, "/userhome/cs3/yihangc/.config/whale-delta/credentials.json",
               max_requests=plan["max_api_requests"], capture_directory=capture) as gateway:
        env = {k: v for k, v in os.environ.items()
               if not k.endswith(("API_KEY", "AUTH_TOKEN")) and k not in
               {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"}}
        env.update(ANTHROPIC_BASE_URL=gateway["base_url"], ANTHROPIC_API_KEY=gateway["token"],
                   ANTHROPIC_DEFAULT_OPUS_MODEL=MODEL, ANTHROPIC_DEFAULT_SONNET_MODEL=MODEL,
                   ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL, CLAUDE_CONFIG_DIR=str(state / "config"),
                   TMPDIR=str(state / "tmp"), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                   DISABLE_AUTOUPDATER="1", DISABLE_UPDATES="1", CLAUDE_CODE_MAX_OUTPUT_TOKENS="4096",
                   CLAUDE_SETTING_SOURCES="project")
        # Claude/Bun honors inherited HTTP proxies, unlike the target client.
        # Always bypass proxies for the node-local budget relay.
        bypass = ",".join(filter(None, [env.get("NO_PROXY", env.get("no_proxy", "")), "127.0.0.1", "localhost"]))
        env.update(NO_PROXY=bypass, no_proxy=bypass)
        with patch.dict(os.environ, env, clear=True), patch.object(wrapper, "build_command", bounded_command), \
                patch.object(wrapper, "_EMPTY_PLUGIN_DIR", state / "empty_plugins"):
            result = native.propose_claude(**dict(kwargs, run_dir=view))
    write_json(output / "proposer-result.json", {"exit_code": result.exit_code,
               "duration_seconds": result.duration_seconds, "budget": journal.summary(),
               "cli_usd_estimate_is_provider_bill": False})
    allowed = check_proposal(view, before)
    # Original evidence stays outside the child's allowed filesystem. Only these
    # explicit artifacts can cross the boundary; logs are copied separately.
    for name in allowed:
        destination = original_run / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == "pending_eval.json":
            write_json(destination, normalize_metadata(json.loads((view / name).read_text())))
        else:
            destination.write_bytes((view / name).read_bytes())
    shutil.copytree(view / "logs/claude_sessions", original_run / "logs/claude_sessions", dirs_exist_ok=True)
    write_json(output / "proposal-integrity.json", {"status": "PASS", "imported": allowed,
               "protected_files": len(before), "landlock_scope": "disposable archive only"})
    return result


def run(args):
    plan = check_plan(args.plan)
    if args.check_only:
        print(json.dumps({"status": "PRE_SUBMISSION_PASS", "plan_sha256": file_sha256(args.plan)}))
        return
    if args.output is None:
        raise ValueError("--output is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(args.plan.read_bytes())
    import torch
    from autoharness_chess_puzzle import runner
    from meta_harness import chess_puzzle_benchmark as benchmark
    from meta_harness import meta_harness_chess_puzzle as native
    if not torch.cuda.is_available():
        raise RuntimeError("An allocated CUDA GPU is required")
    identity = checkpoint_manifest(Path(plan["target_config"]["model"]))
    if identity["weights_sha256"] != plan["target_config"]["expected_weights_sha256"]:
        raise ValueError("Checkpoint bytes changed")
    journal = BudgetJournal(Path("data/glm-budget/ledger.jsonl"))
    clients = []
    evaluations = []
    if plan.get("baseline_cache"):
        evaluations.append(restore_baseline(plan, output))
    started = time.perf_counter()
    report = {"kind": "native_meta_harness_execution", "role": "engineering", "status": "FAILED",
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "model_manifest": identity,
              "veto_mode": "off", "training_steps": 0, "completed_search_iterations": 0,
              "plan_sha256": file_sha256(args.plan), "limitations": plan["limitations"]}
    report["baseline_cache"] = plan.get("baseline_cache")
    report["prepared_proposal"] = plan.get("prepared_proposal")
    old_evaluate = benchmark.evaluate_harness
    old_sweep = native.run_sweep

    class Client(RecordedVLLMClient):
        def __init__(self, config):
            super().__init__(config)
            clients.append(self)

    def recorded_evaluation(**kwargs):
        name = Path(kwargs["harness_path"]).parent.name
        config = dict(kwargs["llm_config"], trace_path=str(output / f"generations-{name}.jsonl"))
        value = old_evaluate(**dict(kwargs, llm_config=config))
        clients[-1].close()
        evaluations.append({"harness": name, "summary": value})
        return value

    def checked_sweep(config, harnesses, logs_dir, **kwargs):
        return require_complete_sweep(old_sweep(config, harnesses, logs_dir, **kwargs), harnesses, logs_dir)

    env = {"CHESS_PUZZLE_DEFAULT_MAX_TURNS": "9", "CHESS_PUZZLE_MAX_TURNS_CAP": "18",
           "CHESS_PUZZLE_FORMAT_RETRIES": "1", "CHESS_PUZZLE_ILLEGAL_RETRIES": "1",
           "CHESS_PUZZLE_PROPOSER_RETRIES": "1", "CHESS_PUZZLE_PROPOSER_RETRY_BACKOFF_S": "0",
           "BASELINE_HARNESS_OVERRIDE": ""}
    try:
        with patch.dict(os.environ, env), target_service(plan, output) as endpoint:
            config = {"task_profiles": {"engineering": {"dataset_path": plan["dataset"], "limit": 8}},
                      "models": [dict(plan["target_config"], base_url=endpoint)], "seeds": [42],
                      "eval": {"assistant_token_budget": 8129, "policy_max_tokens": 8129}}
            config_path = output / "native-config.json"
            write_json(config_path, config)
            native_args = argparse.Namespace(run_name="search", config=str(config_path), iterations=1,
                proposals_per_iter=1, proposer_model=MODEL, proposer_effort="low",
                propose_timeout=plan["propose_timeout_seconds"], early_stop_success_rate=1.0,
                fresh=False, force=not bool(plan.get("baseline_cache")), start_iteration=1, early_stop_min_iters=0,
                early_stop_patience=2, eval_only=False, use_api_key=True, prompt_only=False)
            with patch.object(runner, "LLMClient", Client), patch.object(runner, "LLMConfig", VLLMConfiguration), \
                    patch.object(benchmark, "evaluate_harness", recorded_evaluation), \
                    patch.object(native, "run_sweep", checked_sweep), \
                    patch.object(native, "RUNS_DIR", output), patch.object(native, "PROMPT_ONLY", False), \
                    patch.object(native, "propose_claude_with_retries",
                                 lambda **kw: import_prepared_proposal(native, plan, output, **kw)
                                 if plan.get("prepared_proposal") else isolated_proposal(native, plan, output, journal, **kw)):
                native.run_evolve(native_args)
            if [v["harness"] for v in evaluations] != ["h0", "h1"]:
                raise RuntimeError("Incomplete native iteration")
            report.update(status="COMPLETED", completed_search_iterations=1,
                          accepted=(output / "search/logs/accepted_harness.txt").read_text().strip())
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        raise
    finally:
        for client in clients:
            client.close()
        report.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      seconds=time.perf_counter() - started, evaluations=evaluations, budget=journal.summary())
        write_json(output / "result.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args())
