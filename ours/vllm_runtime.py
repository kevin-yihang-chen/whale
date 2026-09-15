"""Bounded, recorded vLLM execution of the unchanged WHALE Chess runner.

Shared runtime validation for E4, not an additional method contribution.
The local HTTP endpoint keeps reasoning in content, matching the released
server configuration. No external API, parser repair, or generation retry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import urlparse

from .local_completion import LLMConfig, CompletionResponse, checkpoint_manifest
from .preflight import PIN
from .visual_task import file_sha256


@dataclass
class VLLMConfiguration(LLMConfig):
    provider: str = "local-vllm"
    base_url: str = ""
    request_timeout_seconds: int = 420

    def __post_init__(self):
        fields = asdict(self)
        timeout = fields.pop("request_timeout_seconds")
        if type(timeout) is not int or not 0 < timeout <= 420:
            raise ValueError("Expected a bounded request timeout")
        if type(self.batch_concurrency) is not int or not 1 <= self.batch_concurrency <= 4:
            raise ValueError("At most four concurrent engineering requests are supported")
        fields["batch_concurrency"] = 1
        endpoint = urlparse(fields.pop("base_url"))
        if self.provider != "local-vllm" or endpoint.hostname != "127.0.0.1" or endpoint.scheme != "http":
            raise ValueError("A local vLLM endpoint is required")
        if endpoint.path or endpoint.query or endpoint.fragment or endpoint.username or endpoint.password:
            raise ValueError("Expected a plain loopback origin")
        fields["provider"] = "local-transformers"
        LLMConfig(**fields)  # Reuse the validated common sampling contract.


def json_request(url, payload=None, timeout=180):
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    # Node-local traffic must not enter a configured external proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read())


class RecordedVLLMClient:
    """Preserve native content and record exact server-reported token IDs."""

    def __init__(self, config):
        self.config = config
        self.calls = 0
        self.lock = threading.Lock()
        self.ledger = Path(config.trace_path).open("x")

    def complete_response(self, messages, *, max_tokens=None):
        limit = self.config.max_tokens if max_tokens is None else max_tokens
        if type(limit) is not int or not 0 < limit <= self.config.max_tokens:
            raise ValueError("Invalid per-call token allocation")
        payload = {"model": self.config.model, "messages": [asdict(m) for m in messages],
                   "temperature": self.config.temperature, "top_p": self.config.top_p,
                   "top_k": self.config.top_k, "max_tokens": limit,
                   "chat_template_kwargs": self.config.chat_template_kwargs,
                   "seed": self.config.seed, "return_token_ids": True}
        started = time.perf_counter()
        body = json_request(self.config.base_url + "/v1/chat/completions", payload,
                            timeout=self.config.request_timeout_seconds)
        elapsed = time.perf_counter() - started
        with self.lock:
            choice, = body["choices"]
            content = choice["message"]["content"]
            usage = body["usage"]
            record = {"call": self.calls + 1, "request": payload, "raw_response": body,
                      "seconds": elapsed, "weights_sha256": self.config.expected_weights_sha256}
            # Preserve even an invalid response before raising.
            self.ledger.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.ledger.flush()
            if not isinstance(content, str) or body["model"] != self.config.model:
                raise ValueError("Unexpected response identity or content")
            if choice["message"].get("reasoning") or choice["message"].get("reasoning_content"):
                raise ValueError("The frozen plan does not separate reasoning from content")
            if len(choice["token_ids"]) != usage["completion_tokens"] or not 0 < usage["completion_tokens"] <= limit:
                raise ValueError("Output token evidence disagrees with usage")
            if len(body["prompt_token_ids"]) != usage["prompt_tokens"]:
                raise ValueError("Prompt token evidence disagrees with usage")
            self.calls += 1
            return CompletionResponse(content=content, usage=usage)

    def close(self):
        self.ledger.close()


def check_plan(path):
    plan = json.loads(path.read_text())
    if plan["kind"] != "chess_vllm_runtime_plan" or plan["role"] != "engineering":
        raise ValueError("A frozen engineering plan is required")
    upstream = Path("upstream/WHALE")
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("Wrong upstream revision")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("Modified upstream source")
    for source, expected in plan["source_sha256"].items():
        if file_sha256(Path(source)) != expected:
            raise ValueError(f"Frozen source differs: {source}")
    total = 0
    for shard_path in plan["shard_plans"]:
        shard = json.loads(Path(shard_path).read_text())
        manifest = json.loads(Path(shard["dataset_manifest"]).read_text())
        if file_sha256(Path(shard["dataset"])) != manifest["parquet_sha256"]:
            raise ValueError("Changed dataset")
        if shard["examples"] != manifest["examples"] or manifest["role"] != "engineering":
            raise ValueError("Wrong dataset role or coverage")
        total += shard["examples"]
    if total != 8:
        raise ValueError("This diagnosis is restricted to the existing eight engineering inputs")
    actual = {name: version(name) for name in plan["versions"]}
    if actual != plan["versions"]:
        raise ValueError(f"Runtime versions differ: {actual}")
    from vllm.config import SchedulerConfig

    options = plan["server_arguments"]
    def option(name):
        return int(options[options.index(name) + 1])
    chunked = "--enable-chunked-prefill" in options
    SchedulerConfig(max_model_len=option("--max-model-len"), is_encoder_decoder=False,
                    max_num_batched_tokens=option("--max-num-batched-tokens"),
                    max_num_seqs=option("--max-num-seqs"), enable_chunked_prefill=chunked,
                    is_multimodal_model=True)
    if not chunked:
        raise ValueError("This Qwen3.5/vLLM runtime requires supported chunked prefill")
    return plan


def run(args):
    plan = check_plan(args.plan)
    if args.check_only:
        print(json.dumps({"status": "PRE_SUBMISSION_PASS", "plan_sha256": file_sha256(args.plan)}))
        return
    if args.output is None:
        raise ValueError("--output is required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_bytes(args.plan.read_bytes())
    import torch
    from autoharness_chess_puzzle import runner

    if not torch.cuda.is_available():
        raise RuntimeError("An allocated GPU is required")
    identity = checkpoint_manifest(Path(plan["model"]))
    if identity["weights_sha256"] != plan["weights_sha256"]:
        raise ValueError("Checkpoint bytes changed")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", plan["model"], "--host", "127.0.0.1", "--port", str(port),
               *plan["server_arguments"]]
    (args.output / "server-command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
               CHESS_PUZZLE_DEFAULT_MAX_TURNS="9", CHESS_PUZZLE_MAX_TURNS_CAP="18",
               CHESS_PUZZLE_FORMAT_RETRIES="1", CHESS_PUZZLE_ILLEGAL_RETRIES="1")
    os.environ.update({k: env[k] for k in env if k.startswith("CHESS_PUZZLE_")})
    started = time.perf_counter()
    summaries = []
    old_client, old_config = runner.LLMClient, runner.LLMConfig
    clients = []

    class Client(RecordedVLLMClient):
        def __init__(self, config):
            super().__init__(config)
            clients.append(self)

    with (args.output / "server.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + plan["startup_timeout_seconds"]
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited during startup: {process.returncode}")
                try:
                    models = json_request(endpoint + "/v1/models", timeout=2)
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("vLLM startup limit exceeded")
                    time.sleep(2)
            if [m["id"] for m in models["data"]] != [plan["model"]]:
                raise ValueError("Unexpected served model")
            (args.output / "served-models.json").write_text(json.dumps(models, indent=2) + "\n")
            runner.LLMClient, runner.LLMConfig = Client, VLLMConfiguration
            for index, shard_path in enumerate(plan["shard_plans"]):
                shard = json.loads(Path(shard_path).read_text())
                output = args.output / f"shard-{index}"
                output.mkdir()
                config = dict(shard["llm_config"], provider="local-vllm", base_url=endpoint,
                              trace_path=str(output / "generations.jsonl"),
                              batch_concurrency=plan["batch_concurrency"],
                              request_timeout_seconds=plan["request_timeout_seconds"])
                summary = runner.evaluate_harness(
                    harness_path=shard["harness"], dataset_path=shard["dataset"], llm_config=config,
                    output_dir=output / "evaluation", limit=shard["examples"], seed=shard["seed"],
                    assistant_token_budget=shard["assistant_token_budget"],
                    policy_max_tokens=shard["policy_max_tokens"])
                clients[-1].close()
                summaries.append(summary)
                print(json.dumps({"shard": index, "status": "COMPLETED", "summary": summary}), flush=True)
        finally:
            runner.LLMClient, runner.LLMConfig = old_client, old_config
            for client in clients:
                client.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    result = {"kind": "chess_vllm_runtime_check", "role": "engineering", "status": "COMPLETED",
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(),
              "model_manifest": identity, "versions": plan["versions"], "summaries": summaries,
              "plan_sha256": file_sha256(args.plan), "seconds": time.perf_counter() - started,
              "training_steps": 0, "external_api_calls": 0,
              "artifact_sha256": {str(p.relative_to(args.output)): file_sha256(p)
                                  for p in sorted(args.output.rglob("*")) if p.is_file()},
              "limitations": plan["limitations"]}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args())
