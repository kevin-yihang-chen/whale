"""Repeatable, read-only WHALE release/environment readiness checks.

Infrastructure validation, not a Method contribution or numerical reproduction.
No model imports, package installation, network access or credential values.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

PIN = "fbe125eb7abea7f760c99ab9acc1a6261e708fc6"


def inspect_release(root: Path) -> dict:
    root = root.resolve()
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    head = git("rev-parse", "HEAD")
    findings = []
    evidence = {}
    def source(path: str) -> str:
        data = (root / path).read_bytes()
        evidence[path] = hashlib.sha256(data).hexdigest()
        return data.decode()
    def finding(key: str, ready: bool, detail: str, paths: list[str]) -> None:
        for path in paths:
            if (root / path).is_file():
                source(path)
        findings.append({"id": key, "status": "PASS" if ready else "BLOCKED",
                         "detail": detail, "paths": paths})
    finding("source_pin", head == PIN, f"Expected {PIN}; observed {head}", [])
    finding("upstream_clean", not git("status", "--porcelain"), "Upstream checkout must remain clean", [])
    requirements = {
        "math_environment": ("domains/math_reasoning/environments/retool/env.py",
                             "domains/math_reasoning/meta_harness/meta_harness_retool.py"),
        "search_mh_config": ("domains/search_qa/meta_harness/config-search-r1.json",
                             "domains/search_qa/meta_harness/meta_harness_search_r1.py"),
    }
    for key, (needed, reference) in requirements.items():
        finding(key, (root / needed).is_file(), "Required released file: " + needed, [needed, reference])
    for domain in ("chess_puzzles", "math_reasoning", "search_qa"):
        package = f"domains/{domain}/verl"
        finding("editable_install_" + domain,
                any((root / package / f).is_file() for f in ("setup.py", "pyproject.toml")),
                "Documented pip install -e requires packaging metadata", ["docs/reproducing.md"])
        skill = {"chess_puzzles": "chess-puzzle", "math_reasoning": "retool", "search_qa": "search-r1"}[domain]
        needed = f"domains/{domain}/meta_harness/skills/meta-harness-{skill}-promptonly/SKILL.md"
        finding("fst_skill_" + domain, (root / needed).is_file(), "Prompt-only control skill: " + needed, [needed])
    finding("chess_client", importlib.util.find_spec("autoharness_textarena") is not None,
            "Chess runner imports autoharness_textarena; a traceable compatible source is required",
            ["domains/chess_puzzles/autoharness_chess_puzzle/runner.py",
             "domains/chess_puzzles/requirements.txt"])
    finding("proposer_cli", shutil.which("claude") is not None,
            "Released proposer invokes the Claude CLI; presence alone does not validate authentication",
            ["domains/chess_puzzles/meta_harness/claude_wrapper.py"])
    source("run/lib/alternate.sh")
    source("domains/chess_puzzles/meta_harness/chess_puzzle_benchmark.py")
    source("domains/chess_puzzles/meta_harness/config-chess-puzzle.json")
    findings.append({"id": "live_weight_handoff", "status": "NOT_VERIFIED",
                     "detail": "Exporting VLLM_MODEL is not evidence that the evaluator serves the updated weights. "
                               "Require checkpoint export, server reload and content-bound train/eval receipts.",
                     "paths": ["run/lib/alternate.sh", "domains/chess_puzzles/meta_harness/chess_puzzle_benchmark.py",
                               "domains/chess_puzzles/meta_harness/config-chess-puzzle.json"]})
    capabilities = {name: importlib.util.find_spec(name) is not None for name in
                    ("torch", "transformers", "vllm", "ray", "chess", "datasets", "PIL", "pytest")}
    return {"schema": 1, "kind": "release_environment_preflight", "status": "NOT_READY",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(), "upstream_commit": head,
            "python": sys.version.split()[0], "executable": sys.executable,
            "capabilities": capabilities, "findings": findings, "source_sha256": evidence,
            "model_calls": 0, "gpu_jobs_submitted": 0,
            "limitation": "Static readiness checks; no end-to-end training or benchmark scores."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("upstream/WHALE"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_release(args.upstream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "blocked": [f["id"] for f in report["findings"] if f["status"] != "PASS"]}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
