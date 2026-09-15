"""Exercise WHALE's Claude wrapper with GLM on a disposable harness copy.

This verifies file-tool transport, not a search iteration or performance gain.
The only requested edit is a specified USER_PROMPT sentence. AST checks reject
any other change. The original project and pinned upstream remain untouched.
"""
import ast
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from .glm_gateway import BudgetJournal, MODEL, relay
from .visual_task import file_sha256


SENTENCE = "Check the board coordinates before selecting a move."


def normalized_source(text):
    tree = ast.parse(text)
    prompts = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "USER_PROMPT" for t in node.targets):
            prompts.append(ast.literal_eval(node.value))
            node.value = ast.Constant(value="__PROMPT__")
    if len(prompts) != 1 or not isinstance(prompts[0], str):
        raise ValueError("Expected exactly one literal user prompt")
    return ast.dump(tree), prompts[0]


def main(attempt=1):
    root = Path.cwd()
    suffix = "" if attempt == 1 else f"-attempt{attempt}"
    run_dir = root / f"data/glm-proposer-probe-20260909{suffix}"
    run_dir.mkdir(exist_ok=False)
    source = root / "upstream/WHALE/domains/chess_puzzles/environments/chess_puzzle/base_harness.py"
    public_check = json.loads((root / "results/proposer-public-source-verification-20260909.json").read_text())
    if not public_check["identical_public_bytes"] or file_sha256(source) != public_check["sha256"]:
        raise ValueError("Only the independently verified public harness is permitted")
    candidate = run_dir / "candidate.py"
    candidate.write_bytes(source.read_bytes())
    wrapper_path = root / "upstream/WHALE/domains/chess_puzzles/meta_harness/claude_wrapper.py"
    spec = importlib.util.spec_from_file_location("whale_probe_claude_wrapper", wrapper_path)
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    wrapper._EMPTY_PLUGIN_DIR = run_dir / "empty_plugins"
    journal = BudgetJournal(root / "data/glm-budget/ledger.jsonl")
    journal.initialize_probe(root / "results/glm-cn-connectivity-20260909.json")
    original_build = wrapper.build_command

    def bounded_command(*args, **kwargs):
        cmd = original_build(*args, **kwargs)
        cmd.remove("--dangerously-skip-permissions")
        cmd[0] = str(binary)
        cmd += ["--bare", "--no-session-persistence", "--max-turns", "4",
                "--permission-mode", "dontAsk", "--debug-file", str(run_dir / "debug.log")]
        return [sys.executable, str(root / "ours/confined_exec.py"), "--workspace", str(run_dir),
                "--read", str(wrapper._EMPTY_PLUGIN_DIR), "--", *cmd]

    wrapper.build_command = bounded_command
    prompt = ("This is a file-tool connectivity check. Read candidate.py in the current directory, "
              "then use Edit to append exactly this sentence to the USER_PROMPT string: " + SENTENCE +
              " Preserve all other definitions and logic. Do not read or modify any other file. "
              "Do not run tests or search the internet. After the edit, reply DONE.")
    before = normalized_source(source.read_text())
    original_env = os.environ.copy()
    config_dir = run_dir / "claude-config"
    config_dir.mkdir()
    (run_dir / "tmp").mkdir()
    binary = root / "data/claude-runtime-2.1.236/claude"
    bin_dir = run_dir / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").symlink_to(binary)
    try:
        capture = run_dir / "raw-sse"
        capture.mkdir()
        with relay(journal, "/userhome/cs3/yihangc/.config/whale-delta/credentials.json", max_requests=8,
                   capture_directory=capture) as gateway:
            for name in list(os.environ):
                if name.endswith(("API_KEY", "AUTH_TOKEN")) or name in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                    os.environ.pop(name)
            os.environ.update(PATH=str(bin_dir) + os.pathsep + original_env["PATH"],
                ANTHROPIC_BASE_URL=gateway["base_url"], ANTHROPIC_API_KEY=gateway["token"],
                ANTHROPIC_DEFAULT_OPUS_MODEL=MODEL, ANTHROPIC_DEFAULT_SONNET_MODEL=MODEL,
                ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL, CLAUDE_CONFIG_DIR=str(config_dir),
                TMPDIR=str(run_dir / "tmp"),
                CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1", DISABLE_UPDATES="1",
                CLAUDE_CODE_MAX_OUTPUT_TOKENS="4096", CLAUDE_SETTING_SOURCES="project")
            result = wrapper.run(prompt, model=MODEL, tools=["Read", "Edit"],
                allowed_tools=["Read", "Edit"], cwd=str(run_dir), log_dir=str(run_dir / "logs"),
                name="glm_tool_transport", timeout_seconds=90, effort="low", progress=False)
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        wrapper.build_command = original_build
    after = normalized_source(candidate.read_text())
    valid = before[0] == after[0] and after[1].replace(SENTENCE, "").strip() == before[1].strip() and SENTENCE in after[1]
    report = {"kind": "original_whale_wrapper_glm_tool_probe", "role": "engineering",
              "status": "PASS" if valid and result.exit_code == 0 else "FAIL",
              "model": MODEL, "native_wrapper_sha256": file_sha256(wrapper_path),
              "candidate_before_sha256": file_sha256(source), "candidate_after_sha256": file_sha256(candidate),
              "only_requested_prompt_changed": valid, "cli_exit_code": result.exit_code,
              "budget": journal.summary(), "api_tariff_usd_from_cli_is_authoritative": False,
              "filesystem_confinement": "Landlock ABI1, only disposable workspace writable; other user files unreadable",
              "artifacts": str(run_dir), "proposer_search_iterations": 0,
              "probe_source_sha256": file_sha256(Path(__file__)),
              "limitations": "Specified edit on a disposable file; not a candidate search or scientific result"}
    destination = root / f"results/glm-proposer-tool-probe-20260909{suffix}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("Attempt must be positive")
    main(args.attempt)
