"""Recover a completed scoped proposal with a metadata-envelope mismatch.

The paid proposer workspace remains immutable. Recovery renames the top-level
``slots`` field to ``candidates`` and removes leading underscores from local
Python identifiers. The latter reconciles the written no-private-attribute
contract with the stricter AST loader; token-level identifier renaming does not
change candidate control flow, strings, prompts, or metadata.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
from pathlib import Path
import shutil
import tokenize

from .audit_native_training_batch import require
from .fast_chart_search import SLOTS, read, validate_v3_candidate_metadata
from .glm_gateway import BudgetJournal
from .isolated_visual_harness import load_isolated_visual_harness
from .native_search import tree_hashes, write_json
from .native_visual_service import ROOT, write_new
from .scoped_proposer import normalize_candidates
from .visual_task import file_sha256


def normalize_candidate_source(source: str) -> tuple[str, dict[str, str], int]:
    """Apply syntax-only callback-contract normalization to candidate source."""
    tree = ast.parse(source)
    require(not any(isinstance(node, ast.Attribute) and node.attr.startswith("_")
                    for node in ast.walk(tree)), "Private attribute access is not recoverable")
    names = sorted({node.id for node in ast.walk(tree)
                    if isinstance(node, ast.Name) and node.id.startswith("_")})
    require(not any(name.startswith("__") for name in names),
            "Python special identifiers are not recoverable")
    public = {node.id for node in ast.walk(tree)
              if isinstance(node, ast.Name) and not node.id.startswith("_")}
    mapping = {name: name.lstrip("_") for name in names}
    require(all(value and value.isidentifier() and value not in public for value in mapping.values()),
            "Private identifier normalization would collide or produce an invalid name")
    require(len(set(mapping.values())) == len(mapping), "Private identifier normalization would collide")
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in mapping:
            token = tokenize.TokenInfo(token.type, mapping[token.string], token.start, token.end, token.line)
        tokens.append(token)
    renamed = tokenize.untokenize(tokens)
    tokens = list(tokenize.generate_tokens(io.StringIO(renamed).readline))
    normalized_tokens = []
    index = 0
    turn_count_replacements = 0
    while index < len(tokens):
        if (index + 3 < len(tokens) and tokens[index].type == tokenize.NAME and
                tokens[index].string == "len" and tokens[index + 1].string == "(" and
                tokens[index + 2].type == tokenize.NAME and
                tokens[index + 2].string == "assistant_turns" and tokens[index + 3].string == ")"):
            token = tokens[index]
            normalized_tokens.append(tokenize.TokenInfo(
                tokenize.NAME, "assistant_turns", token.start, tokens[index + 3].end, token.line))
            index += 4
            turn_count_replacements += 1
            continue
        normalized_tokens.append(tokens[index])
        index += 1
    normalized = tokenize.untokenize(normalized_tokens)
    ast.parse(normalized)
    return normalized, mapping, turn_count_replacements


def recover(root: Path) -> dict:
    root = root.resolve()
    plan = read(root / "plan.json")
    require(plan.get("selection_protocol") == "safety_v3", "Recovery is V3-only")
    require(not (root / "proposal-ready.json").exists(), "Proposal is already ready")

    request = read(root / "proposal-request.json")
    names = tuple(request.get("next_names", ()))
    require(names == SLOTS and request.get("iteration") == 1, "Unexpected proposal allocation")

    paid = root / "paid-proposal"
    result = read(paid / "proposer-result.json")
    require(result.get("exit_code") == 0, "The proposer process did not complete successfully")
    view = paid / "proposer-workspace"
    run = root / "search"
    require(not (paid / "proposal-integrity.json").exists(), "Proposal already has an integrity receipt")

    before = tree_hashes(run)
    after = tree_hashes(view, ignore_cli_state=True)
    require(all(after.get(name) == digest for name, digest in before.items()),
            "Proposer changed protected evidence")
    allowed = {f"harnesses/{name}/harness.py" for name in names}
    allowed.update({"pending_eval.json", "logs/iteration_001/report.md"})
    require(all(name in allowed or name.startswith("logs/claude_sessions/")
                for name in after.keys() - before.keys()), "Unexpected proposer artifact")
    require(all((view / f"harnesses/{name}/harness.py").is_file() for name in names),
            "Recovery requires all allocated candidates")

    pending_path = view / "pending_eval.json"
    raw = json.loads(pending_path.read_text())
    require(isinstance(raw, dict) and "candidates" not in raw and isinstance(raw.get("slots"), list),
            "Recovery only supports the slots-to-candidates envelope mismatch")
    reframed = dict(raw)
    reframed["candidates"] = reframed.pop("slots")
    parents = {"h0"}
    normalized = normalize_candidates(reframed, names, parents)

    recovery_dir = root / "proposal-recovery"
    recovery_dir.mkdir(exist_ok=True)
    require(not (recovery_dir / "receipt.json").exists(), "Recovery already has a receipt")
    normalized_path = recovery_dir / "pending_eval.normalized.json"
    if normalized_path.exists():
        require(read(normalized_path) == normalized, "Existing normalized metadata differs")
    else:
        write_json(normalized_path, normalized)
    candidate_transforms = {}
    normalized_harnesses = {}
    for name in names:
        source_path = view / f"harnesses/{name}/harness.py"
        normalized_source, mapping, turn_count_replacements = normalize_candidate_source(source_path.read_text())
        normalized_candidate = recovery_dir / "harnesses" / name / "harness.py"
        normalized_candidate.parent.mkdir(parents=True, exist_ok=True)
        if normalized_candidate.exists():
            require(normalized_candidate.read_text() == normalized_source,
                    "Existing normalized candidate differs")
        else:
            normalized_candidate.write_text(normalized_source)
        harness = load_isolated_visual_harness(normalized_candidate)
        harness.unchanged()
        require(isinstance(harness.invoke("format_observation", question="Which bar is higher? A: left; B: right."), str),
                "format_observation callback preflight failed")
        require(isinstance(harness.invoke("prepare_tool", arguments={}, image_size=[640, 480]), dict),
                "prepare_tool callback preflight failed")
        require(isinstance(harness.invoke("format_feedback", text="visible feedback"), str),
                "format_feedback callback preflight failed")
        harness.invoke("parse_answer", text="Final answer: A")
        for assistant_turns in (1, 2, 3):
            value = harness.invoke("nudge", text="Final answer: A", assistant_turns=assistant_turns)
            require(value is None or isinstance(value, str), "nudge callback preflight failed")
        normalized_harnesses[name] = normalized_candidate
        candidate_transforms[name] = {
            "source_sha256": file_sha256(source_path),
            "normalized_sha256": file_sha256(normalized_candidate),
            "identifier_renames": mapping,
            "integer_turn_count_replacements": turn_count_replacements,
            "real_type_callback_preflight": "PASS",
        }

    imported = sorted(allowed)
    for name in imported:
        destination = run / name
        require(not destination.exists(), f"Import destination already exists: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == "pending_eval.json":
            write_json(destination, normalized)
        elif name.startswith("harnesses/"):
            candidate = Path(name).parts[1]
            destination.write_bytes(normalized_harnesses[candidate].read_bytes())
        else:
            destination.write_bytes((view / name).read_bytes())
    shutil.copytree(view / "logs/claude_sessions", run / "logs/claude_sessions", dirs_exist_ok=True)
    validate_v3_candidate_metadata(root, plan)
    require(all(candidate_transforms[name]["normalized_sha256"] ==
                file_sha256(run / f"harnesses/{name}/harness.py") for name in names),
            "Normalized candidate bytes changed during import")
    require(tree_hashes(view, ignore_cli_state=True) == after, "Paid proposer workspace changed during recovery")

    sessions = list((run / "logs/claude_sessions").iterdir())
    require(len(sessions) == 1, "Unexpected proposer session count")
    receipt = {
        "status": "PASS_DETERMINISTIC_SCHEMA_ENVELOPE_RECOVERY",
        "transform": "slots renamed to candidates; local identifiers and integer assistant_turns normalized",
        "original_pending_sha256": file_sha256(pending_path),
        "normalized_pending_sha256": file_sha256(normalized_path),
        "candidate_transforms": candidate_transforms,
        "protected_files": len(before),
        "original_before_sha256": before,
        "paid_workspace_after_sha256": after,
        "imported": imported,
        "proposer_result": result,
    }
    write_new(recovery_dir / "receipt.json", receipt)
    write_new(paid / "proposal-integrity.json", {
        "status": receipt["status"],
        "iteration": 1,
        "allocated_slots": list(names),
        "imported": imported,
        "metadata_source": "deterministic_slots_envelope_recovery",
        "recovery_receipt": str(recovery_dir / "receipt.json"),
        "recovery_receipt_sha256": file_sha256(recovery_dir / "receipt.json"),
        "protected_files": len(before),
        "original_before_sha256": before,
        "landlock_scope": "disposable audited MH archive only",
    })
    journal = BudgetJournal(ROOT / "data/glm-budget/ledger.jsonl")
    write_new(root / "proposal-ready.json", {
        "request_sha256": file_sha256(root / "proposal-request.json"),
        "session": str(sessions[0]),
        "session_sha256": tree_hashes(sessions[0]),
        "budget": journal.summary(),
        "imported_sha256": {name: file_sha256(run / name) for name in set(imported)},
        "actual_data_role": "H",
        "optimization_audit_transferred": False,
        "recovery_receipt_sha256": file_sha256(recovery_dir / "receipt.json"),
    })
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.root), indent=2, default=str))
