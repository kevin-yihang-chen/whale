"""Load WHALE's bundled shared tool schemas missing from its Chess directory.

Only the verl.tools package is sourced from the pinned Math checkout. Chess
trainer, actor, rollout and reward implementations still come from Chess.
This declared release repair is shared across conditions, outside E1-E4.
"""
import importlib.util
from pathlib import Path
import runpy
import subprocess
import sys

from .preflight import PIN
from .visual_task import file_sha256


def restore_bundled_tools(root):
    root = Path(root).resolve()
    upstream = root / "upstream/WHALE"
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("Unexpected WHALE revision")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("Upstream checkout must be unchanged")
    import verl
    if Path(verl.__file__).resolve() != upstream / "domains/chess_puzzles/verl/__init__.py":
        raise ValueError("The native Chess verl package must be first on PYTHONPATH")
    if importlib.util.find_spec("verl.tools") is not None:
        raise ValueError("A tool package already exists; refusing an ambiguous replacement")
    package = upstream / "domains/math_reasoning/verl/tools"
    peer = upstream / "domains/search_qa/verl/tools/schemas.py"
    if file_sha256(package / "schemas.py") != file_sha256(peer):
        raise ValueError("Bundled schema copies differ")
    spec = importlib.util.spec_from_file_location("verl.tools", package / "__init__.py",
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["verl.tools"] = module
    spec.loader.exec_module(module)
    verl.tools = module
    return {"restored_namespace": "verl.tools", "source": str(package.relative_to(root)),
            "sha256": {str(p.relative_to(root)): file_sha256(p) for p in sorted(package.rglob("*.py"))},
            "peer_schema_identical": True, "upstream_commit": PIN}


def prepare_worker():
    """Apply the same declared package repair in each fresh Ray worker."""
    root = Path(__file__).resolve().parents[1]
    existing = sys.modules.get("verl.tools")
    if existing is None:
        return restore_bundled_tools(root)
    expected = root / "upstream/WHALE/domains/math_reasoning/verl/tools/__init__.py"
    if Path(existing.__file__).resolve() != expected:
        raise ValueError("A worker imported an unexpected verl.tools implementation")
    return {"restored_namespace": "verl.tools", "already_loaded": True}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    restore_bundled_tools(root)
    runpy.run_path(str(root / "upstream/WHALE/domains/chess_puzzles/scripts/chess_puzzle_preamble_disagg_rsft.py"),
                  run_name="__main__")
