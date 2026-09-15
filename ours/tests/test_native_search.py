"""Guard against evidence mutation and swallowed native evaluation failures."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.native_search import check_proposal, isolated_proposal, normalize_metadata, require_complete_sweep, tree_hashes, validate_baseline_cache


class NativeSearchTests(unittest.TestCase):
    def test_metadata_alias_does_not_permit_conflicts_or_path_escape(self):
        payload = {"candidates": [{"slot": "h1", "path": "harnesses/h1/harness.py", "notes": "retained"}]}
        normalized = normalize_metadata(payload)
        self.assertEqual(normalized["candidates"][0]["name"], "h1")
        self.assertNotIn("name", payload["candidates"][0])
        for change in ({"name": "h0"}, {"path": "../h0/harness.py"}, {"parent": "h9"}):
            with self.assertRaises(ValueError):
                normalize_metadata({"candidates": [dict(payload["candidates"][0], **change)]})
    def test_evidence_mutation_and_path_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence.json").write_text("original")
            before = tree_hashes(root)
            (root / "harnesses/h1").mkdir(parents=True)
            (root / "harnesses/h1/harness.py").write_text("candidate")
            (root / "pending_eval.json").write_text(json.dumps({"candidates": [{"name": "../h0"}]}))
            with self.assertRaisesRegex(ValueError, "allocated h1"):
                check_proposal(root, before)
            (root / "pending_eval.json").write_text(json.dumps({"candidates": [{"name": "h1"}]}))
            self.assertEqual(len(check_proposal(root, before)), 2)
            (root / ".cli-state").mkdir()
            (root / ".cli-state/latest").symlink_to(root / "evidence.json")
            self.assertEqual(len(check_proposal(root, before)), 2)
            (root / "evidence.json").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "changed its evidence"):
                check_proposal(root, before)

    def test_swallowed_failure_cannot_be_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                require_complete_sweep([("fixture", False)], [("h1", Path("fixture"))], Path(directory))

    def test_cached_scores_reject_a_different_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts.json").write_text("{}")
            previous = {"dataset": "fixture", "examples": 8, "seed": 42,
                        "target_config": {"model": "checkpoint-a"}}
            (root / "plan.json").write_text(json.dumps(previous))
            current = dict(previous, target_config={"model": "checkpoint-b"},
                           baseline_cache={"directory": str(root), "artifact_manifest": str(root / "artifacts.json")})
            with self.assertRaisesRegex(ValueError, "target_config"):
                validate_baseline_cache(current)

    def test_isolated_native_proposer_signature_and_import_boundary(self):
        @contextmanager
        def fake_relay(*args, **kwargs):
            yield {"base_url": "http://127.0.0.1:1", "token": "offline-fixture"}

        def propose(*, task_prompt, iteration, run_dir, proposer_model, proposer_effort,
                    timeout_seconds, use_api_key):
            self.assertTrue(use_api_key)
            self.assertEqual((run_dir / "harnesses/h0/harness.py").read_text(), "baseline")
            (run_dir / "harnesses/h1").mkdir()
            (run_dir / "harnesses/h1/harness.py").write_text("new candidate")
            (run_dir / "pending_eval.json").write_text('{"candidates":[{"name":"h1"}]}')
            return SimpleNamespace(exit_code=0, duration_seconds=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "search"
            (source / "harnesses/h0").mkdir(parents=True)
            (source / "logs/claude_sessions").mkdir(parents=True)
            (source / "harnesses/h0/harness.py").write_text("baseline")
            wrapper = SimpleNamespace(build_command=lambda: None, _EMPTY_PLUGIN_DIR=None)
            native = SimpleNamespace(claude_wrapper=wrapper, propose_claude=propose)
            journal = SimpleNamespace(summary=lambda: {"scope": "offline fixture"})
            with patch("ours.native_search.relay", fake_relay):
                isolated_proposal(native, {"max_api_requests": 1}, root, journal,
                                 task_prompt="fixture", iteration=1, next_names=["h1"], run_dir=source,
                                 proposer_model="fixture", proposer_effort="low", timeout_seconds=1,
                                 use_api_key=True)
            self.assertEqual((source / "harnesses/h0/harness.py").read_text(), "baseline")
            self.assertEqual((source / "harnesses/h1/harness.py").read_text(), "new candidate")
            self.assertFalse((source / ".cli-state").exists())
