"""Exercise the original FST validator, proposer prompt path and search loop.

Rollout scores and replies are CPU fixtures, not model performance evidence.
No proposer subprocess, API request, or GPU allocation is made by these tests.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.prompt_subspace import CONTRACT, prompt_subspace, validate_literal_prompts


BASELINE = Path('upstream/WHALE/domains/chess_puzzles/environments/chess_puzzle/base_harness.py')


def replace_user(source, expression, annotation=''):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == 'USER_PROMPT' for t in n.targets))
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + f'USER_PROMPT{annotation} = {expression}\n' + ''.join(lines[node.end_lineno:])


def write_harness(root, name, source):
    path = root / 'harnesses' / name / 'harness.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


class LiteralPromptTests(unittest.TestCase):
    def test_prompt_values_and_annotations_cannot_execute_code(self):
        base = BASELINE.read_text()
        validate_literal_prompts(base)
        validate_literal_prompts(replace_user(base, "'Choose one move.\\n' '{observation}'", ': str'))
        validate_literal_prompts(replace_user(base, "'{observation}'", ": 'str'"))
        for value, annotation in [
            ('(random.seed(123), "{observation}")[1]', ''),
            ('f"{random.seed(123)}"', ''),
            ('"{observation}"', ': (random.seed(123), str)[1]'),
        ]:
            with self.subTest(value=value, annotation=annotation), self.assertRaises(ValueError):
                validate_literal_prompts(replace_user(base, value, annotation))


@unittest.skipUnless(importlib.util.find_spec('chess'), 'Requires native Chess runtime')
class NativePromptSubspaceTests(unittest.TestCase):
    def test_native_expression_mask_cannot_bypass_the_textual_boundary(self):
        from meta_harness import meta_harness_chess_puzzle as native
        from autoharness_chess_puzzle.harness import load_harness
        base = BASELINE.read_text()
        candidate = replace_user(base, '(random.seed(123), "{observation}")[1]')
        with tempfile.TemporaryDirectory(prefix='whale-fst-expression-') as tmp:
            root = Path(tmp)
            write_harness(root, 'h0', base)
            path = write_harness(root, 'h1', candidate)
            native.validate_prompt_only(root, 'h1')
            state = random.getstate()
            try:
                load_harness(path)
                self.assertNotEqual(random.getstate(), state)
            finally:
                random.setstate(state)
            with prompt_subspace(native), patch.object(native, 'load_harness') as loader:
                with self.assertRaisesRegex(ValueError, 'literal strings'):
                    native.validate_candidate(root, 'h1')
                loader.assert_not_called()
            self.assertEqual(random.getstate(), state)

    def test_original_wrapper_receives_full_contract_and_explicit_restriction(self):
        from meta_harness import meta_harness_chess_puzzle as native
        original_mode, original_skill = native.PROMPT_ONLY, native.PROMPT_ONLY_SKILL_DIR
        original_validate = native.validate_candidate
        class CommandBoundary(Exception):
            pass
        captured = []
        def capture(prompt, model, allowed_tools, system_prompt, *args, **kwargs):
            captured.append(system_prompt)
            self.assertIn((native.SKILL_DIR / 'SKILL.md').read_text(), system_prompt)
            self.assertIn(CONTRACT.read_text(), system_prompt)
            self.assertEqual(allowed_tools, ['Read', 'Glob', 'Grep', 'Write', 'Edit'])
            raise CommandBoundary()
        with tempfile.TemporaryDirectory(prefix='whale-fst-prompt-') as tmp:
            with self.assertRaises(CommandBoundary):
                with prompt_subspace(native) as provenance, patch.object(native.claude_wrapper, 'build_command', capture):
                    self.assertFalse(provenance['original_unpublished_skill_recovered'])
                    native.propose_claude(task_prompt='CPU fixture only', iteration=1, run_dir=Path(tmp),
                        proposer_model='fixture', proposer_effort='low', timeout_seconds=1, use_api_key=True)
        self.assertEqual(len(captured), 1)
        self.assertEqual(native.PROMPT_ONLY, original_mode)
        self.assertEqual(native.PROMPT_ONLY_SKILL_DIR, original_skill)
        self.assertIs(native.validate_candidate, original_validate)

    def test_original_loop_evaluates_text_changes_and_rejects_nonprompt_changes(self):
        from autoharness_chess_puzzle import runner
        from meta_harness import chess_puzzle_benchmark as benchmark
        from meta_harness import meta_harness_chess_puzzle as native
        base = BASELINE.read_text()
        candidates = [
            ('prompt', replace_user(base, "'Use the visible state. {observation}'", ': str'), True),
            ('retry', base.replace('FORMAT_RETRY_BUDGET = 1', 'FORMAT_RETRY_BUDGET = 2'), False),
            ('turns', base.replace('MAX_TURNS = 9', 'MAX_TURNS = 18'), False),
            ('parser', base.replace('return "__ambiguous_multiple_moves__"', 'return "e2e4"'), False),
            ('observation', base.replace('return str(observation).strip()', 'return str(observation).lower()'), False),
        ]
        for name, source, valid in candidates:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='whale-fst-loop-') as tmp:
                root = Path(tmp)
                config = root / 'config.json'
                config.write_text(json.dumps({'models': [{'model': 'fixture'}], 'seeds': [42],
                    'task_profiles': {'mh_val': {'dataset_path': 'fixture-only', 'limit': 2}}}))
                args = SimpleNamespace(run_name='search', config=str(config), iterations=1,
                    proposer_model='fixture', proposer_effort='low', propose_timeout=1,
                    proposals_per_iter=1, early_stop_success_rate=1., fresh=False, force=False,
                    start_iteration=1, early_stop_min_iters=0, early_stop_patience=2,
                    eval_only=False, use_api_key=True, prompt_only=True)
                evaluated = []
                def proposal(**kwargs):
                    self.assertTrue(native.PROMPT_ONLY)
                    write_harness(kwargs['run_dir'], 'h1', source)
                    (kwargs['run_dir'] / 'pending_eval.json').write_text(json.dumps({'candidates': [{'name': 'h1'}]}))
                    return SimpleNamespace(show=lambda: None)
                def evaluate(**kwargs):
                    harness = Path(kwargs['harness_path']).parent.name
                    evaluated.append(harness)
                    score = float(harness == 'h1')
                    rows = [{'example_id': 'redacted', 'reward': score, 'turn_count': 1,
                        'metrics': {'assistant_tokens_used': 1}, 'prompt': [], 'harness_trace': []} for _ in range(2)]
                    summary = runner.summarize_outputs(rows, {'seed': 42, 'fixture_only': True})
                    out = Path(kwargs['output_dir'])
                    out.mkdir(parents=True)
                    (out / 'val.json').write_text(json.dumps(summary))
                    return summary
                with patch.dict(os.environ, BASELINE_HARNESS_OVERRIDE=''), \
                     patch.object(native, 'RUNS_DIR', root), \
                     patch.object(native, 'propose_claude_with_retries', proposal), \
                     patch.object(benchmark, 'evaluate_harness', evaluate), prompt_subspace(native):
                    if valid:
                        native.run_evolve(args)
                        self.assertEqual(evaluated, ['h0', 'h1'])
                        self.assertEqual(native.get_accepted_harness(root / 'search'), 'h1')
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'No valid candidates'):
                            native.run_evolve(args)
                        self.assertEqual(evaluated, ['h0'])
                        self.assertEqual(native.get_accepted_harness(root / 'search'), 'h0')
                        rows = [json.loads(line) for line in (root / 'search/logs/evolution_summary.jsonl').read_text().splitlines()]
                        self.assertEqual(rows[-1]['status'], 'rejected_prompt_only')


if __name__ == '__main__':
    unittest.main()
