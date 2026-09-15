"""Exercise F0 contexts through the native search loop with synthetic scores."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.controlled_conditions import native_search_context
from ours.tests.test_prompt_subspace import BASELINE, replace_user, write_harness


@unittest.skipUnless(importlib.util.find_spec('chess'), 'Requires native Chess runtime')
class NativeConditionTests(unittest.TestCase):
    def test_incoming_harness_and_search_coordinates_reach_original_loop(self):
        from autoharness_chess_puzzle import runner
        from meta_harness import meta_harness_chess_puzzle as original
        incoming_source = replace_user(BASELINE.read_text(), "'Incoming phase prompt. {observation}'")
        changed_prompt = replace_user(incoming_source, "'Candidate phase prompt. {observation}'")
        changed_observation = incoming_source.replace(
            'return str(observation).strip()', 'return str(observation).lower()')
        self.assertNotEqual(changed_observation, incoming_source)
        for condition, source, valid in [
            ('whale', changed_observation, True),
            ('harness_only', changed_observation, True),
            ('whale_fst', changed_observation, False),
            ('whale_fst', changed_prompt, True),
        ]:
            with self.subTest(condition=condition, valid=valid), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                incoming = root / 'incoming.py'
                incoming.write_text(incoming_source)
                config = root / 'config.json'
                config.write_text(json.dumps({'models': [{'model': 'fixture'}], 'seeds': [43],
                    'task_profiles': {'mh_val': {'dataset_path': 'fixture-only', 'limit': 2}}}))
                args = SimpleNamespace(run_name='search', config=str(config), iterations=1,
                    proposer_model='fixture', proposer_effort='low', propose_timeout=1,
                    proposals_per_iter=1, early_stop_success_rate=1., fresh=False, force=False,
                    start_iteration=1, early_stop_min_iters=0, early_stop_patience=2,
                    eval_only=False, use_api_key=True, prompt_only=condition == 'whale_fst')
                evaluated = []
                def proposal(**kwargs):
                    self.assertEqual(original.PROMPT_ONLY, condition == 'whale_fst')
                    write_harness(kwargs['run_dir'], 'h1', source)
                    (kwargs['run_dir'] / 'pending_eval.json').write_text(
                        json.dumps({'candidates': [{'name': 'h1'}]}))
                    return SimpleNamespace(show=lambda: None)
                def evaluate(**kwargs):
                    harness = Path(kwargs['harness_path'])
                    evaluated.append(harness.parent.name)
                    if harness.parent.name == 'h0':
                        self.assertEqual(harness.read_text(), incoming_source)
                    self.assertEqual(kwargs['seed'], 43)
                    rows = [{'example_id': 'redacted', 'reward': float(harness.parent.name == 'h1'),
                             'turn_count': 1, 'metrics': {'assistant_tokens_used': 1},
                             'prompt': [], 'harness_trace': []} for _ in range(2)]
                    summary = runner.summarize_outputs(rows, {'seed': 43, 'fixture_only': True})
                    output = Path(kwargs['output_dir'])
                    output.mkdir(parents=True)
                    (output / 'val.json').write_text(json.dumps(summary))
                    return summary
                old_mode, old_root = original.PROMPT_ONLY, original.RUNS_DIR
                old_override = os.environ.get('BASELINE_HARNESS_OVERRIDE')
                with native_search_context(condition, root, incoming) as (native, benchmark, provenance), \
                     patch.object(native, 'propose_claude_with_retries', proposal), \
                     patch.object(benchmark, 'evaluate_harness', evaluate):
                    if valid:
                        native.run_evolve(args)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'No valid candidates'):
                            native.run_evolve(args)
                    self.assertEqual(evaluated, ['h0', 'h1'] if valid else ['h0'])
                    self.assertEqual(native.get_accepted_harness(root / 'search'), 'h1' if valid else 'h0')
                self.assertEqual((original.PROMPT_ONLY, original.RUNS_DIR), (old_mode, old_root))
                self.assertEqual(os.environ.get('BASELINE_HARNESS_OVERRIDE'), old_override)
                print(json.dumps({'kind': 'native_condition_fixture', 'status': 'PASS',
                    'provenance': provenance, 'evaluated': evaluated,
                    'task_scores_are_synthetic': True}), flush=True)

    def test_weight_only_cannot_invoke_search(self):
        with self.assertRaisesRegex(ValueError, 'must not search'):
            with native_search_context('weight_only', Path('/unused'), BASELINE):
                self.fail('Weight-only search was allowed')


if __name__ == '__main__':
    unittest.main()
