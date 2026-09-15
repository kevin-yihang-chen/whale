"""Check E3/E4 handoff selection against original scoring and real Hydra config.

Search scores here are fixtures; no target inference, training or API request.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('chess') and importlib.util.find_spec('omegaconf'),
                     'Requires native Chess/training configuration runtime')
class AlternationTrainingTests(unittest.TestCase):
    def test_independent_handoff_rule_matches_original_frontier_and_early_stop(self):
        from meta_harness import chess_puzzle_benchmark as benchmark
        from meta_harness import meta_harness_chess_puzzle as native
        from ours.alternation_training import selected_from_audits
        cases = [(0, 0, 32, 32), (0, 0, 64, 32), (0, 0, 32, 64),
                 (16, 15, 64, 32), (15, 16, 32, 64), (32, 32, 32, 64)]
        for s0, s1, c0, c1 in cases:
            with self.subTest(scores=(s0, s1), calls=(c0, c1)), tempfile.TemporaryDirectory(prefix='whale-handoff-rule-') as tmp:
                root = Path(tmp)
                logs = root / 'logs'
                audits = {}
                for h, solved, calls in [('h0', s0, c0), ('h1', s1, c1)]:
                    directory = logs / 'mh_val' / h / 'fixture'
                    directory.mkdir(parents=True)
                    (directory / 'val.json').write_text(json.dumps({'success_rate': solved / 32,
                        'mean_turn_count': calls / 32, 'avg_reward': solved / 32, 'num_examples': 32}))
                    hp = root / 'harnesses' / h / 'harness.py'
                    hp.parent.mkdir(parents=True)
                    hp.write_text('# CPU score fixture only\n')
                    audits[h] = {'kind': 'mh_phase_audit', 'status': 'PASS', 'harness': h,
                                 'examples': 32, 'solved': solved, 'calls': calls}
                # The staged search declares deterministic h0-before-h1 ingestion.
                from unittest.mock import patch
                original_load = benchmark.load_results
                with patch.object(benchmark, 'load_results', lambda p: dict(sorted(original_load(p).items()))):
                    frontier = benchmark.print_frontier(logs, {'task_profiles': {'mh_val': {}}})
                rows = native.candidate_summary_rows(1, [{'name': 'h1'}], logs, frontier)
                stop = native.find_early_stop_candidate(rows, ['h1'], 1.)
                expected = stop['harness'] if stop else native.pick_accepted(frontier, root)
                self.assertEqual(selected_from_audits(audits), expected)
                audits['h1']['status'] = 'PARTIAL'
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    selected_from_audits(audits)

    def test_full_resolved_configuration_rejects_model_harness_or_loss_drift(self):
        from omegaconf import OmegaConf
        from ours.alternation_training import check_config
        path = Path('results/next-rsft-cpu-config-provisional-20260909.log')
        text = path.read_text()
        actual = OmegaConf.to_container(OmegaConf.create(text[text.index('model_engine: dp\n'):]), resolve=True)
        plan = {'resolved_config': str(path)}
        check_config(plan, actual, 'phase2-cpu-provisional')
        mutations = [
            (('actor_rollout_ref', 'model', 'path'), '/wrong/checkpoint'),
            (('ray_kwargs', 'ray_init', 'runtime_env', 'env_vars', 'CHESS_PUZZLE_HARNESS_PATH'), '/wrong/harness.py'),
            (('trainer', 'online_rsft', 'score_threshold'), 0.),
            (('actor_rollout_ref', 'actor', 'optim', 'lr'), 1e-3),
            (('data', 'train_files'), ['/wrong/split.parquet']),
        ]
        for keys, value in mutations:
            with self.subTest(field=keys):
                changed = deepcopy(actual)
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    check_config(plan, changed, 'phase2-cpu-provisional')


if __name__ == '__main__':
    unittest.main()
