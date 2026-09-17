"""Synthetic CPU control-flow fixtures; no VLM scores or API calls are produced."""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours import visual_search_bridge as bridge
from ours.acceptance import CandidateMetrics, EvidenceConstrainedAcceptance
from ours.adapter import WHALEAcceptanceAdapter
from ours.evidence import AuditReceipt, EvaluationIdentity, fingerprint
from meta_harness import meta_harness_chess_puzzle as native


class VisualSearchBridgeTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'search'
        self.identity = EvaluationIdentity(*(fingerprint(k) for k in ('w', 'H', 'C', 'decode', 'v')), phase='fixture')
        self.archive = []
        for name, score, correctness in [('h0', .5, ((1, 1), (1, 1))), ('h1', .75, ((1, 0), (0, 1)))]:
            path = self.run / 'harnesses' / name / 'harness.py'
            path.parent.mkdir(parents=True)
            path.write_text((bridge.ROOT / 'ours/visual_harnesses/canonical_answer_harness.py').read_text() + '\n# ' + name)
            candidate = CandidateMetrics(name, bridge.file_sha256(path), score, 2.)
            self.archive.append({'candidate': candidate, 'audit': AuditReceipt(self.identity,
                candidate.harness_sha256, 'C', ('p0', 'p1'), correctness)})
        self.frontier = {'_best': {'harness': 'h1'}}
        self.rows = [{'harness': 'h1', 'avg_success_rate': 1., 'avg_mean_turn_count': 2.}]

    def test_all_acceptance_paths_and_resume_tamper(self):
        adapter = WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance())
        for stage in ('initial', 'ordinary', 'early_stop'):
            with self.subTest(stage=stage):
                archive = self.archive[:1] if stage == 'initial' else self.archive
                result = bridge.boundary(adapter, native, self.run, archive, self.identity,
                    stage=stage, frontier=self.frontier, rows=self.rows, valid_names=['h1'])
                self.assertEqual(result['accepted_harness'], 'h0')
                self.assertEqual(result['stop']['training_harness'], 'h0')
                restored = bridge.boundary(adapter, native, self.run, archive, self.identity,
                    stage='resume', frontier=self.frontier, rows=self.rows, valid_names=['h1'], resume=result['receipt'])
                self.assertEqual(restored['accepted_harness'], 'h0')
                changed = json.loads(json.dumps(result['receipt']))
                changed['decision']['selected'] = 'h1'
                with self.assertRaises(ValueError):
                    bridge.boundary(adapter, native, self.run, archive, self.identity, stage='resume',
                        frontier=self.frontier, rows=self.rows, valid_names=['h1'], resume=changed)

    def test_partition_cardinality_uses_frozen_manifest_scale(self):
        for pairs in (64, 256):
            with self.subTest(pairs=pairs):
                plan = {'partitions': {'H': {'examples': 128}, 'C': {'examples': 2 * pairs}}}
                self.assertEqual(bridge.partition_example_count(plan, 'H', 128), 128)
                self.assertEqual(bridge.partition_example_count(plan, 'C', 2 * pairs, pair_count=pairs), 2 * pairs)
                with self.assertRaises(ValueError):
                    bridge.partition_example_count(plan, 'C', 2 * pairs - 1, pair_count=pairs)
                with self.assertRaises(ValueError):
                    bridge.partition_example_count(plan, 'C', 2 * pairs, pair_count=pairs + 1)

    def test_off_keeps_native_early_stop_and_never_reads_c(self):
        adapter = WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance('off'))
        frontier = {'_best': {'harness': 'h0'}}
        for stage in ('initial', 'ordinary', 'early_stop', 'resume'):
            with self.subTest(stage=stage):
                value = bridge.boundary(adapter, native, self.run, None, self.identity,
                    stage=stage, frontier=frontier, rows=self.rows, valid_names=['h1'])
                self.assertEqual(value['accepted_harness'], 'h0' if stage == 'initial' else 'h1')
                self.assertEqual(value['receipt']['audit_calls'], 0)

    def test_public_feedback_excludes_private_fields(self):
        item = {**self.archive[0], 'h': {'accuracy': .5, 'mean_native_turns': 2.,
            'records': [{'sample_id': 'private-source-id', 'raw_answer': 'A', 'correct': 1,
                         'private_C': 'NEVER_PUBLISH'}]},
            'h_rows': [{'visual_sample_id': 'private-source-id', 'prompt': [{'content': '<image>\nWhich bar?'}],
                        'reward_model': {'ground_truth': 'A'}, 'private_path': '/SECRET'}]}
        bridge.public_h_feedback(self.run, item)
        output = (self.run / 'logs/H/h0/qwen35-4b/feedback.json').read_text()
        self.assertIn('Which bar?', output)
        for private in ('private-source-id', 'NEVER_PUBLISH', '/SECRET', 'correctness', 'pair_ids'):
            self.assertNotIn(private, output)

    def test_native_loop_ordinary_and_early_stop_both_gate_before_handoff(self):
        for score in (.75, 1.):
            with self.subTest(score=score), TemporaryDirectory() as folder:
                root = Path(folder)
                run = root / 'search'
                for name in ('h0', 'h1'):
                    target = run / 'harnesses' / name / 'harness.py'
                    target.parent.mkdir(parents=True)
                    target.write_bytes((self.run / 'harnesses' / name / 'harness.py').read_bytes())
                (run / 'logs/claude_sessions').mkdir(parents=True)
                bridge.write_new(root / 'native-config.json', {'task_profiles': {'H': {}}, 'models': [], 'seeds': [42]})
                plan = {'kind': 'fixture', 'mode': 'paired', 'epsilon': 0., 'identity': asdict(self.identity),
                        'baseline': {'plan': 'base', 'allocation': 'alloc'}, 'limitations': ['CPU fixture']}
                bridge.write_new(root / 'plan.json', plan)
                source = root / 'source-result'
                source.mkdir()
                bridge.write_new(source / 'result.json', {'status': 'CPU_FIXTURE'})
                archive = [{**x, 'plan': {'output': str(source)}, 'result': {'identity': asdict(self.identity),
                    'repeatability_passed': True}} for x in self.archive]
                archive[1]['candidate'] = CandidateMetrics('h1', archive[1]['candidate'].harness_sha256, score, 2.)
                def certify(path, allocation):
                    return archive[0 if str(path) == 'base' else 1]
                def publish(run, item):
                    path = run / 'logs/H' / item['candidate'].name / 'qwen35-4b'
                    path.mkdir(parents=True, exist_ok=True)
                    bridge.once(path / 'val.json', {'success_rate': item['candidate'].accuracy,
                        'avg_reward': item['candidate'].accuracy, 'mean_turn_count': 2.})
                with patch.object(bridge, 'load', return_value=plan), patch.object(bridge, 'certify', certify), \
                     patch.object(bridge, 'public_h_feedback', publish), patch.object(bridge, 'load_isolated_visual_harness'), \
                     patch.object(native.claude_wrapper, 'parse_stream_events', return_value=SimpleNamespace(show=lambda: None)):
                    bridge.advance(root)
                    self.assertTrue((root / 'proposal-request.json').exists())
                    session = run / 'logs/claude_sessions/fixture'
                    session.mkdir()
                    (session / 'events.jsonl').write_text('{}\n')
                    bridge.write_new(session / 'meta.json', {'duration_seconds': 0, 'exit_code': 0, 'cwd': str(run)})
                    bridge.write_new(run / 'pending_eval.json', {'candidates': [{'name': 'h1'}]})
                    bridge.write_new(root / 'proposal-ready.json', {'request_sha256': bridge.file_sha256(root / 'proposal-request.json'),
                        'imported_sha256': {'pending_eval.json': bridge.file_sha256(run / 'pending_eval.json')},
                        'session': str(session), 'session_sha256': bridge.tree_hashes(session)})
                    bridge.advance(root)
                    self.assertTrue((root / 'evaluation-request.json').exists())
                    self.assertFalse((root / 'accepted-for-training.json').exists())
                    bridge.advance(root, Path('candidate'), Path('allocation'))
                    selected = bridge.read(root / 'accepted-for-training.json')
                    self.assertEqual(selected['upstream_harness'], 'h1')
                    self.assertEqual(selected['accepted_harness'], 'h0')
                    self.assertEqual(selected['stop']['stage'], 'early_stop' if score == 1 else 'ordinary')
                    self.assertEqual(bridge.read(root / 'acceptance-resume.json')['accepted_harness'], 'h0')


if __name__ == '__main__':
    unittest.main()
