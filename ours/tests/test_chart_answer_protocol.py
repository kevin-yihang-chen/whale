import unittest
from types import SimpleNamespace

from ours.chart_answer_protocol import (SharedAnswerHarness, parse_answer, verify,
    encode_truth, disjoint_numeric_answers, number)
from ours.visual_evidence_reward import compute_score


class ChartAnswerTests(unittest.TestCase):
    def test_shared_adapter_survives_worker_serialization(self):
        import pickle
        adapter = SharedAnswerHarness(SimpleNamespace(sha256='worker-identity'))
        restored = pickle.loads(pickle.dumps(adapter))
        self.assertEqual(restored.sha256, 'worker-identity')

    def test_shared_parser_cannot_be_replaced_by_candidate(self):
        def forbidden(*args, **kwargs):
            raise AssertionError('Candidate parser must not run')
        harness = SharedAnswerHarness(SimpleNamespace(unchanged=lambda: None, invoke=forbidden))
        self.assertEqual(harness.invoke('parse_answer', text='Reasoning\nFinal answer: 1,250'), '1250')
        self.assertEqual(parse_answer('Answer: B.'), 'B')
        self.assertEqual(parse_answer('\\boxed{-2.50}'), '-2.5')
        self.assertNotEqual(parse_answer('A or B'), 'A')
        self.assertNotEqual(parse_answer('7 and 9'), '7')

    def test_numeric_boundaries_zero_and_data_errors(self):
        truth = encode_truth('numeric', '100')
        self.assertTrue(verify('105', truth))
        self.assertFalse(verify('105.00001', truth))
        self.assertTrue(verify('-105', encode_truth('numeric', '-100')))
        self.assertTrue(verify('0', encode_truth('numeric', '0')))
        self.assertFalse(verify('1', encode_truth('numeric', '0')))
        for value in ('nan', 'inf', '1.2.3', '1,23', '100 extra'):
            self.assertIsNone(number(value))
        with self.assertRaises(ValueError):
            verify('A', 'not encoded truth')
        self.assertFalse(disjoint_numeric_answers(100, 105))
        self.assertTrue(disjoint_numeric_answers(100, 120))

    def test_reward_matches_shared_evaluation_and_legacy_is_preserved(self):
        self.assertEqual(compute_score('visual_chart_shared_v1', '95', encode_truth('numeric', 100))['score'], 1.)
        self.assertEqual(compute_score('visual_chart_shared_v1', 'B', encode_truth('binary', 'A'))['score'], 0.)
        self.assertEqual(compute_score('visual_evidence_binary', 'A.', 'A')['score'], 1.)


if __name__ == '__main__':
    unittest.main()
