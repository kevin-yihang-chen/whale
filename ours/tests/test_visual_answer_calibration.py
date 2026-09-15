from pathlib import Path
import unittest

from ours.isolated_visual_harness import load_isolated_visual_harness
from ours.visual_task import binary_answer_verifier


class VisualAnswerCalibrationTest(unittest.TestCase):
    def test_explicit_final_answer_without_tool_or_ambiguous_labels(self):
        harness = load_isolated_visual_harness(Path('ours/visual_harnesses/canonical_answer_harness.py'))
        for raw, target in [('A','A'),('Bar B is taller.\n\nB','B'),('Explanation.\n a.','A')]:
            parsed = harness.invoke('parse_answer', text=raw)
            self.assertEqual(parsed,target)
            self.assertTrue(binary_answer_verifier(parsed,target))
        for raw in ('A\nB','B\nA','<tool_call>\nA','Answer could be A or B.','A\nNot sure.'):
            parsed = harness.invoke('parse_answer', text=raw)
            self.assertFalse(any(binary_answer_verifier(parsed,t) for t in ('A','B')))


if __name__ == '__main__':
    unittest.main()
