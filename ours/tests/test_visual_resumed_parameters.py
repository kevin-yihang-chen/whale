import unittest

import torch

from ours.verify_visual_resumed_parameters import compare_states


class NativeResumeComparisonTests(unittest.TestCase):
    def test_actual_fp32_update_and_unchanged_outcome(self):
        before = {'weight': torch.tensor([1., 2.]), 'alias': torch.tensor([1., 2.])}
        same = compare_states(before, before, {'alias': 'weight'}, 1)
        self.assertEqual(same['status'], 'UNCHANGED')
        after = {k: v + torch.tensor([0., 0.25]) for k, v in before.items()}
        changed = compare_states(before, after, {'alias': 'weight'}, 1)
        self.assertEqual(changed['changed_elements'], 1)
        self.assertEqual(changed['parameter_delta_l2'], 0.25)

    def test_alias_graph_dtype_and_nonfinite_corruption_fail(self):
        before = {'weight': torch.tensor([1., 2.]), 'alias': torch.tensor([1., 2.])}
        corruptions = [
            {'weight': before['weight']},
            {**before, 'unexpected': torch.tensor([1.])},
            {**before, 'alias': torch.tensor([1., 3.])},
            {k: torch.tensor([1., 2., 3.]) for k in before},
            {k: v.bfloat16() for k, v in before.items()},
            {k: torch.tensor([float('nan'), 2.]) for k in before},
        ]
        for after in corruptions:
            with self.subTest(after=after), self.assertRaises(ValueError):
                compare_states(before, after, {'alias': 'weight'}, 1)
        for count in (0, 2, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                compare_states(before, before, {'alias': 'weight'}, count)


if __name__ == '__main__':
    unittest.main()
