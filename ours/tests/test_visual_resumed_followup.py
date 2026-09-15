import unittest

import torch

from ours.verify_native_transition import summary, tensor_delta
from ours.visual_resumed_followup import combine_comparisons


class ResumedFollowupTests(unittest.TestCase):
    def comparison(self, before, after):
        return summary([{'name': 'weight', **tensor_delta(before, after)}])

    def test_bf16_rounding_residual_is_not_a_new_update(self):
        theta1 = torch.tensor([1.001], dtype=torch.float32)
        served = theta1.bfloat16()
        converted = {'native_fp32': self.comparison(served, theta1),
                     'exported_bf16': self.comparison(served, served),
                     'export_exact_native_bf16_cast': True}
        result = combine_comparisons(converted, self.comparison(theta1, theta1))
        self.assertEqual(result['native_fp32']['status'], 'UNCHANGED')
        self.assertEqual(result['native_vs_incoming_bf16_reference']['status'], 'CHANGED')
        self.assertEqual(result['exported_bf16']['status'], 'UNCHANGED')
        self.assertEqual(converted['native_fp32']['status'], 'CHANGED')

    def test_real_update_and_serving_change_remain_distinct(self):
        theta1, theta2 = torch.tensor([1.001]), torch.tensor([1.002])
        converted = {'native_fp32': self.comparison(theta1.bfloat16(), theta2),
                     'exported_bf16': self.comparison(theta1.bfloat16(), theta2.bfloat16()),
                     'export_exact_native_bf16_cast': True}
        actual = self.comparison(theta1, theta2)
        result = combine_comparisons(converted, actual)
        self.assertEqual(result['native_fp32'], actual)
        self.assertEqual(result['native_fp32']['status'], 'CHANGED')
        self.assertEqual(result['exported_bf16']['status'], 'UNCHANGED')
        bad = {**actual, 'tensor_count': 2}
        with self.assertRaises(AssertionError):
            combine_comparisons(converted, bad)


if __name__ == '__main__':
    unittest.main()
