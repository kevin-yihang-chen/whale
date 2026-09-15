"""Reject unreviewed graph edits and distinguish updates from serialization casts."""
import importlib.util
import unittest

from ours.verify_native_transition import comparison_names


class ReviewedGraphTests(unittest.TestCase):
    def test_explicit_inactive_keys_and_optional_alias(self):
        self.assertEqual(comparison_names({'a', 'b', 'mtp'}, {'a', 'b', 'head'}, {'mtp'}, {'head': 'a'}), ['a', 'b'])
        self.assertEqual(comparison_names({'a', 'b', 'mtp'}, {'a', 'b'}, {'mtp'}, {'head': 'a'}), ['a', 'b'])
        for invalid in [{'a', 'head'}, {'a', 'b', 'unknown'}, {'a', 'b', 'mtp'}]:
            with self.assertRaisesRegex(ValueError, 'Unreviewed'):
                comparison_names({'a', 'b', 'mtp'}, invalid, {'mtp'}, {'head': 'a'})


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires tensor runtime')
class ReviewedValueTests(unittest.TestCase):
    def test_cast_is_not_update_and_rounding_is_separate(self):
        import torch
        from ours.verify_native_transition import tensor_delta
        a = torch.tensor([1., 2., 0.], dtype=torch.bfloat16)
        self.assertEqual(tensor_delta(a, a.float())['changed_elements'], 0)
        b = a.float() + torch.tensor([1e-7, 0., 1e-7])
        self.assertEqual(tensor_delta(a, b)['changed_elements'], 2)
        self.assertEqual(tensor_delta(a, b.bfloat16(), native=b)['changed_elements'], 1)
        with self.assertRaisesRegex(ValueError, 'Export differs'):
            tensor_delta(a, a, native=b)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            tensor_delta(a, torch.full_like(a, float('nan')))

    def test_alias_values_are_checked(self):
        import torch
        from ours.verify_native_transition import verify_aliases
        state = {'embedding': torch.ones(2), 'head': torch.zeros(2)}
        with self.assertRaisesRegex(ValueError, 'alias values'):
            verify_aliases(state, {'head': 'embedding'}, state.__getitem__)


if __name__ == '__main__':
    unittest.main()
