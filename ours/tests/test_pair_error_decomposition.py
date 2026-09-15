import unittest

from ours.pair_error_decomposition import decompose


class PairErrorDecompositionTests(unittest.TestCase):
    def test_same_marginal_can_have_different_pair_accuracy(self):
        concentrated = decompose([(1, 1), (0, 0)])
        distributed = decompose([(1, 0), (0, 1)])
        self.assertEqual(concentrated['marginal_accuracy'], distributed['marginal_accuracy'])
        self.assertEqual(concentrated['paired_accuracy'], 0.5)
        self.assertEqual(distributed['paired_accuracy'], 0.0)
        self.assertNotEqual(concentrated['both_wrong_fraction'], distributed['both_wrong_fraction'])

    def test_no_both_wrong_pairs_and_invalid_rewards(self):
        result = decompose([(1, 1), (0, 1)])
        self.assertEqual(result['both_wrong'], 0)
        self.assertEqual(result['paired_accuracy'], 2 * result['marginal_accuracy'] - 1)
        for rows in ([], [(1, 2)], [(1,)], [(float('nan'), 0)]):
            with self.subTest(rows=rows), self.assertRaises(AssertionError):
                decompose(rows)


if __name__ == '__main__':
    unittest.main()
