import unittest
from ours.fast_chart_augmentation import replacement_map


class AugmentationTests(unittest.TestCase):
    def test_half_budget_uses_complete_distinct_pairs_and_retains_order(self):
        batches=[[f'w{i*8+j}' for j in range(8)] for i in range(4)]
        mapping,rows=replacement_map(batches,[f'c{i}' for i in range(64)],42)
        self.assertEqual(len(mapping),16)
        self.assertEqual(len(set(mapping.values())),16)
        for batch in batches:
            self.assertFalse(set(batch[:4])&set(mapping))
            self.assertEqual(set(batch[4:])&set(mapping),set(batch[4:]))
        pairs={r['pair_id'] for r in rows};self.assertEqual(len(pairs),8)
        for pair in pairs:self.assertEqual({r['side'] for r in rows if r['pair_id']==pair},{0,1})
        self.assertEqual((mapping,rows),replacement_map(batches,[f'c{i}' for i in range(64)],42))
        self.assertNotEqual(mapping,replacement_map(batches,[f'c{i}' for i in range(64)],43)[0])


if __name__=='__main__':unittest.main()
