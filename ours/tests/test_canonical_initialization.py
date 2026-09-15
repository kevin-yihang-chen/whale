"""Exact untrained-cast audit rejects real value, graph and alias corruption."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires native runtime')
class CanonicalInitializationTests(unittest.TestCase):
    def test_serialized_cast_and_corruption(self):
        import torch
        from safetensors.torch import save_file
        from ours.canonical_initialization import audit_cast
        with tempfile.TemporaryDirectory() as tmp:
            base, target = Path(tmp)/'base', Path(tmp)/'target'
            base.mkdir(); target.mkdir()
            tensors = {'embedding': torch.tensor([[1., 2.]], dtype=torch.bfloat16),
                       'precise': torch.tensor([1.003, 2.007]), 'mtp': torch.ones(1)}
            save_file(tensors, base/'model.safetensors')
            findings = {'base_only_inactive_mtp_names': ['mtp'], 'alias': {'head': 'embedding'},
                        'independent_comparison_names': 2}
            expected = {k: v.bfloat16() for k, v in tensors.items() if k != 'mtp'}
            expected['head'] = expected['embedding'].clone()
            save_file(expected, target/'model.safetensors')
            report = audit_cast(base, target, findings)
            self.assertTrue(report['exact_source_bf16_cast'])
            self.assertEqual(report['original_fp32_elements'], 2)
            self.assertEqual(report['original_bf16_changed_elements'], 0)
            for defect in ('value', 'alias', 'missing', 'extra'):
                changed = {k: v.clone() for k, v in expected.items()}
                if defect == 'value': changed['precise'][0] += 1
                if defect == 'alias': changed['head'][0, 0] += 1
                if defect == 'missing': del changed['precise']
                if defect == 'extra': changed['unreviewed'] = torch.ones(1, dtype=torch.bfloat16)
                save_file(changed, target/'model.safetensors')
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    audit_cast(base, target, findings)

if __name__ == '__main__':
    unittest.main()
