"""Prevent serialization-only changes from being counted as training progress."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ours.check_checkpoint_transition import compare


@unittest.skipUnless(importlib.util.find_spec('safetensors') and importlib.util.find_spec('torch'),
                     'Requires the native tensor runtime')
class CheckpointTransitionTests(unittest.TestCase):
    def checkpoints(self, root, modified=False):
        import torch
        from safetensors.torch import save_file
        a, b = Path(root) / 'base', Path(root) / 'updated'
        for i, p in enumerate((a, b)):
            p.mkdir()
            (p / 'config.json').write_text(json.dumps({'model_type': 'test_fixture', 'export_label': i}))
        save_file({'weight': torch.tensor([1., 2.], dtype=torch.bfloat16)}, str(a / 'model.safetensors'))
        save_file({'weight': torch.tensor([1., 2.25 if modified else 2.], dtype=torch.float32)},
                  str(b / 'model.safetensors'))
        return a, b

    def test_metadata_and_exact_dtype_conversion_are_not_an_update(self):
        with tempfile.TemporaryDirectory() as root:
            report = compare(*self.checkpoints(root))
            self.assertNotEqual(report['base']['weights_sha256'], report['updated']['weights_sha256'])
            self.assertEqual(report['status'], 'UNCHANGED')

    def test_measured_tensor_delta(self):
        with tempfile.TemporaryDirectory() as root:
            report = compare(*self.checkpoints(root, modified=True))
            self.assertEqual(report['status'], 'CHANGED')
            self.assertEqual(report['changed_elements'], 1)
            self.assertEqual(report['parameter_delta_l2'], .25)

    def test_nonfinite_export_is_rejected(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as root:
            a, b = self.checkpoints(root)
            save_file({'weight': torch.tensor([float('nan'), 2.])}, str(b / 'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                compare(a, b)
