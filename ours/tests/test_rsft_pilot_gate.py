"""Catch a transport limit that is invisible to ordinary model-loading checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from ours.rsft_pilot_gate import check_tensor_bucket


@unittest.skipUnless(importlib.util.find_spec('safetensors') and importlib.util.find_spec('torch'),
                     'Requires native tensor runtime')
class NativeTensorBucketTests(unittest.TestCase):
    def test_actor_precision_determines_size_not_stored_precision(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as root:
            save_file({'embedding': torch.zeros(300000, dtype=torch.bfloat16)},
                      str(Path(root) / 'model.safetensors'))
            report = check_tensor_bucket(root, 'bf16', 1)
            self.assertEqual(report['actor_tensor_bytes'], 600000)
            with self.assertRaisesRegex(ValueError, 'cannot hold'):
                check_tensor_bucket(root, 'fp32', 1)
            self.assertEqual(check_tensor_bucket(root, 'fp32', 2)['actor_tensor_bytes'], 1200000)

    def test_unknown_precision_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            check_tensor_bucket('.', 'auto', 3072)
