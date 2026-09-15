"""Compare actual native chunked autograd across multiple chunk boundaries."""
import importlib.util
import json
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires native training runtime')
class NativeFusedChunkTests(unittest.TestCase):
    def test_512_and_128_token_chunks_preserve_loss_with_bounded_gradient_rounding(self):
        import torch
        import verl.utils.experimental.torch_functional as native
        from ours.native_fused_chunk import install_chunk_size
        torch.set_num_threads(1)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                torch.manual_seed(73)
                hidden = torch.randn(2, 777, 32).to(dtype) * .2
                weights = torch.randn(1024, 32).to(dtype) * .2
                labels = torch.randint(0, 1024, (2, 777))
                mask = torch.ones(2, 777)
                mask[:, :41] = 0
                mask[:, 2::7] = 0
                mask[1, 733:] = 0
                original, measurements = native.FusedLinearForPPO, []
                try:
                    for chunk in (512, 128):
                        if chunk == 128:
                            install_chunk_size(chunk)
                            install_chunk_size(chunk)
                        layer = native.FusedLinearForPPO()
                        self.assertEqual(layer.chunk_size, chunk)
                        h, w = hidden.clone().requires_grad_(), weights.clone().requires_grad_()
                        lp, _ = layer(h, w, labels, temperature=.7)
                        loss = -(lp * mask).sum() / mask.sum()
                        loss.backward()
                        self.assertTrue(torch.isfinite(h.grad).all() and torch.isfinite(w.grad).all())
                        measurements.append((float(loss.detach()), h.grad.clone(), w.grad.clone()))
                    difference = sum(float((a.double() - b.double()).square().sum())
                                     for a, b in zip(measurements[0][1:], measurements[1][1:]))
                    magnitude = sum(float(g.double().square().sum()) for g in measurements[0][1:])
                    relative = (difference / magnitude) ** .5
                    if dtype == torch.float32:
                        for a, b in zip(measurements[0], measurements[1]):
                            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
                    else:
                        self.assertLessEqual(abs(measurements[0][0] - measurements[1][0]), 4 * torch.finfo(dtype).eps)
                        self.assertLessEqual(relative, 4 * torch.finfo(dtype).eps)
                    print(json.dumps({'kind': 'native_fused_chunk_cpu_fixture', 'dtype': str(dtype),
                        'tokens': 1554, 'chunks': [512, 128], 'losses': [v[0] for v in measurements],
                        'relative_gradient_l2': relative, 'gpu_validation': False}), flush=True)
                    with self.assertRaisesRegex(ValueError, 'override differs'):
                        native.FusedLinearForPPO(chunk_size=512)
                finally:
                    native.FusedLinearForPPO = original


if __name__ == '__main__':
    unittest.main()
