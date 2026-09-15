"""Compare released dense and Torch fused E4 backends on a hybrid CPU model.

This is a random small-model numerical fixture, not task inference or GPU proof.
BF16 tolerances are declared before comparison; all controls must share a backend.
"""
import importlib.util
import json
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires native training runtime')
class NativeFusedRSFTTests(unittest.TestCase):
    def compare(self, dtype_name):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import torch
        from transformers import AutoModelForImageTextToText, Qwen3_5Config
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        from verl.models.transformers.monkey_patch import patch_forward_with_backends
        from ours.tests.test_compact_native_batch import CompactNativeBatchTests
        torch.manual_seed(31)
        torch.set_num_threads(1)
        dtype = getattr(torch, dtype_name)
        config = Qwen3_5Config(text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            layer_types=['linear_attention'] * 3 + ['full_attention'], max_position_embeddings=128,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_conv_kernel_dim=4, tie_word_embeddings=True,
            rope_parameters=dict(rope_type='default', rope_theta=10000., partial_rotary_factor=1., mrope_section=[1, 1, 2])),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                out_hidden_size=32, num_position_embeddings=16, patch_size=2, temporal_patch_size=1, spatial_merge_size=2),
            tie_word_embeddings=True, image_token_id=60, video_token_id=61,
            vision_start_token_id=62, vision_end_token_id=63)
        model = AutoModelForImageTextToText.from_config(config, dtype=dtype, attn_implementation='sdpa')
        model.train()
        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module, actor.device_name, actor.param_dtype = model, 'cpu', dtype
        actor.use_prefix_grouper = actor.use_remove_padding = actor.use_fused_kernels = False
        actor.config = {}
        batch = CompactNativeBatchTests().fixture()
        original, values = model.__class__.forward, []
        try:
            for fused in (False, True):
                if fused:
                    patch_forward_with_backends(model, use_fused_kernels=True, fused_kernels_backend='torch')
                actor.use_fused_kernels = fused
                model.zero_grad(set_to_none=True)
                value = 0.
                for micro in batch.split(1):
                    lp = actor._forward_micro_batch(dict(micro.batch), temperature=1.)['log_probs']
                    loss = -(lp * micro.batch['response_mask']).sum() / batch.batch['response_mask'].sum()
                    loss.backward()
                    value += float(loss.detach())
                values.append((value, {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}))
        finally:
            model.__class__.forward = original
        self.assertEqual(set(values[0][1]), set(values[1][1]))
        squared_diff = squared_base = 0.
        for name, grad in values[0][1].items():
            other = values[1][1][name]
            self.assertTrue(bool(torch.isfinite(grad).all() and torch.isfinite(other).all()))
            squared_diff += float((grad.double() - other.double()).square().sum())
            squared_base += float(grad.double().square().sum())
            if dtype == torch.float32:
                torch.testing.assert_close(grad, other, rtol=2e-5, atol=2e-6, msg=name)
        relative_gradient_l2 = (squared_diff / squared_base) ** .5
        if dtype == torch.float32:
            torch.testing.assert_close(values[0][0], values[1][0], rtol=2e-6, atol=2e-6)
        else:
            self.assertLessEqual(abs(values[0][0] - values[1][0]), 4 * torch.finfo(dtype).eps)
            self.assertLessEqual(relative_gradient_l2, 4 * torch.finfo(dtype).eps)
        print(json.dumps({'kind': 'native_fused_cpu_fixture', 'dtype': dtype_name,
              'losses': [v[0] for v in values], 'gradient_tensors': len(values[0][1]),
              'relative_gradient_l2': relative_gradient_l2, 'gpu_validation': False}), flush=True)

    def test_fp32_loss_and_each_parameter_gradient(self):
        self.compare('float32')

    def test_bf16_loss_and_aggregate_gradient_rounding(self):
        self.compare('bfloat16')


if __name__ == '__main__':
    unittest.main()
