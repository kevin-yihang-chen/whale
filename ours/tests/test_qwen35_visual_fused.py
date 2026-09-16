"""Real tiny Qwen3.5: reproduce lost pixels and check E4 gradient equivalence."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from verl.models.transformers.dense_common import forward_with_torch_backend
from ours.qwen35_visual_fused import forward_with_visual_evidence


def fixture():
    torch.manual_seed(42)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, layer_types=['full_attention'], use_cache=False,
            rope_parameters=dict(rope_type='default', rope_theta=10000.,
                partial_rotary_factor=1., mrope_section=[2, 3, 3])),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
            num_heads=4, patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
            out_hidden_size=32, num_position_embeddings=16),
        image_token_id=60, video_token_id=63, vision_start_token_id=61,
        vision_end_token_id=62, tie_word_embeddings=False)
    config._attn_implementation = 'eager'
    model = Qwen3_5ForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 60, 2, 3, 4]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(5).view(1, 1, 5).expand(3, 1, 5),
        pixel_values=torch.randn(4, 12), image_grid_thw=torch.tensor([[1, 2, 2]]),
        use_cache=False, return_dict=True)
    return model, inputs


class VisualFusedTest(unittest.TestCase):
    def test_unsupported_labels_or_missing_ids_fail_before_backbone(self):
        model, inputs = fixture()
        labels = inputs['input_ids'].clone()
        labels[:, :2] = -100
        with patch.object(model.model, 'forward', side_effect=AssertionError('Backbone must not run')):
            for supplied in (labels, inputs['input_ids']):
                with self.subTest(labels=supplied.tolist()), self.assertRaisesRegex(ValueError, 'labels are unsupported'):
                    forward_with_visual_evidence(model, **inputs, labels=supplied)
            with self.assertRaisesRegex(ValueError, 'requires input_ids'):
                forward_with_visual_evidence(model, **(inputs | {'input_ids': None}))

    def test_original_drops_pixels_and_vision_gradient(self):
        model, inputs = fixture()
        seen = []
        hook = model.model.visual.register_forward_pre_hook(lambda *_: seen.append(True))
        first = forward_with_torch_backend(model, **inputs).log_probs
        second = forward_with_torch_backend(model, **(inputs | {'pixel_values': -inputs['pixel_values']})).log_probs
        self.assertTrue(torch.equal(first, second))
        (-first[:, -2].mean()).backward()
        self.assertEqual(seen, [])
        self.assertTrue(all(p.grad is None for p in model.model.visual.parameters()))
        hook.remove()

    def test_fixed_pixels_likelihood_and_gradient_match_native_hf(self):
        model, inputs = fixture()
        seen = []
        hook = model.model.visual.register_forward_pre_hook(lambda *_: seen.append(True))
        logits = model(**inputs).logits
        expected = logits.log_softmax(-1).gather(-1, inputs['input_ids'].roll(-1, -1).unsqueeze(-1)).squeeze(-1)
        (-expected[:, -2].mean()).backward()
        reference_grad = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        fixed = forward_with_visual_evidence(model, **inputs).log_probs
        torch.testing.assert_close(fixed, expected, atol=1e-6, rtol=1e-5)
        (-fixed[:, -2].mean()).backward()
        for name, param in model.named_parameters():
            if name in reference_grad:
                torch.testing.assert_close(param.grad, reference_grad[name], atol=1e-6, rtol=1e-4)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.visual.parameters()))
        changed = forward_with_visual_evidence(model, **(inputs | {'pixel_values': -inputs['pixel_values']})).log_probs
        self.assertFalse(torch.equal(fixed[:, -2], changed[:, -2]))
        self.assertEqual(len(seen), 3)
        hook.remove()

    def test_missing_image_payload_fails_closed(self):
        model, inputs = fixture()
        inputs.pop('pixel_values')
        inputs.pop('image_grid_thw')
        with self.assertRaisesRegex(ValueError, 'without pixels'):
            forward_with_visual_evidence(model, **inputs)

    def test_native_sequence_metadata_preserves_images_and_checks_grid(self):
        from verl.utils.model import extract_multi_modal_inputs
        model, inputs = fixture()
        image = {key:inputs[key] for key in ('pixel_values','image_grid_thw')}
        image['images_seqlens'] = torch.tensor([4])
        actual = extract_multi_modal_inputs([image])
        expected = forward_with_visual_evidence(model, **inputs).log_probs
        result = forward_with_visual_evidence(model, **(inputs | actual)).log_probs
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'sequence lengths'):
            forward_with_visual_evidence(model, **(inputs | {'images_seqlens': torch.tensor([3])}))

    def test_observer_records_consumed_pixels_and_matching_gradient(self):
        from ours.visual_pixel_bootstrap import observed_forward
        model, inputs = fixture()
        with TemporaryDirectory() as directory, patch.dict('os.environ', {'VETO_VISUAL_FORWARD_AUDIT': directory}):
            out = observed_forward(model, **inputs)
            (-out.log_probs[:, -2].mean()).backward()
            rows = [json.loads(line) for p in Path(directory).glob('*.jsonl') for line in p.read_text().splitlines()]
            self.assertEqual([r['kind'] for r in rows], ['visual_backbone_input', 'visual_feature_gradient'])
            self.assertEqual(rows[0]['event'], rows[1]['event'])
            self.assertEqual(rows[0]['actor_pixels'], rows[0]['consumed_pixels'])
            self.assertGreater(rows[1]['nonzero_elements'], 0)


if __name__ == '__main__':
    unittest.main()
