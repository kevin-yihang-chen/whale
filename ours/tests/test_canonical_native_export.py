"""Check raw serialized active names, not only Transformers reload compatibility."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Requires native training runtime')
class CanonicalNativeExportTests(unittest.TestCase):
    def test_canonical_merger_preserves_raw_visual_keys_and_values(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import torch
        from transformers import AutoModelForImageTextToText, Qwen3_5Config
        from safetensors import safe_open
        from verl.model_merger.base_model_merger import ModelMergerConfig
        from ours.canonical_native_export import canonical_merger_class
        config = Qwen3_5Config(text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            layer_types=['full_attention'], max_position_embeddings=128, tie_word_embeddings=True,
            eos_token_id=2, rope_parameters=dict(rope_type='default', rope_theta=10000.,
                partial_rotary_factor=1., mrope_section=[1, 1, 2])),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                out_hidden_size=32, num_position_embeddings=16, patch_size=2, temporal_patch_size=1, spatial_merge_size=2),
            tie_word_embeddings=True, image_token_id=60, video_token_id=61,
            vision_start_token_id=62, vision_end_token_id=63)
        config.architectures = ['Qwen3_5ForConditionalGeneration']
        torch.manual_seed(17)
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix='whale-canonical-export-') as root:
            root = Path(root)
            base, native, exported = root / 'base', root / 'native', root / 'export'
            base.mkdir(); native.mkdir()
            config.save_pretrained(base)
            model.generation_config.save_pretrained(base)
            torch.save(state, native / 'model_world_size_1_rank_0.pt')
            (native / 'fsdp_config.json').write_text(json.dumps({'world_size': 1}))
            merger = canonical_merger_class()(ModelMergerConfig(operation='merge', backend='fsdp',
                local_dir=str(native), target_dir=str(exported), hf_model_config_path=str(base)))
            with patch('verl.model_merger.base_model_merger.hf_processor', return_value=None), \
                 patch('verl.model_merger.base_model_merger.hf_tokenizer', return_value=None):
                merger.merge_and_save()
            with safe_open(str(exported / 'model.safetensors'), framework='pt', device='cpu') as f:
                self.assertEqual(set(f.keys()), set(state))
                self.assertTrue(any(k.startswith('model.visual.') for k in f.keys()))
                self.assertFalse(any(k.startswith('model.language_model.visual.') for k in f.keys()))
                for name, expected in state.items():
                    torch.testing.assert_close(f.get_tensor(name), expected.bfloat16(), rtol=0, atol=0)
            reloaded = AutoModelForImageTextToText.from_pretrained(exported, local_files_only=True, dtype=torch.bfloat16)
            for name, expected in state.items():
                torch.testing.assert_close(reloaded.state_dict()[name], expected.bfloat16(), rtol=0, atol=0)

    def test_actual_native_config_changes_are_limited_to_reviewed_fields(self):
        from ours.canonical_native_export import validate_inference_config
        base = Path('data/models/qwen3.5-4b-851bf6e')
        native = Path('data/native-rsft/replay-222156/global_step_1/actor/huggingface')
        if not native.exists():
            self.skipTest('Actual native checkpoint not present')
        differences = validate_inference_config(base, native)
        self.assertIn('eos_token_id', {d['path'] for d in differences})
        with tempfile.TemporaryDirectory(prefix='whale-bad-native-config-') as root:
            changed = json.loads((native / 'config.json').read_text())
            changed['text_config']['rope_parameters']['rope_theta'] += 1
            (Path(root) / 'config.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Unreviewed model configuration difference'):
                validate_inference_config(base, Path(root))


if __name__ == '__main__':
    unittest.main()
