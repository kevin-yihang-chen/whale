"""Exercise the released merger with a tiny synthetic Qwen3.5 checkpoint.

This validates the tensor export/reload API, including tied names and the native
BF16 cast. Tokenizer/processor export and actual4B training remain separate checks.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Requires native training runtime')
class NativeCheckpointExportTests(unittest.TestCase):
    def test_original_fsdp_merger_roundtrips_synthetic_conditional_model(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import torch
        from safetensors import safe_open
        from transformers import AutoModelForImageTextToText, Qwen3_5Config
        from verl.model_merger.base_model_merger import ModelMergerConfig
        from verl.model_merger.fsdp_model_merger import FSDPModelMerger

        torch.manual_seed(17)
        torch.set_num_threads(1)
        config = Qwen3_5Config(
            text_config={'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64,
                         'num_hidden_layers': 1, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                         'head_dim': 8, 'layer_types': ['full_attention'], 'max_position_embeddings': 128,
                         'tie_word_embeddings': True, 'eos_token_id': 2,
                         'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.,
                                             'partial_rotary_factor': 1., 'mrope_section': [1, 1, 2]}},
            vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64,
                           'num_heads': 4, 'out_hidden_size': 32, 'num_position_embeddings': 16,
                           'patch_size': 2, 'temporal_patch_size': 1, 'spatial_merge_size': 2},
            tie_word_embeddings=True, image_token_id=60, video_token_id=61,
            vision_start_token_id=62, vision_end_token_id=63)
        config.architectures = ['Qwen3_5ForConditionalGeneration']
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
        original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix='whale-checkpoint-export-') as root:
            root = Path(root)
            source, target = root / 'native', root / 'exported'
            hf_config = source / 'huggingface'
            hf_config.mkdir(parents=True)
            config.save_pretrained(hf_config)
            model.generation_config.save_pretrained(hf_config)
            (source / 'fsdp_config.json').write_text(json.dumps({'world_size': 1}))
            torch.save(original, source / 'model_world_size_1_rank_0.pt')
            merger = FSDPModelMerger(ModelMergerConfig(operation='merge', backend='fsdp',
                local_dir=str(source), target_dir=str(target), hf_model_config_path=str(hf_config)))
            with patch('verl.model_merger.base_model_merger.hf_processor', return_value=None), \
                 patch('verl.model_merger.base_model_merger.hf_tokenizer', return_value=None):
                merger.merge_and_save()
            stored = {}
            for path in target.glob('*.safetensors'):
                with safe_open(str(path), framework='pt', device='cpu') as reader:
                    stored.update({name: reader.get_tensor(name) for name in reader.keys()})
            self.assertTrue(stored)
            reloaded = AutoModelForImageTextToText.from_pretrained(target, local_files_only=True, dtype=torch.bfloat16)
            restored = reloaded.state_dict()
            self.assertEqual(set(original), set(restored))
            for name, expected in original.items():
                torch.testing.assert_close(restored[name], expected.bfloat16(), rtol=0, atol=0)
            alias, embedding = 'lm_head.weight', 'model.language_model.embed_tokens.weight'
            self.assertEqual(restored[alias].data_ptr(), restored[embedding].data_ptr())
            if alias in stored:
                torch.testing.assert_close(stored[alias], stored[embedding], rtol=0, atol=0)
            print(json.dumps({'kind': 'synthetic_native_checkpoint_export', 'status': 'PASS',
                              'active_names': len(original), 'stored_names': len(stored),
                              'head_alias_serialized': alias in stored, 'tied_after_reload': True,
                              'new_model_calls': 0, 'actual_checkpoint_updates': 0,
                              'scope': 'Tiny synthetic tensor fixture; processor/tokenizer export replaced by None.'}))
