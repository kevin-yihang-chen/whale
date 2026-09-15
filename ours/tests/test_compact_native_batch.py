"""Check E4 tensor preservation and original actor loss/gradient equivalence."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Requires native training runtime')
class CompactNativeBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()

    def fixture(self):
        import torch
        from verl import DataProto
        torch.set_num_threads(1)
        prompts = torch.tensor([[0, 0, 4, 5, 6], [0, 0, 0, 7, 8]])
        responses = torch.tensor([[9, 10, 11, 12, 0, 0], [13, 14, 15, 0, 0, 0]])
        ids = torch.cat([prompts, responses], -1)
        attention = (ids != 0).long()
        loss_mask = (responses != 0).long()
        loss_mask[0, 1] = 0  # observed tool token, retained in attention and context
        pos = (attention.cumsum(-1) - 1).clamp(min=0).unsqueeze(1).repeat(1, 4, 1)
        return DataProto.from_dict(tensors=dict(prompts=prompts, responses=responses,
            response_mask=loss_mask, input_ids=ids, attention_mask=attention, position_ids=pos,
            dummy_tensor=torch.zeros(2, 1, dtype=torch.uint8)))

    def test_preserves_tokens_positions_and_does_not_mutate(self):
        import torch
        from ours.compact_native_batch import compact_batch
        batch = self.fixture()
        compact, report = compact_batch(batch)
        self.assertEqual((report['prompt_after'], report['response_after']), (5, 4))
        for i in range(2):
            old_mask, new_mask = batch.batch['attention_mask'][i].bool(), compact.batch['attention_mask'][i].bool()
            for key in ['input_ids', 'position_ids']:
                torch.testing.assert_close(batch.batch[key][i][..., old_mask], compact.batch[key][i][..., new_mask], rtol=0, atol=0)
        compact.batch['input_ids'][0, 2] = 50
        self.assertEqual(batch.batch['input_ids'][0, 2].item(), 4)

    def test_rejects_loss_on_padding_and_unknown_fields(self):
        from ours.compact_native_batch import compact_batch
        batch = self.fixture()
        batch.batch['response_mask'][0, -1] = 1
        with self.assertRaisesRegex(ValueError, 'Loss mask'):
            compact_batch(batch)
        batch = self.fixture()
        batch.batch['unreviewed'] = batch.batch['responses'].clone()
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            compact_batch(batch)
        batch = self.fixture()
        batch.batch['dummy_tensor'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            compact_batch(batch)

    def test_hybrid_qwen_original_actor_loss_and_gradients(self):
        import torch
        from transformers import AutoModelForImageTextToText, Qwen3_5Config
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        from ours.compact_native_batch import compact_batch
        torch.manual_seed(31)
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
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32, attn_implementation='sdpa')
        model.train()
        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module, actor.device_name, actor.param_dtype = model, 'cpu', torch.float32
        actor.use_prefix_grouper = actor.use_remove_padding = actor.use_fused_kernels = False
        actor.config = {}
        batch = self.fixture()
        compact, _ = compact_batch(batch)
        values = []
        for data in [batch, compact]:
            model.zero_grad(set_to_none=True)
            total = data.batch['response_mask'].sum()
            loss_value = 0.
            for micro in data.split(1):
                lp = actor._forward_micro_batch({k: v for k, v in micro.batch.items()}, temperature=1.)['log_probs']
                loss = -(lp * micro.batch['response_mask']).sum() / total
                loss.backward()
                loss_value += loss.detach()
            values.append((loss_value, {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(values[0][0], values[1][0], rtol=2e-6, atol=2e-6)
        self.assertEqual(set(values[0][1]), set(values[1][1]))
        for name, grad in values[0][1].items():
            torch.testing.assert_close(grad, values[1][1][name], rtol=2e-5, atol=2e-6, msg=name)


if __name__ == '__main__':
    unittest.main()
