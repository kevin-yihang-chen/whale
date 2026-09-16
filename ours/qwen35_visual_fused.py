"""Forward visual evidence through E4's existing fused token likelihood.

The pinned WHALE generic dense backend consumes image kwargs without forwarding
them. This Qwen3.5 adapter calls its native multimodal backbone and retains
WHALE's FusedLinearForPPO, next-token alignment and external RSFT loss/mask.
It is a shared model compatibility repair, not the E2--E3 acceptance mechanism.

Fused output construction follows WHALE/verl dense_common.py (Apache-2.0,
Copyright 2024 Bytedance Ltd. and/or its affiliates).
"""
import torch


def forward_with_visual_evidence(
    self, input_ids=None, attention_mask=None, position_ids=None,
    past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
    output_attentions=None, output_hidden_states=None, return_dict=None,
    cache_position=None, logits_to_keep=0, temperature=1.0,
    pixel_values=None, image_grid_thw=None, pixel_values_videos=None,
    video_grid_thw=None, mm_token_type_ids=None, images_seqlens=None, **kwargs,
):
    from verl.models.transformers.dense_common import CausalLMOutputForPPO
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    if labels is not None:
        raise ValueError('Visual fused RSFT uses input_ids and an external response_mask; labels are unsupported')
    if input_ids is None:
        raise ValueError('Visual fused token likelihood requires input_ids')
    if not return_dict or logits_to_keep != 0:
        raise ValueError('Visual fused training requires full token positions and return_dict=True')
    if kwargs:
        raise ValueError(f'Unreviewed fused visual arguments: {sorted(kwargs)}')
    if (pixel_values is None) != (image_grid_thw is None):
        raise ValueError('Image pixels and grid must be provided together')
    if (pixel_values_videos is None) != (video_grid_thw is None):
        raise ValueError('Video pixels and grid must be provided together')
    if images_seqlens is not None:
        # Native agent-loop FLOP-accounting metadata, not a HF model argument.
        if image_grid_thw is None or not torch.equal(images_seqlens,
                torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])):
            raise ValueError('Native image sequence lengths differ from image grid')
    if input_ids is not None:
        if (input_ids == self.config.image_token_id).any() and pixel_values is None:
            raise ValueError('Image placeholders reached training without pixels')
        if (input_ids == self.config.video_token_id).any() and pixel_values_videos is None:
            raise ValueError('Video placeholders reached training without pixels')
    outputs = self.model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds,
        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids, use_cache=use_cache,
        output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        return_dict=True, cache_position=cache_position,
    )
    log_probs, entropy = FusedLinearForPPO().forward(
        hidden_states=outputs[0], vocab_weights=self.lm_head.weight,
        input_ids=torch.roll(input_ids, shifts=-1, dims=-1), temperature=temperature,
    )
    return CausalLMOutputForPPO(
        log_probs=log_probs, entropy=entropy, past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions,
    )


def install_visual_fused_backend():
    """Intercept only Qwen3.5 conditional-generation's torch fused backend."""
    import verl.models.transformers.monkey_patch as native
    original = native.patch_forward_with_backends
    if getattr(original, '_visual_evidence_forward', False):
        return

    def dispatch(model, use_fused_kernels=False, fused_kernels_backend=None):
        if model.config.model_type == 'qwen3_5' and use_fused_kernels:
            if (fused_kernels_backend != 'torch' or
                    model.__class__.__name__ != 'Qwen3_5ForConditionalGeneration'):
                raise ValueError('Unreviewed Qwen3.5 fused model/backend')
            model.__class__.forward = forward_with_visual_evidence
            print('Qwen3.5 visual backbone with native WHALE fused token likelihood')
            return
        return original(model, use_fused_kernels, fused_kernels_backend)

    dispatch._visual_evidence_forward = True
    native.patch_forward_with_backends = dispatch
