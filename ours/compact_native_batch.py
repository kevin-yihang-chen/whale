"""Remove only globally masked response suffixes before the original E4 actor.

The response SFT objective remains -sum(m_t log p_t) / sum(m_t). No valid
token, position, observation, accepted row or optimizer minibatch is removed.
This text-only execution adapter is shared infrastructure, outside E1-E3.
"""
from copy import deepcopy


def compact_batch(data):
    import torch
    from verl import DataProto

    tensors = data.batch
    prompt_keys = {'prompts'}
    response_keys = {'responses', 'response_mask', 'rm_scores', 'token_level_scores',
                     'token_level_rewards', 'advantages', 'returns'}
    sequence_keys = {'input_ids', 'attention_mask', 'position_ids'}
    placeholder_keys = {'dummy_tensor'}
    unknown = set(tensors.keys()) - prompt_keys - response_keys - sequence_keys - placeholder_keys
    if unknown:
        raise ValueError(f'Unreviewed tensor fields: {sorted(unknown)}')
    if any(bool(x) for x in data.non_tensor_batch.get('multi_modal_inputs', [])):
        raise ValueError('This adapter is restricted to text-only batches')
    p, r = tensors['prompts'].shape[-1], tensors['responses'].shape[-1]
    mask = tensors['attention_mask']
    if mask.shape != (len(data), p + r) or not len(data):
        raise ValueError('Invalid sequence layout')
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('Attention mask must be binary')
    if not torch.all(mask[:, p - 1]) or not torch.all(mask[:, p]):
        raise ValueError('Each response must follow an attended prompt token')
    if torch.any(tensors['response_mask'].bool() & ~mask[:, p:].bool()):
        raise ValueError('Loss mask includes padding')
    if not torch.equal(tensors['input_ids'], torch.cat([tensors['prompts'], tensors['responses']], dim=-1)):
        raise ValueError('Input token layout differs')
    # Preserve the prompt margin: hybrid recurrent models are not assumed to be
    # invariant to changes in their left padding. Only causal suffixes are trimmed.
    attended = mask.bool().any(dim=0).nonzero().flatten()
    left, end = 0, int(attended[-1]) + 1
    response_end = end - p
    compact = {}
    for key, value in tensors.items():
        if key in placeholder_keys:
            if value.shape != (len(data), 1) or value.dtype != torch.uint8 or torch.any(value):
                raise ValueError('Native dataset placeholder must be zero uint8 [N,1]')
            compact[key] = value.clone()
            continue
        expected = p if key in prompt_keys else r if key in response_keys else p + r
        if value.shape[-1] != expected:
            raise ValueError(f'Invalid time axis for {key}')
        if key in prompt_keys:
            selected = value[..., left:]
        elif key in response_keys:
            if torch.any(value[..., response_end:]) and key != 'responses':
                raise ValueError(f'Nonzero discarded values in {key}')
            selected = value[..., :response_end]
        else:
            selected = value[..., left:end]
        compact[key] = selected.clone()
    result = DataProto.from_dict(tensors=compact, non_tensors=deepcopy(data.non_tensor_batch),
                                 meta_info=deepcopy(data.meta_info))
    if not torch.equal(mask.sum(-1), result.batch['attention_mask'].sum(-1)):
        raise ValueError('Attended token count changed')
    report = {'kind': 'native_masked_margin_compaction', 'rows': len(data),
              'prompt_before': p, 'prompt_after': p - left,
              'response_before': r, 'response_after': response_end,
              'attended_tokens': int(mask.sum()),
              'loss_tokens': int(tensors['response_mask'].sum()),
              'accepted_rows_removed': 0, 'valid_tokens_removed': 0}
    return result, report
