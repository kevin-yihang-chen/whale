"""Compare archived E1 processor tensors across a native E4 checkpoint change.

This complements the actual request RGB/token/decoding comparison. It checks
the native agent's processor output, not every vLLM internal activation or
run-to-run numerical determinism. No model calls or parameter updates occur.
"""
import argparse
import json
from pathlib import Path

from .visual_task import file_sha256


def audit(before, after, output):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import torch
    from verl import DataProto
    roots = [before / 'evaluation', after / 'evaluation']
    results = [json.loads((p / 'result.json').read_text()) for p in roots]
    artifacts = [r['batch_artifact_sha256'] for r in results]
    assert set(artifacts[0]) == set(artifacts[1])
    sources, ids, changes = {}, set(), []
    for name in sorted(artifacts[0]):
        paths = [p / name for p in roots]
        for i, p in enumerate(paths):
            sources[str(p)] = file_sha256(p)
            assert sources[str(p)] == artifacts[i][name]
        a, b = [DataProto.load_from_disk(p) for p in paths]
        lookup = {k: i for i, k in enumerate(b.non_tensor_batch['visual_sample_id'])}
        assert len(a) == len(b) == len(lookup)
        assert set(a.non_tensor_batch['visual_sample_id']) == set(lookup)
        for i, sample in enumerate(a.non_tensor_batch['visual_sample_id']):
            assert sample not in ids
            ids.add(sample)
            j = lookup[sample]
            width_a, width_b = a.batch['prompts'].shape[-1], b.batch['prompts'].shape[-1]
            assert width_a == width_b
            for key in ('prompts', 'attention_mask', 'position_ids'):
                x, y = a.batch[key][i][..., :width_a], b.batch[key][j][..., :width_b]
                assert x.dtype == y.dtype and torch.equal(x, y), (sample, key)
            x = a.non_tensor_batch['multi_modal_inputs'][i]
            y = b.non_tensor_batch['multi_modal_inputs'][j]
            assert set(x) == set(y) and {'pixel_values', 'image_grid_thw'} <= set(x)
            for key in x:
                assert x[key].dtype == y[key].dtype and torch.equal(x[key], y[key]), (sample, key)
            for key in ('reward_model', 'visual_harness_sha256', 'raw_prompt'):
                assert a.non_tensor_batch[key][i] == b.non_tensor_batch[key][j], (sample, key)
            answers = [r.non_tensor_batch['visual_committed_answer'][k] for r, k in ((a, i), (b, j))]
            if answers[0] != answers[1]:
                changes.append({'sample_id': sample, 'answers': answers,
                    'chosen_token_log_probabilities': [r.batch['rollout_log_probs'][k][r.batch['response_mask'][k].bool()].tolist()
                                                      for r, k in ((a, i), (b, j))]})
    assert len(ids) == results[0]['examples'] == results[1]['examples'] == 128
    result = {'status': 'PASS_NATIVE_VISUAL_PROCESSOR_INPUT_EQUALITY', 'examples': len(ids),
        'batch_pairs': len(artifacts[0]), 'changed_answers': changes,
        'source_sha256': sources, 'audit_source_sha256': file_sha256(Path(__file__)),
        'model_calls': 0, 'optimizer_steps': 0,
        'limitations': ['Exact native-agent pixel/position/prompt tensors, not every internal serving activation.',
            'Does not test repeated-run numerical determinism or establish a co-adaptation mechanism.']}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.before.resolve(), args.after.resolve(), args.output.resolve())
