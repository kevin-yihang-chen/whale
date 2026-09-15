"""Measure the next E4 update against its actual canonical incoming checkpoint.

Unlike the initial base-to-native transition, this incoming graph already omits
inactive MTP parameters and stores all active parameters in BF16. No graph edits
are allowed. Native FP32 changes and changes surviving BF16 export are separate.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from .audit_native_training_batch import require
from .local_completion import checkpoint_manifest
from .verify_native_transition import summary, tensor_delta, verify_aliases
from .visual_task import file_sha256


def compare_active_parameters(base, native_dir, exported, *, aliases, tensor_count):
    import torch
    from safetensors import safe_open
    state_path = native_dir / 'model_world_size_1_rank_0.pt'
    require(json.loads((native_dir / 'fsdp_config.json').read_text())['world_size'] == 1,
            'Only the reviewed single-actor graph is supported')
    state = torch.load(state_path, map_location='cpu', mmap=True, weights_only=True)
    require(all(type(t) is torch.Tensor for t in state.values()), 'Unreviewed distributed tensor type')
    with ExitStack() as stack:
        maps = []
        for root in (base, exported):
            mapping = {}
            for path in sorted(root.glob('*.safetensors')):
                reader = stack.enter_context(safe_open(str(path), framework='pt', device='cpu'))
                for name in reader.keys():
                    require(name not in mapping, 'Duplicate tensor in shards')
                    mapping[name] = reader
            maps.append(mapping)
        base_map, export_map = maps
        names = set(base_map) - set(aliases)
        require(len(names) == tensor_count and set(aliases.values()) <= names, 'Incoming active graph differs')
        require(set(state) - set(aliases) == names == set(export_map) - set(aliases),
                'Unreviewed missing or extra active tensor')
        verify_aliases(state, aliases, state.__getitem__)
        for mapping in maps:
            verify_aliases(mapping, aliases, lambda k: mapping[k].get_tensor(k))
        native_rows, export_rows = [], []
        for name in sorted(names):
            a, b, c = base_map[name].get_tensor(name), state[name], export_map[name].get_tensor(name)
            require(a.dtype == c.dtype == torch.bfloat16 and b.dtype == torch.float32,
                    'Unexpected incoming/native/export precision')
            native_rows.append({'name': name, **tensor_delta(a, b)})
            export_rows.append({'name': name, **tensor_delta(a, c, native=b)})
    return {'native_fp32': summary(native_rows), 'exported_bf16': summary(export_rows),
            'export_exact_native_bf16_cast': True}


def verify(plan_path, native_dir, exported, previous_transition):
    from omegaconf import OmegaConf
    plan = json.loads(plan_path.read_text())
    require(plan['kind'] == 'native_alternation_training_plan', 'Wrong training plan')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(Path(name)) == digest, f'Frozen source changed: {name}')
    previous = json.loads(previous_transition.read_text())
    require(plan['source_sha256'].get(str(previous_transition)) == file_sha256(previous_transition),
            'Incoming transition evidence is not bound to the training plan')
    require(previous['export_exact_native_bf16_cast'] is True, 'Incoming canonical export lacks value verification')
    base = Path(plan['model'])
    manifest = checkpoint_manifest(base)
    require(manifest == previous['exported'], 'Incoming checkpoint differs from reviewed transition')
    raw = Path(plan['resolved_config']).read_text()
    config = OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
    mtp = config.actor_rollout_ref.model.mtp
    require(not (mtp.enable or mtp.enable_train or mtp.enable_rollout), 'Unreviewed MTP activation')
    result = compare_active_parameters(base, native_dir, exported, aliases=previous['verified_aliases'],
                                      tensor_count=previous['exported_bf16']['tensor_count'])
    return {'kind': 'native_alternation_checkpoint_transition', 'base': manifest,
            'native_checkpoint': str(native_dir / 'model_world_size_1_rank_0.pt'),
            'native_checkpoint_sha256': file_sha256(native_dir / 'model_world_size_1_rank_0.pt'),
            'exported': checkpoint_manifest(exported), 'plan_sha256': file_sha256(plan_path),
            'previous_transition_sha256': file_sha256(previous_transition),
            'verified_aliases': previous['verified_aliases'],
            'audit_source_sha256': file_sha256(Path(__file__)),
            'audit_dependency_sha256': {name: file_sha256(Path(name)) for name in
                ('ours/verify_native_transition.py', 'ours/local_completion.py', 'ours/visual_task.py')}, **result,
            'limitations': ['All incoming active tensors are BF16; dtype conversion alone cannot count as an update.',
                'This checks values and the active graph, not optimizer completion or task improvement.',
                'Tokenizer/config semantics and actual serving weights require separate verification.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--exported', type=Path, required=True)
    parser.add_argument('--previous-transition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing transition evidence')
    report = verify(args.plan, args.native, args.exported, args.previous_transition)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: {n: v for n, v in report[k].items() if n != 'tensors'}
                      for k in ('native_fp32', 'exported_bf16')}), flush=True)
    require(report['native_fp32']['status'] == report['exported_bf16']['status'] == 'CHANGED',
            'Changed native and exported parameters are both required')
