"""Verify the reviewed E4 active graph, FP32 update and BF16 export separately.

Only the pinned inactive MTP keys and explicitly tied output alias are handled.
This measures parameter values, not accuracy, novelty or a completed live handoff.
"""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path

from .audit_native_training_batch import require
from .local_completion import checkpoint_manifest
from .visual_task import file_sha256


def comparison_names(base_names, updated_names, inactive, aliases):
    base_names, updated_names, inactive = set(base_names), set(updated_names), set(inactive)
    require(inactive <= base_names, 'Reviewed inactive tensors missing from base')
    active = base_names - inactive
    require(not (active & set(aliases)), 'Ambiguous base aliases')
    require(set(aliases.values()) <= active, 'Alias target missing')
    require(updated_names - set(aliases) == active, 'Unreviewed missing or extra active tensor')
    return sorted(active)


def tensor_delta(base, changed, *, native=None):
    import torch
    require(base.shape == changed.shape, 'Tensor shape changed')
    require(base.is_floating_point() and changed.is_floating_point(), 'Expected floating-point tensors')
    if native is not None:
        require(native.shape == changed.shape, 'Native/export tensor shapes differ')
    count, maximum, squared = 0, 0., 0.
    a, b = base.reshape(-1), changed.reshape(-1)
    n = None if native is None else native.reshape(-1)
    for i in range(0, a.numel(), 1_000_000):
        ac, bc = a[i:i + 1_000_000].double(), b[i:i + 1_000_000].double()
        require(bool(torch.isfinite(ac).all() and torch.isfinite(bc).all()), 'Nonfinite parameter')
        if n is not None:
            require(torch.equal(n[i:i + 1_000_000].bfloat16(), b[i:i + 1_000_000]),
                    'Export differs from native BF16 cast')
        delta = bc - ac
        count += int(torch.count_nonzero(delta))
        maximum = max(maximum, float(delta.abs().max()))
        squared += float(torch.dot(delta, delta))
    return {'numel': a.numel(), 'changed_elements': count, 'max_abs_delta': maximum,
            'squared_delta': squared, 'base_dtype': str(base.dtype), 'updated_dtype': str(changed.dtype)}


def verify_aliases(mapping, aliases, get_tensor):
    import torch
    for alias, target in aliases.items():
        if alias in mapping:
            a, b = get_tensor(alias), get_tensor(target)
            require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), 'Tied alias values differ')


def summary(rows):
    count = sum(r['changed_elements'] for r in rows)
    return {'status': 'CHANGED' if count else 'UNCHANGED', 'tensor_count': len(rows),
            'total_elements': sum(r['numel'] for r in rows), 'changed_elements': count,
            'changed_tensors': sum(r['changed_elements'] > 0 for r in rows),
            'parameter_delta_l2': math.sqrt(sum(r['squared_delta'] for r in rows)),
            'max_abs_delta': max(r['max_abs_delta'] for r in rows), 'tensors': rows}


def verify(base, native_dir, exported, review_path, recovery_plan):
    import torch
    from safetensors import safe_open
    from omegaconf import OmegaConf
    review = json.loads(review_path.read_text())
    require(review['kind'] == 'native_checkpoint_graph_review', 'Wrong graph review')
    require(file_sha256(base / 'config.json') == review['base_config_sha256'], 'Base configuration changed')
    for name, digest in review['source_sha256'].items():
        require(file_sha256(Path(name)) == digest, f'Graph implementation changed: {name}')
    recovery = json.loads(recovery_plan.read_text())
    require(str(base.resolve()) == recovery['model'], 'Wrong base model')
    base_manifest = checkpoint_manifest(base)
    require(base_manifest['weights_sha256'] == recovery['weights_sha256'], 'Base checkpoint changed')
    config = OmegaConf.load(recovery['resolved_config'])
    mtp = config.actor_rollout_ref.model.mtp
    require(not (mtp.enable or mtp.enable_train or mtp.enable_rollout), 'Inactive MTP mapping inapplicable')
    state_path = native_dir / 'model_world_size_1_rank_0.pt'
    require(json.loads((native_dir / 'fsdp_config.json').read_text())['world_size'] == 1, 'Only reviewed single actor supported')
    state = torch.load(state_path, map_location='cpu', mmap=True, weights_only=True)
    require(all(type(t) is torch.Tensor for t in state.values()), 'Unreviewed native distributed tensor type')
    inactive = review['findings']['base_only_inactive_mtp_names']
    aliases = review['findings']['alias']
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
        names = comparison_names(base_map, state, inactive, aliases)
        require(names == comparison_names(base_map, export_map, inactive, aliases), 'Export graph differs')
        require(len(names) == review['findings']['independent_comparison_names'], 'Active graph count differs')
        verify_aliases(state, aliases, state.__getitem__)
        verify_aliases(export_map, aliases, lambda k: export_map[k].get_tensor(k))
        native_rows, export_rows = [], []
        for name in names:
            a, b, c = base_map[name].get_tensor(name), state[name], export_map[name].get_tensor(name)
            require(b.dtype == torch.float32 and c.dtype == torch.bfloat16, 'Unexpected native/export dtype')
            native_rows.append({'name': name, **tensor_delta(a, b)})
            export_rows.append({'name': name, **tensor_delta(a, c, native=b)})
    return {'kind': 'reviewed_native_checkpoint_transition', 'base': base_manifest,
            'native_checkpoint': str(state_path), 'native_checkpoint_sha256': file_sha256(state_path),
            'exported': checkpoint_manifest(exported), 'graph_review_sha256': file_sha256(review_path),
            'recovery_plan_sha256': file_sha256(recovery_plan),
            'inactive_base_keys': inactive, 'verified_aliases': aliases, 'native_fp32': summary(native_rows),
            'exported_bf16': summary(export_rows), 'export_exact_native_bf16_cast': True,
            'limitations': ['Value changes do not establish model improvement.',
                           'Generation/tokenizer/config semantics and live inference require separate checks.',
                           'Optimizer completion and finite gradient evidence are checked from the native job separately.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--exported', type=Path, required=True)
    p.add_argument('--review', type=Path, default=Path('results/native-checkpoint-graph-review-20260909.json'))
    p.add_argument('--recovery-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = verify(args.base, args.native, args.exported, args.review, args.recovery_plan)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: {n: v for n, v in report[k].items() if n != 'tensors'}
                      for k in ('native_fp32', 'exported_bf16')}), flush=True)
    require(report['native_fp32']['status'] == report['exported_bf16']['status'] == 'CHANGED',
            'A changed native and exported checkpoint are both required')
