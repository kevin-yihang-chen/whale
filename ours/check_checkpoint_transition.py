"""Measure the actual E4 parameter transition between HF safetensors checkpoints.

File, config and dtype changes alone do not establish a weight update. This
compares tensor values in bounded CPU chunks and does not evaluate model quality.
"""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path

from .local_completion import checkpoint_manifest


def compare(base, updated):
    import torch
    from safetensors import safe_open

    manifests = [checkpoint_manifest(base), checkpoint_manifest(updated)]
    results = []
    with ExitStack() as stack:
        maps = []
        for root in (base, updated):
            mapping = {}
            for path in sorted(root.glob('*.safetensors')):
                reader = stack.enter_context(safe_open(str(path), framework='pt', device='cpu'))
                for key in reader.keys():
                    if key in mapping:
                        raise ValueError(f'Duplicate tensor: {key}')
                    mapping[key] = reader
            maps.append(mapping)
        if set(maps[0]) != set(maps[1]):
            raise ValueError('Checkpoint tensor names differ; mapping requires explicit review')
        if not maps[0]:
            raise ValueError('No parameter tensors found')
        for name in sorted(maps[0]):
            a, b = [mapping[name].get_tensor(name) for mapping in maps]
            if a.shape != b.shape:
                raise ValueError(f'Tensor shape differs: {name}')
            if not a.is_floating_point() or not b.is_floating_point():
                raise ValueError('Only real floating-point parameter tensors are supported')
            elements, maximum, squared = 0, 0.0, 0.0
            av, bv = a.reshape(-1), b.reshape(-1)
            for offset in range(0, a.numel(), 1_000_000):
                ac = av[offset:offset + 1_000_000].to(torch.float64)
                bc = bv[offset:offset + 1_000_000].to(torch.float64)
                if not torch.isfinite(ac).all() or not torch.isfinite(bc).all():
                    raise ValueError(f'Nonfinite tensor: {name}')
                delta = bc - ac
                elements += int(torch.count_nonzero(delta))
                maximum = max(maximum, float(delta.abs().max()))
                squared += float(torch.dot(delta, delta))
            results.append({'name': name, 'numel': a.numel(), 'changed_elements': elements,
                            'max_abs_delta': maximum, 'squared_delta': squared,
                            'base_dtype': str(a.dtype), 'updated_dtype': str(b.dtype)})
    changed = sum(r['changed_elements'] for r in results)
    return {'kind': 'checkpoint_tensor_transition', 'status': 'CHANGED' if changed else 'UNCHANGED',
            'base': manifests[0], 'updated': manifests[1], 'tensor_count': len(results),
            'total_elements': sum(r['numel'] for r in results), 'changed_elements': changed,
            'changed_tensors': sum(r['changed_elements'] > 0 for r in results),
            'parameter_delta_l2': math.sqrt(sum(r['squared_delta'] for r in results)),
            'max_abs_delta': max(r['max_abs_delta'] for r in results), 'tensors': results,
            'limitations': ['Tensor-value differences are not proof of improved model performance.',
                            'Training provenance, configuration semantics and live service handoff require separate checks.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('updated', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-changed', action='store_true')
    args = parser.parse_args()
    report = compare(args.base, args.updated)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'changed_tensors', 'changed_elements', 'parameter_delta_l2')}))
    if args.require_changed and report['status'] != 'CHANGED':
        raise SystemExit('No tensor-value update was verified')
