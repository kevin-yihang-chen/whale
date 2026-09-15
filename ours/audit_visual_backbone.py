"""Bind actual E4 backbone pixels/gradients to the successful native batch."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .visual_pixel_bootstrap import tensor_identity
from .visual_task import file_sha256


def audit(plan_path, batch_path, output):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from verl import DataProto
    import torch
    from .research_budget import terminal_allocation
    plan = json.loads(plan_path.read_text())
    assert plan.get('execution_variant') == 'qwen35_visual_pixel_restart'
    root = Path(plan['output'])
    result = json.loads((root / 'visual-forward-result.json').read_text())
    batch_audit = json.loads(batch_path.read_text())
    assert result['plan_sha256'] == batch_audit['plan_sha256'] == file_sha256(plan_path)
    assert result['job_id'] == batch_audit['job_id']
    assert batch_audit['status'] == 'AUDITED_NATIVE_VISUAL_RSFT_BATCH'
    terminal = terminal_allocation((root / 'slurm-terminal.txt').read_text(), result['job_id'])
    assert terminal['state'] == 'COMPLETED'
    for name, digest in result['artifacts_sha256'].items():
        assert file_sha256(Path(name)) == digest
    accepted_path = root / 'checkpoints/audit/accepted-step1.pkl'
    accepted_receipt = json.loads(accepted_path.with_suffix('.json').read_text())
    assert file_sha256(accepted_path) == accepted_receipt['sha256']
    accepted = DataProto.load_from_disk(accepted_path)
    assert plan['config']['trainer']['online_rsft']['sft_epochs'] == 1
    rows = [json.loads(line) for name in result['artifacts_sha256'] for line in Path(name).read_text().splitlines()]
    forwards = [r for r in rows if r['kind'] == 'visual_backbone_input']
    gradients = [r for r in rows if r['kind'] == 'visual_feature_gradient']
    assert all(r['job_id'] == result['job_id'] for r in rows)
    assert all(r['grad_enabled'] and r['exact_after_native_dtype_cast'] for r in forwards)
    assert forwards and {r['actor_pixels']['dtype'] for r in forwards} == {'torch.bfloat16'}
    assert {r['consumed_pixels']['dtype'] for r in forwards} == {'torch.float32'}
    # FSDP's native root-input cast precedes the observed conditional model;
    # Qwen's get_image_features then casts to visual.dtype. Replay both casts.
    expected = Counter(json.dumps({
        'actor_pixels': tensor_identity(image['pixel_values'].to(torch.bfloat16)),
        'consumed_pixels': tensor_identity(image['pixel_values'].to(torch.bfloat16).float()),
        'grid': tensor_identity(image['image_grid_thw'])}, sort_keys=True)
        for image in accepted.non_tensor_batch['multi_modal_inputs'])
    actual = Counter(json.dumps({k:r[k] for k in ('actor_pixels','consumed_pixels','grid')}, sort_keys=True) for r in forwards)
    assert actual == expected, 'Actual backbone image multiset differs from native accepted batch'
    assert len({r['event'] for r in forwards}) == len(forwards)
    assert Counter(r['event'] for r in gradients) == Counter(r['event'] for r in forwards)
    assert all(r['finite'] and r['l1'] >= 0 for r in gradients)
    assert any(r['nonzero_elements'] > 0 for r in gradients)
    report = {'status': 'AUDITED_NATIVE_IMAGE_CONSUMPTION_AND_GRADIENTS',
        'job_id': result['job_id'], 'plan_sha256': file_sha256(plan_path),
        'batch_audit_sha256': file_sha256(batch_path), 'accepted_examples': len(accepted),
        'actual_backbone_forwards': len(forwards), 'paired_gradient_events': len(gradients),
        'nonzero_gradient_events': sum(r['nonzero_elements'] > 0 for r in gradients),
        'actual_images_match_native_success_subset': True,
        'native_pixel_casts_replayed': ['accepted float32', 'FSDP root bfloat16', 'Qwen visual-input float32'],
        'evidence_sha256': {str(p): file_sha256(p) for p in [accepted_path,
            root / 'visual-forward-result.json', root / 'slurm-terminal.txt', Path(__file__)]},
        'limitations': ['Image-backbone use and feature gradients verified; full model parameter comparison is separate.',
                       'One corrected engineering batch; no method advantage, independent test or causal degradation claim.']}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--batch-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.plan.resolve(), args.batch_audit.resolve(), args.output.resolve())
