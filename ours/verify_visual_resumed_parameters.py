"""Measure E4's actual native FP32 change without another model export.

Compare restored theta1 directly with saved theta2. This distinguishes a new
update from a dtype conversion or an earlier update, and does not claim better
answers. All tensors and the reviewed tied aliases are checked on CPU.
"""
import argparse
import json
from pathlib import Path

from .verify_native_transition import summary, tensor_delta, verify_aliases
from .visual_task import file_sha256


def compare_states(before, after, aliases, tensor_count):
    import torch
    assert set(before) == set(after)
    names = sorted(set(before) - set(aliases))
    assert len(names) == tensor_count and set(aliases.values()) <= set(names)
    for state in (before, after):
        assert all(type(t) is torch.Tensor and t.dtype == torch.float32 for t in state.values())
        verify_aliases(state, aliases, state.__getitem__)
    return summary([{'name': name, **tensor_delta(before[name], after[name])} for name in names])


def verify(plan_path, batch_audit_path, allocation_path, output):
    import torch
    from .research_budget import terminal_allocation
    read = lambda p: json.loads(Path(p).read_text())
    plan, audit, allocation = map(read, (plan_path, batch_audit_path, allocation_path))
    assert plan['kind'] == 'native_visual_rsft_resume' and plan['role'] == 'W'
    assert audit['status'] == 'AUDITED_NATIVE_VISUAL_RSFT_RESUMED_BATCH'
    assert audit['plan_sha256'] == file_sha256(plan_path)
    assert audit['audit_source_sha256'] == file_sha256(Path(__file__).with_name('audit_native_visual_resume.py'))
    for name, digest in audit['source_sha256'].items():
        assert file_sha256(Path(name)) == digest
    terminal_path = allocation_path.parent / 'slurm-terminal.txt'
    assert file_sha256(terminal_path) == allocation['terminal_sha256']
    terminal = terminal_allocation(terminal_path.read_text(), allocation['job_id'])
    assert terminal and terminal['state'] == allocation['state'] == 'COMPLETED'
    assert terminal['seconds'] == allocation['seconds']
    root = Path(plan['output'])
    execution = read(root / 'execution-result.json')
    assert execution['status'] == 'NATIVE_VISUAL_RSFT_RESUME_RETURNED'
    assert execution['plan_sha256'] == file_sha256(plan_path)
    assert execution['job_id'] == allocation['job_id'] == audit['job_id']
    previous = Path(plan['resume_checkpoint']['directory'])
    for name, digest in plan['resume_checkpoint']['artifact_sha256'].items():
        assert file_sha256(previous / name) == digest
    current = root / 'checkpoints/global_step_2'
    assert Path(execution['native_checkpoint']) == current / 'actor'
    assert (current / 'data.pt').is_file()
    assert (current / 'actor/extra_state_world_size_1_rank_0.pt').is_file()
    assert not list(current.rglob('optim_world_size*'))
    paths = [p / 'actor/model_world_size_1_rank_0.pt' for p in (previous, current)]
    for directory in (previous, current):
        assert read(directory / 'actor/fsdp_config.json')['world_size'] == 1
    followup_path = Path(plan['handoff']['followup_plan'])
    assert file_sha256(followup_path) == plan['handoff']['followup_plan_sha256']
    followup = read(followup_path)
    states = [torch.load(p, map_location='cpu', mmap=True, weights_only=True) for p in paths]
    result = compare_states(*states, followup['aliases'], followup['active_tensors'])
    report = {'status': 'COMPLETE_NATIVE_RESUMED_PARAMETER_COMPARISON',
        'job_id': allocation['job_id'], 'plan_sha256': file_sha256(plan_path),
        'batch_audit_sha256': file_sha256(batch_audit_path),
        'allocation_sha256': file_sha256(allocation_path),
        'native_checkpoint_sha256': {str(p): file_sha256(p) for p in paths},
        'source_sha256': {str(p): file_sha256(p) for p in (Path(__file__),
            Path(__file__).with_name('verify_native_transition.py'))},
        'incoming_resume_artifacts_unchanged': True, 'native_fp32': result,
        'new_model_exports': 0, 'model_calls': 0, 'optimizer_steps_by_verifier': 0,
        'limitations': ['Native theta1-to-theta2 values only; unchanged outcomes are retained.',
                       'No BF16 export, independent answer evaluation or VETO efficacy is established here.']}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'batch-audit', 'allocation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.plan.resolve(), args.batch_audit.resolve(), args.allocation.resolve(), args.output.resolve())
