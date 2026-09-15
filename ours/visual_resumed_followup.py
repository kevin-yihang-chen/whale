"""Measure E1 under fixed h0 after the accepted E4 native continuation.

Reuse the existing canonical exporter, real visual evaluator and matched-input
comparison. Native update evidence comes from FP32 theta1 versus FP32 theta2;
the separate BF16 comparison measures what survives the serving export.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

from . import native_visual_followup as followup
from .evidence import fingerprint
from .local_completion import checkpoint_manifest
from .research_budget import terminal_allocation
from .visual_task import file_sha256

ROOT = followup.ROOT
SOURCES = ('ours/visual_resumed_followup.py', 'ours/run_visual_resumed_followup.sh',
           'ours/verify_visual_resumed_parameters.py', 'ours/audit_native_visual_resume.py')


def read(path):
    return json.loads(Path(path).read_text())


def combine_comparisons(export_comparison, resumed_comparison):
    """Keep a prior BF16 rounding residual out of the new-update measurement."""
    assert export_comparison['export_exact_native_bf16_cast'] is True
    assert export_comparison['native_fp32']['tensor_count'] == resumed_comparison['tensor_count']
    assert export_comparison['native_fp32']['total_elements'] == resumed_comparison['total_elements']
    assert {r['name'] for r in export_comparison['native_fp32']['tensors']} == {
        r['name'] for r in resumed_comparison['tensors']}
    return {**export_comparison,
            'native_vs_incoming_bf16_reference': export_comparison['native_fp32'],
            'native_fp32': resumed_comparison,
            'native_fp32_reference': 'Original restored FP32 theta1, before the accepted E4 continuation.'}


def evidence(training_path, audit_path, parameter_path, allocation_path, baseline_path):
    training, audit, parameters, allocation, baseline = map(read, (
        training_path, audit_path, parameter_path, allocation_path, baseline_path))
    assert training['kind'] == 'native_visual_rsft_resume' and training['role'] == 'W'
    root = Path(training['output'])
    execution = read(root / 'execution-result.json')
    assert execution['status'] == 'NATIVE_VISUAL_RSFT_RESUME_RETURNED'
    assert execution['plan_sha256'] == audit['plan_sha256'] == parameters['plan_sha256'] == file_sha256(training_path)
    assert execution['job_id'] == audit['job_id'] == parameters['job_id'] == allocation['job_id']
    assert audit['status'] == 'AUDITED_NATIVE_VISUAL_RSFT_RESUMED_BATCH'
    assert audit['input_pixels_and_tokens_replayed'] and audit['native_success_subset_exact']
    assert audit['native_restore_and_both_transports_observed'] and audit['native_step'] == 2
    assert audit['audit_source_sha256'] == file_sha256(ROOT / 'ours/audit_native_visual_resume.py')
    assert parameters['status'] == 'COMPLETE_NATIVE_RESUMED_PARAMETER_COMPARISON'
    assert parameters['batch_audit_sha256'] == file_sha256(audit_path)
    assert parameters['allocation_sha256'] == file_sha256(allocation_path)
    assert parameters['incoming_resume_artifacts_unchanged']
    for report in (audit, parameters):
        followup.verify_files(report['source_sha256'])
    followup.verify_files(parameters['native_checkpoint_sha256'])
    terminal_path = allocation_path.parent / 'slurm-terminal.txt'
    assert file_sha256(terminal_path) == allocation['terminal_sha256']
    terminal = terminal_allocation(terminal_path.read_text(), allocation['job_id'])
    assert terminal and terminal['state'] == allocation['state'] == 'COMPLETED'
    assert terminal['seconds'] == allocation['seconds']
    native = root / 'checkpoints/global_step_2/actor'
    assert Path(execution['native_checkpoint']) == native
    previous = Path(training['resume_checkpoint']['directory'])
    assert set(parameters['native_checkpoint_sha256']) == {
        str(previous / 'actor/model_world_size_1_rank_0.pt'), str(native / 'model_world_size_1_rank_0.pt')}
    resume = {str(previous / p): sha for p, sha in training['resume_checkpoint']['artifact_sha256'].items()}
    resume.update({str(p): file_sha256(p) for p in native.parent.rglob('*') if p.is_file()})
    followup.verify_files(resume)
    assert str(native.parent / 'data.pt') in resume
    assert str(native / 'extra_state_world_size_1_rank_0.pt') in resume
    assert not any('optim_world_size' in p for p in resume)
    handoff = training['handoff']
    old_plan_path, old_transition_path = Path(handoff['followup_plan']), Path(handoff['transition_path'])
    assert file_sha256(old_plan_path) == handoff['followup_plan_sha256']
    assert file_sha256(old_transition_path) == handoff['transition_sha256']
    old_plan, old_transition = read(old_plan_path), read(old_transition_path)
    assert baseline['mode'] == 'direct' and baseline['role'] == 'V'
    assert baseline['model'] == old_transition['exported'] == checkpoint_manifest(Path(baseline['model']['path']))
    baseline_root = Path(baseline['output'])
    before = read(baseline_root / 'result.json')
    assert before['status'] == 'COMPLETE_POST_TRAINING_VISUAL_CASE'
    assert before['plan_sha256'] == file_sha256(baseline_path)
    h0 = Path(handoff['search_root']) / 'search/harnesses/h0/harness.py'
    assert file_sha256(Path(baseline['config']['data']['visual_harness_path'])) == file_sha256(h0)
    manifest, pairs, _ = followup.pair_inputs(Path(baseline['manifest']))
    assert manifest['role'] == 'V' and len(pairs) == 64
    assert parameters['native_fp32']['tensor_count'] == old_plan['active_tensors']
    inputs = (training_path, audit_path, parameter_path, allocation_path, terminal_path, baseline_path,
              root / 'execution-result.json', root / 'checkpoints/audit/update-step2.json',
              old_plan_path, old_transition_path, baseline_root / 'result.json',
              baseline_root / 'evaluation/result.json', Path(baseline['manifest']))
    return training, baseline, parameters, native, resume, old_plan, inputs


def prepare(path, output, training_path, audit_path, parameter_path, allocation_path, baseline_path):
    assert not path.exists() and not output.exists()
    training, baseline, parameters, native, resume, old_plan, inputs = evidence(
        training_path, audit_path, parameter_path, allocation_path, baseline_path)
    sources = {**training['source_sha256'], **baseline['source_sha256']}
    sources.update({p: file_sha256(ROOT / p) for p in (*followup.EXTRA, *SOURCES)})
    followup.verify_files(sources)
    plan = {'kind': 'native_visual_checkpoint_followup', 'followup_mode': 'accepted_native_resume',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_plan': str(training_path), 'training_job_id': parameters['job_id'],
        'baseline_plan': str(baseline_path), 'baseline_output': baseline['output'],
        'base': baseline['model'], 'native': str(native), 'resume_sha256': resume,
        'output': str(output), 'target': str(output / 'hf-step-2-canonical'),
        'source_sha256': sources, 'inputs_sha256': {str(p): file_sha256(p) for p in inputs},
        'aliases': old_plan['aliases'], 'active_tensors': old_plan['active_tensors'],
        'reviewed_native_config_differences': followup.validate_inference_config(
            Path(baseline['model']['path']), native / 'huggingface'),
        'resumed_evidence': {'training_plan': str(training_path), 'batch_audit': str(audit_path),
            'parameter_audit': str(parameter_path), 'allocation': str(allocation_path),
            'baseline_plan': str(baseline_path), 'native_fp32_sha256': fingerprint(parameters['native_fp32'])},
        'resources': {'gpus': 1, 'cpus': 12, 'ram_gib': 96, 'time_limit_seconds': 3600},
        'storage': {'additional_working_space_gib': 32, 'minimum_free_margin_gib': 40,
                    'new_model_exports': 1, 'delete_existing_files': False},
        'bounds': {'development_pairs': 64, 'maximum_generation_calls': 384,
                   'maximum_generated_assistant_tokens': 131072, 'new_optimizer_steps': 0, 'api_calls': 0},
        'limitations': ['One accepted continuation and fixed h0 on the already used 64-pair V subset; not an independent method comparison.',
            'Native FP32 update is theta1-to-theta2. The incoming BF16 reference residual is separately labelled.',
            'Exported BF16 differences measure serving precision; neither parameter change proves better answers.',
            'A single after-evaluation does not resolve the known inference variability or establish a training mechanism.',
            'Both native checkpoints, scheduler/RNG/sampler evidence and failures are retained. No T or R access.']}
    followup.native.write_new(path, plan)


def check(path):
    plan = followup.check(path)
    assert plan['followup_mode'] == 'accepted_native_resume'
    item = plan['resumed_evidence']
    _, baseline, parameters, native, resume, _, _ = evidence(*[
        Path(item[k]) for k in ('training_plan', 'batch_audit', 'parameter_audit', 'allocation', 'baseline_plan')])
    assert plan['base'] == baseline['model'] and plan['native'] == str(native) and plan['resume_sha256'] == resume
    assert fingerprint(parameters['native_fp32']) == item['native_fp32_sha256']
    return plan


def run(path):
    plan = check(path)
    parameters = read(plan['resumed_evidence']['parameter_audit'])
    original_compare, original_matched = followup.compare_active_parameters, followup.matched_results
    def compare(*args, **kwargs):
        return combine_comparisons(original_compare(*args, **kwargs), parameters['native_fp32'])
    def matched(*args, **kwargs):
        result = original_matched(*args, **kwargs)
        result['limitations'] = plan['limitations']
        return result
    with patch.object(followup, 'compare_active_parameters', compare), patch.object(followup, 'matched_results', matched):
        followup.run(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'check', 'run'), required=True)
    for name in ('plan', 'output', 'training-plan', 'batch-audit', 'parameter-audit', 'allocation', 'baseline-plan'):
        parser.add_argument('--' + name, type=Path, required=name == 'plan')
    args = parser.parse_args()
    path = args.plan.resolve()
    if args.phase == 'prepare':
        prepare(path, args.output.resolve(), args.training_plan.resolve(), args.batch_audit.resolve(),
                args.parameter_audit.resolve(), args.allocation.resolve(), args.baseline_plan.resolve())
    elif args.phase == 'check':
        check(path)
    else:
        run(path)
