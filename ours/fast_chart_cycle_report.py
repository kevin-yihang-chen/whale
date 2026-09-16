"""Reconstruct a completed E4 cycle and its descriptive V-set trajectory.

No new model calls, training, selection, or independent method comparison.
Different evaluation harnesses are explicit; equal choices are not a VETO gain.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .chart_answer_protocol import parse_answer, verify
from .fast_chart_admission import read, require_parameter_update, verified_calibration
from .fast_chart_protocol import OUTPUT, ROOT
from .fast_chart_statistics import paired_records
from .native_visual_service import write_new
from .visual_task import file_sha256


def difference(before, after):
    if set(before) != set(after) or not before:
        raise ValueError('Require identical complete paired coverage')
    for key in before:
        if before[key]['source_id'] != after[key]['source_id']:
            raise ValueError('Source chart identity changed')
    return {
        'paired_improved': sum(after[k]['paired'] > before[k]['paired'] for k in before),
        'paired_worsened': sum(after[k]['paired'] < before[k]['paired'] for k in before),
        'paired_pp': 100 * sum(after[k]['paired'] - before[k]['paired'] for k in before) / len(before),
        'ordinary_pp': 100 * sum(after[k]['marginal'] - before[k]['marginal'] for k in before) / len(before),
        'interpretation': 'Descriptive joint weight/harness change on V; not a causal gate comparison.',
    }


def latex(report):
    if report['status'] != 'COMPLETE_RECONSTRUCTED_FIRST_CYCLE':
        raise ValueError('Incomplete cycles cannot enter the completed trajectory table')
    labels = {'initial': r'$\theta_0,h_0$', 'prefix': r'$\theta_1,h_0$', 'continued': r'$\theta_2,h_3$'}
    lines = [r'\begin{table}[h]\centering\small',
             r'\begin{tabular}{lrr}\toprule', r'State / harness & Single & Paired\\\midrule']
    for name in ('initial', 'prefix', 'continued'):
        row = report['development'][name]
        lines.append(labels[name] + f" & {row['images_correct']}/512 & {row['pairs_correct']}/256" + r'\\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Seed42 development trajectory through one completed native cycle. The final row changes both weights and harness; V was used for configuration calibration. This is not an independent method comparison.}\end{table}']
    if report['equivalent_decisions']:
        lines.append('WHALE, VETO and the marginal gate all select h3. Only the prospectively specified VETO continuation was executed; its outcome is not substituted for unexecuted control branches.')
    lines.append('The second stage contains256 trajectories, of which ' + str(report['training']['successful_trajectories']) + ' passed the shared verifier. Full native parameter change and the exact BF16 export were verified. No independent test score is reported.')
    return '\n'.join(lines) + '\n'


def reconstruct(cycle_path, configuration):
    cycle_path, configuration = Path(cycle_path).resolve(), Path(configuration).resolve()
    cycle = read(cycle_path)
    if cycle['status'] != 'COMPLETE_BOUNDED_FIRST_VISUAL_CYCLE' or cycle['seed'] != 42:
        raise ValueError('Require the complete prospectively bounded seed42 cycle')
    evidence = {}

    def keep(path, expected=None):
        path = Path(path)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Cycle evidence changed: {path}')
        evidence[str(path)] = digest
        return read(path)

    keep(cycle_path)
    for path, digest in cycle['evidence_sha256'].items():
        keep(path, digest)
    cfg = verified_calibration(configuration)
    keep(configuration)
    selected = cfg['results'][str(cfg['learning_rate'])]
    keep(selected['training_plan'], selected['training_plan_sha256'])
    followup_path = Path(cycle['development']['evaluation_plan']).parent/'plan.json'
    followup = keep(followup_path, cycle['development']['plan_sha256'])
    train = keep(followup['training_plan'], followup['training_plan_sha256'])
    if (train['stage'] != 2 or train['seed'] != 42 or train['parent_plan'] != selected['training_plan']
            or train['rate'] != cfg['learning_rate'] or train['augmentation'] is not None
            or train['bounds']['native_batches'] != 4 or train['bounds']['trajectories'] != 256):
        raise ValueError('Continuation differs from the shared selected prefix and fixed budget')
    selection = keep(train['selection'])
    if (selection['condition'] != 'veto' or selection['parent_plan_sha256'] != selected['training_plan_sha256']
            or selection['harness_sha256'] != train['harness_sha256']
            or cycle['selections'] != {'whale': 'h3', 'veto': 'h3', 'marginal_gate': 'h3'}
            or not cycle['equivalent_decisions']):
        raise ValueError('This report requires the observed common h3 choice, not a new gate comparison')
    proposal = keep(ROOT/'results/fast-chart-first-proposal-tool-use-review-20260915-v1.json')
    events = Path(proposal['events'])
    if (proposal['status'] != 'COMPLETE_PROPOSER_TOOL_USE_REVIEW'
            or proposal['recorded_H_feedback_read'] is not False
            or not events.is_relative_to(Path(train['selection']).parent)
            or file_sha256(events) != proposal['events_sha256']):
        raise ValueError('Recorded proposer limitation is not bound to this candidate archive')
    evidence[str(events)] = proposal['events_sha256']
    execution = keep(Path(train['output'])/'execution-result.json', followup['training_result_sha256'])
    transition = keep(followup_path.parent/'transition.json', cycle['development']['transition_sha256'])
    if (execution['plan_sha256'] != followup['training_plan_sha256']
            or transition['stage2_parent_sha256'] != train['resume_checkpoint']['artifact_sha256']['actor/model_world_size_1_rank_0.pt']
            or transition['native_checkpoint'] != execution['checkpoint']
            or execution['checkpoint']['step'] != 8 or not transition['resume_artifacts_unchanged']):
        raise ValueError('Native continuation or parent/export identity changed')
    require_parameter_update(execution, transition, stage=2)
    image_audit = keep(followup_path.parent/'image-consumption-audit.json', transition['image_consumption_audit_sha256'])
    if (image_audit['status'] != 'AUDITED_COMPACT_IMAGE_CONSUMPTION_AND_GRADIENTS'
            or image_audit['plan_sha256'] != followup['training_plan_sha256']
            or not image_audit['actual_images_match_native_success_subset']):
        raise ValueError('Image consumption is not bound to this successful training subset')

    batches = []
    for offset, step in enumerate(range(5, 9)):
        folder = Path(train['output'])/'checkpoints/audit'
        rollout = keep(folder/f'rollout-step{step}.json')
        accepted = keep(folder/f'accepted-step{step}.json')
        update = keep(folder/f'update-step{step}.json')
        path = Path(train['output'])/f'rollouts/{step}.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        evidence[str(path)] = file_sha256(path)
        if len(records) != 64 or any(r['step'] != step or int(verify(parse_answer(r['output']), r['gts'])) != r['score'] for r in records):
            raise ValueError('Native rollout scores differ from the fixed shared verifier')
        expected = [key for key in train['cpu_preflight']['next_four_batches'][offset] for _ in range(8)]
        success_ids = Counter(k for k, score in zip(rollout['sample_ids'], rollout['scores'], strict=True) if score == 1)
        if (rollout['sample_ids'] != expected or Counter(accepted['sample_ids']) != success_ids
                or accepted['examples'] != sum(r['score'] for r in records)
                or any(score != 1 for score in accepted['scores'])
                or update['accepted_receipt'] != accepted or update['status'] != 'NATIVE_SFT_CALL_RETURNED'):
            raise ValueError('Sample order/multiplicity or native successful filtering changed')
        expected_receipt = execution['batches'][offset]
        if (expected_receipt['step'] != step or expected_receipt['successful_trajectories'] != accepted['examples']
                or expected_receipt['rollout_receipt_sha256'] != file_sha256(folder/f'rollout-step{step}.json')):
            raise ValueError('Execution summary does not match the actual ordered batch')
        metrics = update['metrics']
        batches.append({'step': step, 'successful_trajectories': accepted['examples'],
            'optimizer_updates': int(sum(metrics['actor/sft_updates'])),
            'assistant_loss_tokens': int(sum(metrics['actor/sft_token_count']))})
    if image_audit['accepted_examples'] != sum(b['successful_trajectories'] for b in batches):
        raise ValueError('Image audit omits an accepted training example')

    h0 = keep(cfg['h0_calibration'])
    initial_path = Path(h0['results'][h0['selected']]['result'])
    prefix_path = Path(selected['result_path']).parent/'development/result.json'
    final_path = followup_path.parent/'development/result.json'
    keep(final_path, cycle['development']['evaluation_result_sha256'])
    manifest = Path(followup['manifest'])
    keep(manifest, followup['manifest_sha256'])
    development, per_pair = {}, {}
    for name, path in (('initial', initial_path), ('prefix', prefix_path), ('continued', final_path)):
        value = keep(path)
        if value['status'] != 'COMPLETE_COMPACT_CHART_PAIR_EVALUATION' or value['role'] != 'V':
            raise ValueError('Require completed V results, not another data role')
        raw_path = path.parent/'evaluation/result.json'
        raw = keep(raw_path)
        if any(r['committed_answer'] != parse_answer(r['raw_answer']) for r in raw['records']):
            raise ValueError('Stored answers differ from the shared raw-answer parser')
        pairs, _ = paired_records(manifest, raw_path)
        if len(pairs) != 256:
            raise ValueError('Require all256 development pairs')
        single = sum(sum(row['correctness']) for row in pairs.values())
        both = sum(row['paired'] for row in pairs.values())
        if single/512 != value['marginal_accuracy'] or both/256 != value['paired_accuracy']:
            raise ValueError('Published scores differ from complete raw answers')
        per_pair[name] = pairs
        development[name] = {'images_correct': single, 'images': 512, 'pairs_correct': both, 'pairs': 256,
            'generated_tokens': sum(row['generated_tokens'] for row in pairs.values()),
            'result': str(path), 'identity': value['audit']['identity'], 'harness_sha256': value['audit']['harness_sha256']}
    return {'status': 'COMPLETE_RECONSTRUCTED_FIRST_CYCLE', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': 42, 'equivalent_decisions': True, 'selections': cycle['selections'],
        'training': {'batches': batches, 'successful_trajectories': sum(b['successful_trajectories'] for b in batches),
            'optimizer_updates': sum(b['optimizer_updates'] for b in batches),
            'nonzero_visual_gradient_events': execution['nonzero_gradient_events'],
            'stage2_changed_parameter_elements': transition['stage2_native_fp32_change']['changed_elements'],
            'exact_native_bf16_export': transition['export_exact_native_bf16_cast']},
        'development': development, 'changes': {name: difference(per_pair[name], per_pair['continued']) for name in ('initial', 'prefix')},
        'evidence_sha256': evidence, 'scientific_method_verified': False,
        'independent_test_evaluated': False, 'completed_control_branches_imputed': False,
        'recorded_proposer_H_feedback_read': False,
        'limitation': 'One executed continuation with a changed evaluation harness; no VETO-versus-WHALE effect or three-seed interval.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle', type=Path, required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(OUTPUT):
        raise ValueError('Require a fresh compact evidence directory')
    report = reconstruct(args.cycle, args.configuration)
    output.mkdir()
    write_new(output/'result.json', report)
    (output/'cycle-development.tex').write_text(latex(report))
    write_new(output/'manifest.json', {'report_sha256': file_sha256(output/'result.json'),
        'table_sha256': file_sha256(output/'cycle-development.tex'), 'source_sha256': file_sha256(Path(__file__))})
    print(json.dumps({k: report[k] for k in ('status', 'training', 'changes', 'scientific_method_verified')}, indent=2))
