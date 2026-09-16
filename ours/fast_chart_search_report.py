"""Reconstruct E1 and all E2/E3 decisions from a complete real candidate archive.

This is optimization-set evidence, not a final method comparison. Failed slots
remain explicit and are never assigned artificial zero accuracies.
"""
import argparse
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from .acceptance import EvidenceConstrainedAcceptance
from .adapter import WHALEAcceptanceAdapter
from .chart_answer_protocol import decode_truth, number, parse_answer, verify
from .chart_selection_protocol import selection_modes
from .evidence import EvaluationIdentity, fingerprint
from .fast_chart_protocol import OUTPUT
from .fast_chart_search import load, read, SLOTS, verified_failure, candidate_slots
from .fast_chart_search_evaluation import certify
from .native_visual_service import write_new
from .visual_search_bridge import boundary
from .visual_task import file_sha256


def answer_categories(records, truth):
    """Describe answer-type parseability without inferring visual causality."""
    actual = {row['sample_id']: row for row in records}
    if len(records) != len(actual) or set(actual) != set(truth):
        raise ValueError('Format diagnosis requires complete identity-matched records')
    result = {'correct': 0, 'parseable_wrong': 0, 'unparseable': 0}
    for key, row in actual.items():
        value, label = row['committed_answer'], truth[key]
        if value != parse_answer(row['raw_answer']) or row['correct'] != int(verify(value, label)):
            raise ValueError('Format diagnosis differs from the fixed host scorer')
        valid = value in ('A','B') if decode_truth(label)['kind'] == 'binary' else number(value) is not None
        result['correct' if row['correct'] else ('parseable_wrong' if valid else 'unparseable')] += 1
    return result


def counts(item):
    audit = item['audit']
    if len(audit.correctness) != 64 or len(item['h']['records']) != 128:
        raise ValueError('Require all registered H128 and C64-pair outcomes')
    correct = audit.correctness
    zero, one, both = [sum(a+b == k for a, b in correct) for k in range(3)]
    single = sum(a+b for a, b in correct)
    if both != single-64+zero:
        raise ValueError('Paired and marginal decomposition disagree')
    return {'candidate': item['candidate'].name, 'H_correct': sum(r['correct'] for r in item['h']['records']),
        'H_total': 128, 'mean_turns': item['candidate'].mean_turns,
        'C_single_correct': single, 'C_images': 128, 'C_both_correct': both, 'C_pairs': 64,
        'C_zero_one_two_correct': [zero, one, both]}


def reconstruct(root):
    root = Path(root).resolve()
    result = read(root/'result.json')
    if result['status'] != 'COMPLETE_COMPACT_SEARCH_AND_SELECTION':
        raise ValueError('A partial search cannot enter the completed comparison table')
    if result.get('schema_version') != 2:
        raise ValueError('Legacy searches must be reconstructed with their preserved verifier source')
    from meta_harness import meta_harness_chess_puzzle as native
    plan = load(root)
    evidence, archive, failed, allocations = {}, [], {}, {}

    def keep(path):
        path = Path(path)
        evidence[str(path)] = file_sha256(path)

    def measurement(receipt):
        for key in ('plan', 'allocation'):
            if file_sha256(Path(receipt[key])) != receipt[key+'_sha256']:
                raise ValueError('Measurement receipt changed')
            keep(receipt[key])
        item = certify(receipt['plan'], receipt['allocation'])
        out = Path(item['plan']['output'])
        for path in (out/'result.json', out/'H/result.json', out/'C/result.json'):
            keep(path)
        archive.append(item)
        allocations[receipt['allocation']] = read(receipt['allocation'])

    measurement(plan['baseline'])
    for name in SLOTS:
        good, bad = root/f'evaluation-{name}.json', root/f'failure-{name}.json'
        if good.exists() == bad.exists():
            raise ValueError('Every allocated slot needs exactly one success or failure receipt')
        if good.exists():
            keep(good)
            measurement(read(good))
            if archive[-1]['candidate'].name != name or archive[-1]['result']['identity'] != plan['identity']:
                raise ValueError('Measurement belongs to another candidate slot or phase')
        else:
            keep(bad)
            failure = verified_failure(root, name, plan)
            if 'allocation' in failure:
                for key in ('plan', 'allocation'):
                    keep(failure[key])
                allocation = read(failure['allocation'])
                failed_output = Path(read(failure['plan'])['output'])/'failure.json'
                keep(failed_output)
                for filename in ('submission.json', 'slurm-terminal.txt'):
                    keep(Path(failure['allocation']).parent/filename)
                allocations[failure['allocation']] = allocation
            failed[name] = failure
    slots = candidate_slots(root, archive, plan)
    slots_path = root/'candidate-slots.json'
    slots_sha = file_sha256(slots_path)
    if (result['candidate_slots_sha256'] != slots_sha or
            any(result[key] != value for key, value in slots['counts'].items())):
        raise ValueError('Final search result differs from its complete candidate slots')
    keep(slots_path)
    identity = EvaluationIdentity(**plan['identity'])
    comparison_path = root/'search/logs/iteration_001/comparison.json'
    comparison = read(comparison_path)
    stage = 'early_stop' if comparison['early_stop'] else 'ordinary'
    decisions = {}
    for condition, mode in selection_modes(plan):
        saved_path = root/f'selection-{condition}.json'
        restored_path = root/f'restored-selection-{condition}.json'
        saved, restored = read(saved_path), read(restored_path)
        adapter = WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance(mode))
        kwargs = dict(frontier=comparison['frontier'], rows=comparison['summary'], valid_names=list(SLOTS))
        actual = boundary(adapter, native, root/'search', archive, identity, stage=stage, **kwargs)
        resumed = boundary(adapter, native, root/'search', archive, identity, stage='resume', resume=saved['receipt'], **kwargs)
        actual_archive = {i['candidate'].name: {'candidate': asdict(i['candidate']), 'audit': asdict(i['audit'])} for i in archive}
        if (saved['status'] != 'COMPLETE_COMPACT_SELECTION' or saved['condition'] != condition or
                saved.get('selection_protocol','gate_v1') != plan.get('selection_protocol','gate_v1') or
                saved['candidate_slots_sha256'] != slots_sha or saved['candidate_counts'] != slots['counts'] or
                saved['identity'] != plan['identity'] or saved['parent_plan_sha256'] != plan['parent_plan_sha256'] or
                fingerprint(saved['archive']) != fingerprint(actual_archive) or
                saved['accepted_harness'] != result['selections'][condition] or
                file_sha256(Path(saved['selected_harness'])) != saved['harness_sha256'] or
                any(fingerprint(saved[k]) != fingerprint(actual[k]) for k in actual) or
                fingerprint(restored) != fingerprint(resumed)):
            raise ValueError('Saved, reconstructed and resumed selections differ')
        keep(saved_path)
        keep(restored_path)
        decisions[condition] = actual
    for path in (root/'plan.json', root/'result.json', comparison_path):
        keep(path)
    equivalent = len(set(result['selections'].values())) == 1
    if result['equivalent_decisions'] != equivalent:
        raise ValueError('Equivalent-decision flag disagrees with actual choices')
    rows = []
    for item in archive:
        row = counts(item)
        h_truth = {r['visual_sample_id']: r['reward_model']['ground_truth'] for r in item['h_rows']}
        manifest = read(item['plan']['partitions']['C']['manifest'])
        c_truth = {fingerprint({'pair_id': pair['pair_id'], 'side': side}): pair['answers'][side]
            for pair in manifest['pairs'] for side in (0,1)}
        c = read(Path(item['plan']['output'])/'C/result.json')
        row['answer_categories'] = {'H': answer_categories(item['h']['records'], h_truth),
            'C': answer_categories(c['records'], c_truth)}
        rows.append(row)
    return {'status': 'COMPLETE_RECONSTRUCTED_SEARCH_REPORT', 'search': str(root), 'seed': plan['seed'],
        'selection_protocol':plan.get('selection_protocol','gate_v1'),
        'candidate_counts': slots['counts'],
        'rows': rows, 'failed_slots': failed, 'decisions': decisions,
        'equivalent_decisions': equivalent, 'evidence_sha256': evidence,
        'candidate_evaluation_gpu_hours': str(sum((Decimal(a['gpu_hours']) for a in allocations.values()), Decimal(0))),
        'cost_scope': 'Incoming and candidate H/C GPU allocations only, including allocated failures; excludes prefix training, API generation and continuation.',
        'scientific_method_verified': False,
        'interpretation': 'H/C are optimization data. Selection differences or score gains here are not independent post-training benefits.'}


def latex(report):
    lines = [r'\begin{table}[h]\centering\small', r'\begin{tabular}{lrrr}\toprule',
        r'Candidate & H correct & C single & C both\\\midrule']
    for row in report['rows']:
        lines.append(f"{row['candidate']} & {row['H_correct']}/128 & {row['C_single_correct']}/128 & {row['C_both_correct']}/64"+r'\\')
    for name in report['failed_slots']:
        lines.append(name+r' & \multicolumn{3}{c}{Failed; no score assigned}\\')
    lines.extend([r'\bottomrule\end{tabular}',
        r'\caption{Complete fixed-weight optimization archive. These are H/C selection measurements, not independent method results.}\end{table}'])
    labels = {'whale': 'WHALE', 'veto': 'VETO', 'marginal_gate': 'Marginal gate',
              'marginal_rank_floor':'Marginal ranking with competence floor'}
    lines.append('Selections: '+', '.join(labels[k]+' '+v['accepted_harness'] for k,v in report['decisions'].items())+'.')
    if report['equivalent_decisions']:
        lines.append('All three rules select the same harness; this archive supplies no evidence of an independent VETO selection benefit.')
    else:
        lines.append('Selection differs; independent evaluation after actual continuation is still required.')
    formats = [row for row in report['rows'] if 'answer_categories' in row]
    if formats:
        lines.append('Unparseable answers on H/C: '+', '.join(
            f"{row['candidate']} {row['answer_categories']['H']['unparseable']}/{row['answer_categories']['C']['unparseable']}"
            for row in formats)+'. These descriptive counts do not identify causal visual improvements.')
    return '\n'.join(lines)+'\n'


def build(root, output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT):
        raise ValueError('Require a fresh compact report directory')
    report = reconstruct(root)
    output.mkdir()
    write_new(output/'result.json', report)
    (output/'candidate-selection.tex').write_text(latex(report))
    write_new(output/'manifest.json', {'report_sha256': file_sha256(output/'result.json'),
        'table_sha256': file_sha256(output/'candidate-selection.tex'), 'source_sha256': file_sha256(Path(__file__)),
        'scientific_method_verified': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.search, args.output)
