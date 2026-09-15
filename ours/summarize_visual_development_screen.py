"""Recompute E1 from completed development responses and retain matched contrasts.

This is descriptive evidence analysis, not an E2/E3 selection rule. Bootstrap
intervals describe this development sample, not variation across training seeds.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .evidence import fingerprint
from .research_budget import terminal_allocation
from .visual_task import binary_answer_verifier, file_sha256


def summarize(plan_path, output):
    plan = json.loads(plan_path.read_text())
    root = Path(plan['output'])
    result_path = root / 'result.json'
    result = json.loads(result_path.read_text())
    assert result['status'] == 'COMPLETE_VISUAL_DEVELOPMENT_SCREEN'
    assert result['plan_sha256'] == file_sha256(plan_path)
    assert result['optimizer_steps'] == 0 and not result['formal_matrix_started']
    terminal = terminal_allocation((root / 'slurm-terminal.txt').read_text(), result['job_id'])
    assert terminal is not None and terminal['state'] == 'COMPLETED'
    manifest_path = Path(plan['manifest'])
    assert file_sha256(manifest_path) == plan['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['role'] == 'V'
    pairs = manifest['pairs']
    groups = list(dict.fromkeys(p['source_id'] for p in pairs))
    assert len(pairs) == len(groups) == 64  # One pair per distinct source here.
    draws = np.random.default_rng(20260910).integers(0, len(groups), size=(20000, len(groups)))
    sources = {str(plan_path): file_sha256(plan_path), str(result_path): file_sha256(result_path),
               str(manifest_path): file_sha256(manifest_path),
               str(root / 'slurm-terminal.txt'): file_sha256(root / 'slurm-terminal.txt')}
    modes, vectors = {}, {}
    for mode, item in plan['cases'].items():
        case_path = Path(item['path'])
        assert file_sha256(case_path) == item['sha256']
        receipt_path = root / mode / 'result.json'
        receipt = json.loads(receipt_path.read_text())
        assert file_sha256(receipt_path) == result['results'][mode]['result_sha256']
        assert receipt['plan_sha256'] == item['sha256'] and receipt['job_id'] == result['job_id']
        evaluation_path = root / mode / 'evaluation/result.json'
        evaluation = json.loads(evaluation_path.read_text())
        assert evaluation['status'] == 'COMPLETE_NATIVE_PAIR_EVALUATION'
        records = {r['sample_id']: r for r in evaluation['records']}
        assert len(records) == len(evaluation['records']) == 2 * len(pairs)
        correctness, evidence = [], []
        for pair in pairs:
            rows = [records[fingerprint({'pair_id': pair['pair_id'], 'side': side})] for side in (0, 1)]
            correct = [int(binary_answer_verifier(row['committed_answer'], truth))
                       for row, truth in zip(rows, pair['answers'])]
            assert correct == [r['correct'] for r in rows]
            correctness.append(correct)
            evidence.append({'pair_id': pair['pair_id'], 'source_id': pair['source_id'],
                             'answers': [r['committed_answer'] for r in rows], 'correctness': correct})
        assert receipt['audit']['pair_ids'] == [p['pair_id'] for p in pairs]
        assert receipt['audit']['correctness'] == correctness
        values = np.asarray(correctness)
        pair_values, marginal_values = values.prod(axis=1), values.mean(axis=1)
        assert float(pair_values.mean()) == receipt['paired_accuracy'] == evaluation['paired_accuracy']
        assert float(marginal_values.mean()) == receipt['marginal_accuracy'] == evaluation['marginal_accuracy']
        assert sum(r['policy_calls'] for r in records.values()) == receipt['observed']['policy_calls']
        assert sum(r['generated_tokens'] for r in records.values()) == receipt['observed']['generated_tokens']
        for artifact, sha in evaluation['batch_artifact_sha256'].items():
            assert file_sha256(evaluation_path.parent / artifact) == sha
        sources.update({str(p): file_sha256(p) for p in (case_path, receipt_path, evaluation_path)})
        modes[mode] = {'correct_images': int(values.sum()), 'images': int(values.size),
            'correct_pairs': int(pair_values.sum()), 'pairs': len(pairs),
            'marginal_accuracy': float(marginal_values.mean()), 'paired_accuracy': float(pair_values.mean()),
            'policy_calls': receipt['observed']['policy_calls'],
            'generated_tokens': receipt['observed']['generated_tokens'],
            'output_labels': dict(Counter(r['committed_answer'] for r in records.values())),
            'pair_evidence': evidence}
        traces = [t for r in records.values() for t in r['tool_trace']]
        modes[mode]['tool_calls'] = len(traces)
        modes[mode]['tool_calls_returning_images'] = sum(bool(t['returned_images']) for t in traces)
        modes[mode]['unfinished_final_tool_calls'] = sum('<tool_call' in r['raw_answer'] for r in records.values())
        vectors[mode] = {'paired_accuracy': pair_values, 'marginal_accuracy': marginal_values}
        if mode == 'blind':
            assert receipt['observed']['image_absence_verified']
            modes[mode]['same_answer_both_sides'] = sum(p['answers'][0] == p['answers'][1] for p in evidence)
    contrasts = {}
    for left, right in (('direct', 'blind'), ('normalized_zoom', 'direct')):
        contrasts[left + '_minus_' + right] = {}
        for metric in ('paired_accuracy', 'marginal_accuracy'):
            delta = vectors[left][metric] - vectors[right][metric]
            means = delta[draws].mean(axis=1)
            contrasts[left + '_minus_' + right][metric] = {
                'difference': float(delta.mean()),
                'source_paired_bootstrap_95_percentile_interval': np.quantile(means, [.025, .975]).tolist()}
    value = {'status': 'RESCORED_COMPLETED_DEVELOPMENT_SCREEN', 'job_id': result['job_id'],
        'role': 'V', 'modes': modes, 'contrasts': contrasts, 'source_sha256': sources,
        'analysis_source_sha256': file_sha256(Path(__file__)), 'gpu_hours': str(terminal['gpu_hours']),
        'bootstrap': {'unit': 'source chart, both sides and all modes together', 'replicates': 20000,
                      'seed': 20260910},
        'limitations': ['Development subset, not official PlotQA or sealed test performance.',
            'Tool availability differs; no training or VETO acceptance occurred.',
            'Intervals condition on one fixed model and subset, not multiple training seeds.',
            'Percentile intervals may be degenerate at a boundary and do not establish population certainty.']}
    with output.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({mode: {k: v for k, v in d.items() if k not in ('pair_evidence', 'output_labels')}
                      for mode, d in modes.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.plan.resolve(), args.output.resolve())
