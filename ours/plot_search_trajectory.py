"""Plot complete native MH selection and allocation history, without new inference.

This is an observation of the existing E3 selection step, not a Method component
or a heldout/ablation estimate. Extraction reruns the native completion verifier;
rendering uses the separately installed plotting environment without changing the
frozen training runtime.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .visual_task import file_sha256


def extract(plan_path, proof_path, output):
    from .search_completion import verify_completed_search
    if output.exists():
        raise ValueError('Preserve the existing trajectory data')
    proof = json.loads(proof_path.read_text())
    if verify_completed_search(plan_path) != proof:
        raise ValueError('The completed native search no longer matches its proof')
    result = json.loads(Path(proof['result_path']).read_text())
    entries = result['evaluations']
    rounds = {name: row['iteration'] for row in proof['history'] for name in row['evaluated']}
    changes = {row['evaluated'][-1]: row['accepted'] for row in proof['history']}
    scores = {entry['harness']: entry['solved'] for entry in entries}
    accepted = entries[0]['harness']
    cumulative = 0.
    rows = []
    for index, entry in enumerate(entries):
        name = entry['harness']
        audit = json.loads((Path(entry['directory']) / 'audit.json').read_text())
        cumulative += entry['gpu_hours']
        accepted = changes.get(name, accepted)
        rows.append({'evaluation_index': index, 'round': rounds.get(name, 0), 'candidate': name,
            'solved': entry['solved'], 'examples': audit['examples'], 'policy_calls': audit['calls'],
            'generated_tokens': audit['generated_tokens'], 'job_id': entry['job_id'],
            'evaluation_gpu_hours': entry['gpu_hours'], 'cumulative_evaluation_gpu_hours': cumulative,
            'accepted_after_evaluation': accepted, 'accepted_solved': scores[accepted],
            'selection_boundary': index == 0 or name in changes})
    if len(rows) != proof['totals']['evaluations'] or abs(cumulative - proof['totals']['gpu_hours']) > 1e-10:
        raise ValueError('Extracted rows differ from verified totals')
    report = {'kind': 'completed_native_search_trajectory', 'status': 'VERIFIED_NATIVE_HISTORY',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'condition': proof['condition'],
        'seed': proof['seed'], 'rows': rows, 'totals': proof['totals'],
        'source_sha256': {str(Path(__file__).resolve()): file_sha256(Path(__file__))},
        'input_sha256': {str(p.resolve()): file_sha256(p) for p in (plan_path, proof_path)},
        'new_model_calls': 0, 'new_api_calls': 0, 'test_task_rows_loaded': False,
        'limitations': ['Repeated fixed MH optimization tasks, one trial; not independent test data.',
            'Accepted values change only at actual native selection boundaries, not after every candidate.',
            'GPU costs include every completed candidate evaluation allocation in this search.',
            'Proposer, training, export and unrelated engineering costs are outside this figure.',
            'No error bars, pooled independent sample claim or causal attribution from this single trial.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def render(data_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(data_path.read_text())
    if report['kind'] != 'completed_native_search_trajectory' or report['status'] != 'VERIFIED_NATIVE_HISTORY':
        raise ValueError('Require a complete native search trajectory')
    for name, digest in {**report['source_sha256'], **report['input_sha256']}.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f'Changed figure source: {name}')
    rows = report['rows']
    totals = {row['examples'] for row in rows}
    if len(totals) != 1:
        raise ValueError('Unequal task coverage cannot share a count axis')
    count = totals.pop()
    paths = [output.with_suffix(suffix) for suffix in ('.csv', '.png', '.pdf', '.receipt.json')]
    if any(path.exists() for path in paths):
        raise ValueError('Preserve existing figure outputs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.3), sharex=True,
        gridspec_kw={'height_ratios': [1.5, 1]}, layout='constrained')
    x = [row['evaluation_index'] for row in rows]
    axes[0].scatter(x, [row['solved'] for row in rows], color='#62788A', s=45,
        label='Each evaluated candidate', zorder=3)
    axes[0].step(x, [row['accepted_solved'] for row in rows], where='post', color='#176AA6',
        linewidth=2, label='Accepted harness after completed round')
    boundaries = [row for row in rows if row['selection_boundary']]
    axes[0].scatter([row['evaluation_index'] for row in boundaries],
        [row['accepted_solved'] for row in boundaries], marker='s', color='#176AA6', s=30, zorder=4)
    for row in rows:
        axes[0].annotate(str(row['solved']), (row['evaluation_index'], row['solved']),
            xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8, color='#334554')
    axes[0].set_ylim(-0.8, count)
    axes[0].set_yticks(range(0, count + 1, 8))
    axes[0].set_ylabel(f'Solved / {count} fixed MH tasks')
    axes[0].legend(loc='upper left', frameon=False)
    axes[0].set_title(f"{report['condition']} · seed {report['seed']}\nOptimization-set search history; no heldout or ablation claim", loc='left')
    axes[1].plot(x, [row['cumulative_evaluation_gpu_hours'] for row in rows], color='#B45C22', marker='.', linewidth=2)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel('Cumulative evaluation GPU-hours')
    axes[1].set_xlabel('Evaluation order: incoming h0, then up to three candidates per round')
    axes[1].set_xticks(x, [row['candidate'] for row in rows])
    last = rows[-1]
    axes[1].annotate(f"{last['cumulative_evaluation_gpu_hours']:.3f} GPUh", (x[-1], last['cumulative_evaluation_gpu_hours']),
        xytext=(-6, -18), textcoords='offset points', ha='right', color='#874014')
    for ax in axes:
        ax.grid(axis='y', color='#E4E8EB', linewidth=0.7)
        ax.set_xlim(-0.4, x[-1] + 0.4)
        for row in boundaries[:-1]:
            ax.axvline(row['evaluation_index'] + 0.5, color='#D9DFE3', linewidth=0.7, linestyle='--')
    fig.savefig(paths[1], dpi=180)
    fig.savefig(paths[2])
    plt.close(fig)
    receipt = {'status': 'RENDERED_VERIFIED_NATIVE_HISTORY', 'data_path': str(data_path),
        'data_sha256': file_sha256(data_path), 'plotted_evaluations': len(rows),
        'figure_sha256': {str(path): file_sha256(path) for path in paths[:3]},
        'source_sha256': file_sha256(Path(__file__)), 'matplotlib_version': matplotlib.__version__,
        'new_model_calls': 0, 'limitations': report['limitations']}
    paths[3].write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('extract', 'render'), required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase == 'extract':
        if args.plan is None or args.proof is None:
            parser.error('extract requires --plan and --proof')
        result = extract(args.plan, args.proof, args.data)
    else:
        if args.output is None:
            parser.error('render requires --output')
        result = render(args.data, args.output)
    print(json.dumps({key: value for key, value in result.items() if key in ('status', 'totals', 'plotted_evaluations')}))
