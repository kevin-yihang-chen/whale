"""Materialize the approved 51-condition matrix without inventing outcomes."""
import argparse
import itertools
import json
from pathlib import Path

from .visual_task import file_sha256

PROTOCOL = Path(__file__).with_name('veto_execution_protocol_20260910.md')
MAIN = ('weight_only', 'harness_only', 'whale_fst', 'whale', 'veto')
ABLATIONS = ('marginal_gate', 'counterfactual_augmentation', 'ordinary_reevaluation',
             'soft_pair', 'matched_random_rejection')


def schedule():
    rows = []
    groups = [('main', ('charts', 'clevr'), ('4B',), MAIN),
              ('scale', ('charts',), ('2B',), ('whale', 'veto')),
              ('ablation', ('charts',), ('4B',), ABLATIONS)]
    for group, domains, sizes, conditions in groups:
        for domain, size, condition, seed in itertools.product(domains, sizes, conditions, (42, 43, 44)):
            rows.append({'run_id': f'{domain}-qwen35-{size.lower()}-{condition}-seed{seed}',
                'group': group, 'domain': domain, 'model': f'Qwen3.5-{size}',
                'condition': condition, 'seed': seed, 'status': 'PLANNED_NOT_EXECUTED',
                'requires': ['verified_scientific_go', 'frozen_visual_configuration', 'storage_capacity', 'resource_reservation'],
                'result_path': None, 'failure_path': None})
    assert len(rows) == len({r['run_id'] for r in rows}) == 51
    return {'kind': 'approved_veto_research_matrix', 'protocol_path': str(PROTOCOL.resolve()),
        'protocol_sha256': file_sha256(PROTOCOL), 'source_sha256': file_sha256(Path(__file__)),
        'formal_runs': len(rows), 'runs': rows,
        'pilot': {'domain': 'charts', 'models': ['Qwen3.5-4B'], 'seeds': [42, 43, 44],
            'conditions': ['weight_only', 'harness_only', 'whale'], 'status': 'NOT_EXECUTED'},
        'budget_gpu_hours': {'pilot': 100, 'main': 280, 'mechanism': 160, 'verification': 60},
        'gpu_total_ceiling': 600, 'global_proposer_ceiling_cny': 45,
        'chess': {'finish': ['whale_seed42_search', 'whale_seed42_phase2_and_handoff'],
            'defer': ['remaining_chess_seeds', 'chess_whale_fst'], 'test64': 'SEALED'},
        'evidence_status': 'NO_VISUAL_METHOD_RESULT',
        'scope': 'A schedule is not a submitted job, completed experiment or performance observation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = schedule()
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'PLANNED_NOT_EXECUTED', 'formal_runs': record['formal_runs'],
        'output_sha256': file_sha256(args.output)}))
