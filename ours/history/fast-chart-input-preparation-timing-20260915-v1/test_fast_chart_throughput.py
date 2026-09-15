from copy import deepcopy
import pytest

from ours.fast_chart_throughput import validate_case, select_fastest


def plans():
    reference = dict(partition='V', batch_size=8, manifest='V.json', manifest_sha256='data',
        audit_data_sha256='pairs', verifier_sha256='host', model={'weights': 'same'},
        worker_coordinates=[42], harness_sha256='shared-h0', seed=42,
        config={'data': {'cache_dir': 'old', 'prompt': 'same'},
            'trainer': {'experiment_name': 'old'},
            'actor_rollout_ref': {'rollout': {'max_num_seqs': 8,
                'trace': {'experiment_name': 'old'}, 'max_assistant_tokens': 1024}}})
    candidate = deepcopy(reference)
    candidate.update(kind='compact_chart_throughput_case', batch_size=32,
        bounds={'images': 512, 'maximum_generation_calls': 1536})
    candidate['config']['data']['cache_dir'] = 'new'
    candidate['config']['trainer']['experiment_name'] = 'new'
    candidate['config']['actor_rollout_ref']['rollout'].update(max_num_seqs=32,
        trace={'experiment_name': 'new'})
    return reference, candidate


def test_concurrency_change_preserves_scientific_inputs():
    reference, candidate = plans()
    validate_case(candidate, reference)
    for mutation in ('prompt', 'tokens', 'seed', 'test', 'coverage'):
        altered = deepcopy(candidate)
        if mutation == 'prompt': altered['config']['data']['prompt'] = 'easier'
        if mutation == 'tokens': altered['config']['actor_rollout_ref']['rollout']['max_assistant_tokens'] = 512
        if mutation == 'seed': altered['seed'] = 43
        if mutation == 'test': altered['partition'] = 'T'
        if mutation == 'coverage': altered['bounds']['images'] = 128
        with pytest.raises(ValueError):
            validate_case(altered, reference)


def test_fastest_selection_ignores_accuracy_and_requires_all_passes():
    rows = {'8': {'seconds_per_image': 2., 'accuracy': 1.},
        '16': {'seconds_per_image': 1., 'accuracy': .9},
        '32': {'seconds_per_image': .5, 'accuracy': .8}}
    assert select_fastest(rows) == 32
    rows['16']['seconds_per_image'] = .5
    assert select_fastest(rows) == 16
    rows.pop('8')
    with pytest.raises(ValueError):
        select_fastest(rows)
