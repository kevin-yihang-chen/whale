from copy import deepcopy
from unittest.mock import patch
import pytest

from ours.fast_chart_throughput import (validate_case, select_fastest,
    input_preparation_timing, preparation_seconds)


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


def test_preparation_timer_preserves_calls_returns_and_restores_functions():
    from ours import native_visual_service, visual_native_evaluation
    payload, returned = object(), object()
    with patch.object(visual_native_evaluation, 'write_pair_parquet', return_value=returned) as writer, \
            patch.object(native_visual_service, 'dataset_for', return_value=returned) as dataset, \
            patch('ours.fast_chart_throughput.time.monotonic', side_effect=[1, 3, 4, 7]):
        with input_preparation_timing() as measured:
            assert visual_native_evaluation.write_pair_parquet(payload, output='same') is returned
            assert native_visual_service.dataset_for(payload, 'unchanged') is returned
        assert visual_native_evaluation.write_pair_parquet is writer
        assert native_visual_service.dataset_for is dataset
        writer.assert_called_once_with(payload, output='same')
        dataset.assert_called_once_with(payload, 'unchanged')
        assert measured == {'write_pair_parquet': [2], 'dataset_for': [3]}


def test_failed_preparation_is_not_recorded_as_completed():
    from ours import native_visual_service, visual_native_evaluation
    original = native_visual_service.dataset_for
    with patch.object(visual_native_evaluation, 'write_pair_parquet', side_effect=RuntimeError('failure')) as writer:
        with pytest.raises(RuntimeError, match='failure'):
            with input_preparation_timing() as measured:
                visual_native_evaluation.write_pair_parquet('input', 'output')
        assert measured == {}
        assert visual_native_evaluation.write_pair_parquet is writer
        assert native_visual_service.dataset_for is original


def test_preparation_requires_matching_complete_finite_measurements():
    record = dict(status='COMPLETE_INPUT_PREPARATION_TIMING', plan_sha256='plan', job_id='1',
        images=512, stage_seconds={'write_pair_parquet': [2], 'dataset_for': [3]})
    result = dict(job_id='1', timing_seconds={'loading': 10})
    assert preparation_seconds(record, 'plan', result) == 5
    for field, value in [('job_id', '2'), ('images', 2048), ('plan_sha256', 'wrong')]:
        with pytest.raises(ValueError):
            preparation_seconds({**record, field: value}, 'plan', result)
    for values in ([], [1, 2], [float('nan')], [float('inf')], [-1], [11]):
        changed = deepcopy(record)
        changed['stage_seconds']['dataset_for'] = values
        with pytest.raises(ValueError):
            preparation_seconds(changed, 'plan', result)
