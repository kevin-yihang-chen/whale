"""Validate real E4 evidence before expanding the calibrated experiment."""
import json
from pathlib import Path

from .visual_task import file_sha256


def read(path):
    return json.loads(Path(path).read_text())


def require_parameter_update(execution, transition, *, stage):
    if execution['status'] != 'COMPLETE_COMPACT_NATIVE_STAGE':
        raise ValueError('The native stage did not finish')
    if execution['nonzero_gradient_events'] < 1:
        raise ValueError('No observed image-loss gradient; expansion is not admitted')
    names = ('native_fp32', 'exported_bf16')
    if stage == 2:
        names += ('stage2_native_fp32_change',)
    for name in names:
        value = transition[name]
        if value['status'] != 'CHANGED' or value['changed_elements'] <= 0:
            raise ValueError(f'No real {name} update; preserve the result as a diagnostic')
    if transition['export_exact_native_bf16_cast'] is not True:
        raise ValueError('Serving export was not verified against native parameters')


def verified_calibration(path):
    """Do not replace an unchanged winning LR with a more favorable runner-up."""
    cfg = read(path)
    if cfg['status'] != 'COMPLETE_SHARED_CHART_CONFIGURATION':
        raise ValueError('Shared h0/LR calibration is incomplete')
    calibration = Path(cfg['h0_calibration'])
    if file_sha256(calibration) != cfg['h0_calibration_sha256']:
        raise ValueError('Initial harness calibration changed')
    h0 = read(calibration)
    if h0['status'] != 'COMPLETE_SHARED_H0_CALIBRATION' or h0['harness'] != cfg['harness']:
        raise ValueError('The common harness differs from its calibration')
    selected = cfg['results'][str(cfg['learning_rate'])]
    result_path, training_path = Path(selected['result_path']), Path(selected['training_plan'])
    if (file_sha256(result_path) != selected['result_sha256'] or
            file_sha256(training_path) != selected['training_plan_sha256']):
        raise ValueError('Selected calibration evidence changed')
    result, training = read(result_path), read(training_path)
    if training['harness_sha256'] != file_sha256(Path(cfg['harness'])):
        raise ValueError('Calibrated harness contents changed')
    transition_path = result_path.parent / 'transition.json'
    if file_sha256(transition_path) != result['transition_sha256']:
        raise ValueError('Selected calibration parameter evidence changed')
    execution = read(Path(training['output']) / 'execution-result.json')
    if execution['plan_sha256'] != file_sha256(training_path):
        raise ValueError('Selected calibration ran another training plan')
    require_parameter_update(execution, read(transition_path), stage=1)
    return cfg
