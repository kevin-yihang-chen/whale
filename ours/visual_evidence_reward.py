"""Shared binary visual-task verification for native WHALE RSFT.

Method E4 uses the existing verifier r=v(answer, truth) and the original
success-filtered SFT objective. This callback is not VETO E1--E3, a reward-shaping
term, or a claim that the model used the image. Every visual condition must use
the same verifier. The current domain contract requires a single A/B answer.
"""
from .visual_task import binary_answer_verifier

DATA_SOURCE = 'visual_evidence_binary'


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Grade decoded answer text against dataset truth, ignoring claimed rewards.

    Ground truth and row metadata remain outside the model request. Invalid task
    identity or labels are data errors, not zero-valued task outcomes. Ordinary
    incorrect, empty or badly formatted model responses receive zero.
    """
    from .chart_answer_protocol import DATA_SOURCE as CHART_SOURCE, verify
    if data_source == CHART_SOURCE:
        correct = float(verify(solution_str, ground_truth))
        return {'score': correct, 'acc': correct}
    if data_source != DATA_SOURCE:
        raise ValueError('Visual reward requires the visual_evidence_binary domain')
    if not isinstance(ground_truth, str) or ground_truth not in ('A', 'B'):
        raise ValueError('Visual binary ground truth must be A or B')
    if not isinstance(solution_str, str):
        raise TypeError('Visual reward requires decoded response text')
    correct = float(binary_answer_verifier(solution_str, ground_truth))
    return {'score': correct, 'acc': correct}
