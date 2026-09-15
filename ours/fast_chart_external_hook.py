"""Shared ChartQA host reward dispatch in native workers; candidates cannot override it."""
from .chartqa_open_answer import PROTOCOL,relaxed_correct


def compute_score(data_source,solution_str,ground_truth,extra_info=None):
    if data_source!=PROTOCOL:raise ValueError('External scorer requires the registered ChartQA domain')
    correct=float(relaxed_correct(solution_str,ground_truth))
    return {'score':correct,'acc':correct}


def prepare_worker():
    from .training_bootstrap import prepare_worker as original
    result=original()
    from . import visual_harness_loop,visual_evidence_tool_loop
    visual_harness_loop.compute_score=compute_score
    visual_evidence_tool_loop.compute_score=compute_score
    return result
