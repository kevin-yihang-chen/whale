from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from ours.fast_chart_search_report_v3 import counts, latex
from ours.fast_chart_paper import search_table
from ours.visual_task import file_sha256


def test_v3_counts_use_plan_bound_c256_size():
    item = {
        'audit': SimpleNamespace(correctness=[(1, 1)] * 203 + [(1, 0)] * 19
                                           + [(0, 1)] * 14 + [(0, 0)] * 20),
        'h': {'examples': 128, 'records': [{'correct': 1}] * 108 + [{'correct': 0}] * 20},
        'candidate': SimpleNamespace(name='h0', mean_turns=2.0),
    }
    row = counts(item)
    assert (row['H_correct'], row['H_total']) == (108, 128)
    assert (row['C_single_correct'], row['C_images']) == (439, 512)
    assert (row['C_both_correct'], row['C_pairs']) == (203, 256)

    item['h']['examples'] = 127
    with pytest.raises(ValueError, match='plan-bound'):
        counts(item)


def test_v3_latex_reports_dynamic_denominators_and_point_gate():
    report = {
        'rows': [{'candidate': 'h0', 'H_correct': 108, 'H_total': 128,
                  'C_single_correct': 439, 'C_images': 512,
                  'C_both_correct': 203, 'C_pairs': 256}],
        'failed_slots': {},
        'equivalent_decisions': True,
        'decisions': {name: {'accepted_harness': 'h0'}
                      for name in ('whale', 'veto', 'point_gate')},
    }
    text = latex(report)
    assert '108/128 & 439/512 & 203/256' in text
    assert 'Point-estimate paired gate h0' in text


def test_paper_accepts_complete_v3_report(tmp_path):
    rows = [
        {'candidate': name, 'H_correct': score, 'H_total': 128,
         'C_single_correct': 400 + score % 10, 'C_images': 512,
         'C_both_correct': 180 + score % 10, 'C_pairs': 256}
        for name, score in [('h0', 108), ('h1', 106), ('h2', 63), ('h3', 107)]
    ]
    report = {
        'status': 'COMPLETE_RECONSTRUCTED_SEARCH_REPORT',
        'selection_protocol': 'safety_v3',
        'rows': rows,
        'failed_slots': {},
        'equivalent_decisions': True,
        'decisions': {name: {'accepted_harness': 'h0'}
                      for name in ('whale', 'veto', 'point_gate')},
        'evidence_sha256': {},
    }
    report_path = tmp_path / 'result.json'
    table_path = tmp_path / 'candidate-selection.tex'
    manifest_path = tmp_path / 'manifest.json'
    report_path.write_text(json.dumps(report))
    table_path.write_text(latex(report))
    source = Path(__file__).parents[1] / 'fast_chart_search_report_v3.py'
    manifest_path.write_text(json.dumps({
        'report_sha256': file_sha256(report_path),
        'table_sha256': file_sha256(table_path),
        'source_sha256': file_sha256(source),
    }))

    table, evidence = search_table(tmp_path)
    assert table == latex(report)
    assert evidence['equivalent_decisions'] is True
