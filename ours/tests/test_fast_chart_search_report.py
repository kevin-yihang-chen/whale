import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ours.fast_chart_search_report import answer_categories, counts, latex, reconstruct
from ours.chart_answer_protocol import encode_truth
from ours.fast_chart_paper import search_table, ROOT
from ours.visual_task import file_sha256


class SearchReportTests(unittest.TestCase):
    def test_format_diagnosis_uses_the_shared_parser_and_answer_type(self):
        records = [dict(sample_id='a', raw_answer='Final answer: 9', committed_answer='9', correct=0),
            dict(sample_id='b', raw_answer='Final answer: A', committed_answer='A', correct=0),
            dict(sample_id='c', raw_answer='Final answer: A', committed_answer='A', correct=1)]
        truth = {'a':encode_truth('numeric',10), 'b':encode_truth('numeric',10), 'c':encode_truth('binary','A')}
        self.assertEqual(answer_categories(records,truth), {'correct':1,'parseable_wrong':1,'unparseable':1})
        records[0]['committed_answer']='10'
        with self.assertRaisesRegex(ValueError, 'fixed host scorer'):
            answer_categories(records,truth)

    def test_complete_pair_counts_distinguish_single_and_paired_success(self):
        # Constructed correctness patterns, never published as model scores.
        item = {'audit': SimpleNamespace(correctness=[(1,1)]*49+[(1,0)]*12+[(0,0)]*3),
            'h': {'records': [{'correct':1}]*107+[{'correct':0}]*21},
            'candidate': SimpleNamespace(name='h0', mean_turns=2.)}
        row = counts(item)
        self.assertEqual((row['C_single_correct'], row['C_both_correct']), (110,49))
        self.assertEqual(row['C_zero_one_two_correct'], [3,12,49])
        item['audit'].correctness.pop()
        with self.assertRaisesRegex(ValueError, 'all registered'):
            counts(item)

    def test_partial_archive_cannot_be_published_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'result.json').write_text(json.dumps({'status':'WAITING_EVALUATION_h3'}))
            with self.assertRaisesRegex(ValueError, 'partial search'):
                reconstruct(root)

    def test_failed_slot_has_no_zero_score_and_equal_choices_have_no_benefit_claim(self):
        report = {'rows':[], 'failed_slots':{'h2':{}}, 'equivalent_decisions':True,
            'decisions':{name:{'accepted_harness':'h0'} for name in ('whale','veto','marginal_gate')}}
        text = latex(report)
        self.assertIn('Failed; no score assigned', text)
        self.assertIn('no evidence of an independent VETO', text)
        self.assertNotIn('0/128', text)

    def test_manuscript_rejects_changed_underlying_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root/'actual-records.json'
            evidence.write_text('[]')
            report = {'status':'COMPLETE_RECONSTRUCTED_SEARCH_REPORT',
                'rows':[{'candidate':'h0','H_correct':0,'C_single_correct':0,'C_both_correct':0}],
                'failed_slots':{name:{} for name in ('h1','h2','h3')}, 'equivalent_decisions':True,
                'decisions':{name:{'accepted_harness':'h0'} for name in ('whale','veto','marginal_gate')},
                'evidence_sha256':{str(evidence):file_sha256(evidence)}}
            result, table = root/'result.json', root/'candidate-selection.tex'
            result.write_text(json.dumps(report))
            table.write_text(latex(report))
            (root/'manifest.json').write_text(json.dumps({'report_sha256':file_sha256(result),
                'table_sha256':file_sha256(table), 'source_sha256':file_sha256(ROOT/'ours/fast_chart_search_report.py')}))
            self.assertEqual(search_table(root)[0], table.read_text())
            evidence.write_text('["changed"]')
            with self.assertRaisesRegex(ValueError, 'evidence source changed'):
                search_table(root)


if __name__ == '__main__':
    unittest.main()
