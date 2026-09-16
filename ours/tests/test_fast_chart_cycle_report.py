import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ours.fast_chart_cycle_report import difference, latex, reconstruct
from ours.fast_chart_paper import cycle_table, ROOT
from ours.visual_task import file_sha256


class CycleReportTests(unittest.TestCase):
    def test_net_pair_gain_preserves_harmed_examples_and_is_not_attributed_to_gate(self):
        before = {'a': {'source_id':'x','paired':0,'marginal':.5},
                  'b': {'source_id':'y','paired':1,'marginal':1}}
        after = {'a': {'source_id':'x','paired':1,'marginal':1},
                 'b': {'source_id':'y','paired':0,'marginal':.5}}
        change = difference(before, after)
        self.assertEqual((change['paired_improved'], change['paired_worsened'], change['paired_pp']), (1,1,0))
        self.assertIn('not a causal gate comparison', change['interpretation'])
        after['a']['source_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            difference(before, after)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            difference(before, {'a':after['a']})

    def test_running_cycle_cannot_be_reported_as_complete(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/'result.json'
            p.write_text(json.dumps({'status':'WAITING_VETO_CONTINUATION'}))
            with self.assertRaisesRegex(ValueError, 'complete prospectively'):
                reconstruct(p, Path(tmp)/'not-yet-available.json')

    def test_changed_primary_evidence_blocks_manuscript_even_when_table_is_unchanged(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root/'raw-records.json'
            evidence.write_text('[]')
            report = {'status':'COMPLETE_RECONSTRUCTED_FIRST_CYCLE', 'equivalent_decisions':True,
                'development':{k:{'images_correct':0,'pairs_correct':0} for k in ('initial','prefix','continued')},
                'training':{'successful_trajectories':0}, 'evidence_sha256':{str(evidence):file_sha256(evidence)}}
            result, table = root/'result.json', root/'cycle-development.tex'
            result.write_text(json.dumps(report)); table.write_text(latex(report))
            (root/'manifest.json').write_text(json.dumps({'report_sha256':file_sha256(result),
                'table_sha256':file_sha256(table), 'source_sha256':file_sha256(ROOT/'ours/fast_chart_cycle_report.py')}))
            self.assertIn('not substituted for unexecuted control', cycle_table(root)[0])
            evidence.write_text('["changed"]')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                cycle_table(root)


if __name__ == '__main__':
    unittest.main()
