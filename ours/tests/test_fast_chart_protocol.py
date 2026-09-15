import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ours.fast_chart_protocol import CompactExecutionBudget, schedule
from ours.plotqa_multitask import balanced_assignment


class CompactProtocolTests(unittest.TestCase):
    def test_old_charges_and_new_reservations_share_the_same_cap(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger.jsonl'
            rows = [{'event': 'reserve', 'id': 'old', 'stage': 'pilot', 'gpu_hours': '7', 'gpus': 1},
                    {'event': 'bind', 'id': 'old', 'job_id': '1'},
                    {'event': 'settle', 'id': 'old', 'job_id': '1', 'gpu_hours': '7'}]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            budget = CompactExecutionBudget(path)
            budget.reserve(phase='setup', gpus=1, time_limit_seconds=13*3600, purpose='fixture')
            with self.assertRaises(ValueError):
                budget.reserve(phase='setup', gpus=1, time_limit_seconds=1, purpose='over cap')
            budget.reserve(phase='core', gpus=4, time_limit_seconds=10*3600, purpose='fixture')
            self.assertEqual(budget.compact_summary()['remaining_unreserved'], '40')

    def test_condition_graph_shares_only_matching_prefixes(self):
        data = schedule()
        self.assertEqual(len(data['runs']), 18)
        for seed in data['seeds']:
            rows = {r['condition']: r for r in data['runs'] if r['seed'] == seed}
            self.assertEqual(rows['whale']['first_stage_id'], rows['veto']['first_stage_id'])
            self.assertNotEqual(rows['whale']['search_id'], rows['harness_only']['search_id'])
            self.assertIsNone(rows['weight_only']['search_id'])

    def test_assignment_preserves_all_sources_and_rejects_infeasible_balance(self):
        rows = [{'source_table_id': str(i)} for i in range(6)]
        allowed = {'0': ['comparison'], '1': ['comparison'], '2': ['comparison', 'read_value'],
                   '3': ['read_value', 'difference'], '4': ['read_value', 'difference'], '5': ['difference']}
        assignment = balanced_assignment(rows, allowed)
        self.assertEqual(set(assignment), set(allowed))
        self.assertEqual([list(assignment.values()).count(t) for t in ('comparison','read_value','difference')], [2,2,2])
        with self.assertRaises(ValueError):
            balanced_assignment(rows, {str(i): ['comparison'] for i in range(6)})


if __name__ == '__main__':
    unittest.main()
