import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ours import fast_chart_first_cycle_controller as c


class FirstCycleHandoffTests(unittest.TestCase):
    def test_pending_submission_is_never_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'plan.json'
            path.write_text(json.dumps({'seed': 42, 'bounds': {'gpus': 1, 'time_limit_seconds': 3600}}))
            receipt = root/'submission.json'
            receipt.write_text(json.dumps({'phase': 'setup', 'plan_sha256': c.file_sha256(path), 'job_id': '123'}))
            with patch.object(c, 'allocation', return_value=root/'allocation-result.json'), \
                    patch.object(c, 'terminal', return_value='PENDING'), patch.object(c, 'command') as command:
                self.assertFalse(c.wait_or_submit(root, path, 'search'))
                self.assertFalse(c.wait_or_submit(root, path, 'search'))
                command.assert_not_called()

    def test_ambiguous_submission_and_resource_expansion_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'plan.json'
            record = {'seed': 42, 'bounds': {'gpus': 1, 'time_limit_seconds': 3600}}
            path.write_text(json.dumps(record))
            (root/'submission.json').write_text(json.dumps({'phase': 'setup', 'plan_sha256': c.file_sha256(path)}))
            with patch.object(c, 'allocation', return_value=root/'allocation-result.json'), patch.object(c, 'command') as command:
                with self.assertRaisesRegex(ValueError, 'ambiguous'):
                    c.wait_or_submit(root, path, 'search')
                record['bounds']['gpus'] = 2
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, 'bounded first cycle'):
                    c.wait_or_submit(root, path, 'search')
                command.assert_not_called()

    def candidate_context(self, root):
        search = root/'search'
        search.mkdir()
        (root/'results').mkdir()
        request = {'identity': {'phase': 'fixed'}, 'reference': 'reference', 'harness_sha256': 'source'}
        (search/'evaluation-request-h1.json').write_text(json.dumps(request))
        plan = {**request, 'kind': 'compact_chart_search_evaluation', 'candidate': 'h1',
            'reference_plan': 'reference', 'output': str(root/'output')}
        (root/'results/fast-chart-seed42-first-search-h1-plan-20260915-v1.json').write_text(json.dumps(plan))
        return search

    def test_completed_allocation_waits_for_visible_result_before_attach(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search = self.candidate_context(root)
            with patch.object(c, 'ROOT', root), patch.object(c, 'SEARCH', search), \
                    patch.object(c, 'allocation', return_value=root/'allocation-result.json'), \
                    patch.object(c, 'wait_or_submit', return_value=True), \
                    patch.object(c, 'completed_artifact', return_value=None), patch.object(c, 'command') as command:
                self.assertEqual(c.candidate(root, 'h1'), 'WAITING_CANDIDATE_ARTIFACT')
                command.assert_not_called()

    def test_failed_candidate_requires_evidence_and_consumes_no_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search = self.candidate_context(root)
            end = root/'allocation-result.json'
            end.write_text(json.dumps({'state': 'FAILED'}))
            with patch.object(c, 'ROOT', root), patch.object(c, 'SEARCH', search), \
                    patch.object(c, 'allocation', return_value=end), patch.object(c, 'wait_or_submit') as submit, \
                    patch.object(c, 'completed_artifact', side_effect=[None, {'error': 'candidate failure'}]), \
                    patch.object(c, 'command') as command:
                self.assertEqual(c.candidate(root, 'h1'), 'WAITING_FAILED_CANDIDATE_EVIDENCE')
                self.assertEqual(c.candidate(root, 'h1'), 'CONSUMED_FAILED_CANDIDATE')
                submit.assert_not_called()
                self.assertEqual(command.call_count, 1)
                self.assertEqual(command.call_args.args[2][:2], ['ours.fast_chart_search', 'failed-evaluation'])


if __name__ == '__main__':
    unittest.main()
