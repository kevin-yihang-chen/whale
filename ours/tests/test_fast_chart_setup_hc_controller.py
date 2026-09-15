from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ours import fast_chart_setup_hc_controller as controller


class SetupHandoffTests(unittest.TestCase):
    def test_unfinished_or_failed_prerequisite_never_submits_next_job(self):
        with patch.object(controller, 'admission', return_value={}), \
                patch.object(controller, 'read', return_value={'kind': 'compact_chart_throughput_calibration'}), \
                patch.object(controller, 'terminal', return_value='PENDING') as terminal, \
                patch.object(controller, 'command') as command:
            result = controller.advance(Path('/tmp'), Path('admission'), Path('throughput'))
            self.assertEqual(result['status'], 'WAITING_THROUGHPUT_ALLOCATION')
            terminal.side_effect = RuntimeError('Allocation failed')
            with self.assertRaisesRegex(RuntimeError, 'Allocation failed'):
                controller.advance(Path('/tmp'), Path('admission'), Path('throughput'))
            command.assert_not_called()

    def test_completed_but_invisible_result_waits_without_submission(self):
        with patch.object(controller, 'admission', return_value={}), \
                patch.object(controller, 'read', return_value={'kind': 'compact_chart_throughput_calibration', 'output': '/tmp/output'}), \
                patch.object(controller, 'terminal', return_value='COMPLETED'), \
                patch.object(controller, 'completed_artifact', return_value=None), \
                patch.object(controller, 'freeze') as freeze, patch.object(controller, 'command') as command:
            result = controller.advance(Path('/tmp'), Path('admission'), Path('throughput'))
            self.assertEqual(result['status'], 'WAITING_THROUGHPUT_ARTIFACT')
            freeze.assert_not_called()
            command.assert_not_called()

    def test_one_setup_submission_and_existing_receipt_prevent_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            throughput = directory/'throughput.json'
            suite = {'kind': 'compact_chart_throughput_calibration', 'output': str(directory/'measurements')}
            record = {'plan': str(directory/'hc.json'), 'configuration': str(directory/'configuration.json')}
            policy = {'throughput_result': str(Path(suite['output'])/'result.json'), 'throughput_suite': str(throughput)}
            with patch.object(controller, 'admission', return_value=record), \
                    patch.object(controller, 'read', return_value=suite), \
                    patch.object(controller, 'terminal', side_effect=['COMPLETED', None, 'COMPLETED', 'PENDING']), \
                    patch.object(controller, 'completed_artifact', return_value={'status': 'complete'}), \
                    patch.object(controller, 'POLICY', directory/'policy.json'), \
                    patch.object(controller, 'freeze'), patch.object(controller, 'load_policy', return_value=policy), \
                    patch.object(controller, 'forecast', return_value={'formal_expansion_admitted': False}), \
                    patch.object(controller, 'command') as command:
                self.assertEqual(controller.advance(directory, Path('admission'), throughput)['status'], 'SUBMITTED_SETUP_HC')
                self.assertEqual(controller.advance(directory, Path('admission'), throughput)['status'], 'WAITING_SETUP_HC_ALLOCATION')
                command.assert_called_once_with(directory, 'setup-hc-submit',
                    ['ours.fast_chart_submit', '--kind', 'search', '--phase', 'setup', '--plan', record['plan']])
                self.assertTrue((directory/'measured-expansion-forecast.json').is_file())

    def test_formal_expansion_or_larger_allocation_is_not_this_scope(self):
        record = {'status': 'PREPARED_BOUNDED_SETUP_HC_MEASUREMENT', 'phase': 'setup',
            'maximum_additional_gpu_hours': 1, 'formal_expansion_admitted': False}
        for key, value in [('phase', 'core'), ('maximum_additional_gpu_hours', 2), ('formal_expansion_admitted', True)]:
            with patch.object(controller, 'read', return_value={**record, key: value}):
                with self.assertRaisesRegex(ValueError, 'bounded setup'):
                    controller.admission(Path('admission'))


if __name__ == '__main__':
    unittest.main()
