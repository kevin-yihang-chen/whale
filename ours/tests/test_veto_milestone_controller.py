"""Sequential execution must retain failed costs and never duplicate submissions."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ours.veto_milestone_controller import Milestones, active_jobs, recovery_finished


class MilestoneExecutionTest(unittest.TestCase):
    def test_recovery_must_finish_and_release_controller(self):
        self.assertFalse(recovery_finished({'status': 'COMPLETED'}, process_alive=True))
        self.assertTrue(recovery_finished({'status': 'COMPLETED'}, process_alive=False))
        for state, alive in [('FAILED', True), ('RUNNING', False)]:
            with self.assertRaises(ValueError):
                recovery_finished({'status': state}, process_alive=alive)
        self.assertEqual(active_jobs('123\n124_2\n'), ['123', '124_2'])
        with self.assertRaises(ValueError):
            active_jobs('slurm controller unavailable')

    def test_failed_allocation_charged_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_plan = root/'input.json'
            model_plan.write_text('{}')
            controller = Milestones({'output': tmp, 'gpu_ledger': str(root/'gpu.jsonl')})
            terminal = 'JobId=123 JobState=FAILED RunTime=00:12:00 AllocTRES=cpu=16,gres/gpu=2 ExitCode=1:0\n'
            with patch.object(controller, 'queue', return_value=[]), \
                    patch('ours.veto_milestone_controller.check'), \
                    patch('ours.veto_milestone_controller.shutil.disk_usage') as disk, \
                    patch('ours.veto_milestone_controller.subprocess.check_output', side_effect=['123\n', terminal]) as call:
                disk.return_value.free = 100*1024**3
                with self.assertRaisesRegex(ValueError, 'without successful completion'):
                    controller.allocation('native', 'ours/run_joint_training_phase2.sh', model_plan,
                        gpus=2, seconds=3600, workspace_gib=30)
            self.assertEqual(call.call_count, 2)
            summary = controller.budget.summary()
            self.assertEqual(summary['charged_gpu_hours']['pilot'], '0.4')
            self.assertFalse(summary['pending_allocations'])
            self.assertTrue((root/'native.slurm-terminal.txt').exists())

    def test_ambiguous_submission_keeps_entire_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = Milestones({'output': tmp, 'gpu_ledger': str(root/'gpu.jsonl')})
            with patch.object(controller, 'queue', return_value=[]), \
                    patch('ours.veto_milestone_controller.check'), \
                    patch('ours.veto_milestone_controller.shutil.disk_usage') as disk, \
                    patch('ours.veto_milestone_controller.subprocess.check_output',
                        side_effect=subprocess.TimeoutExpired('sbatch', 60)) as call:
                disk.return_value.free = 100*1024**3
                with self.assertRaises(subprocess.TimeoutExpired):
                    controller.allocation('native', 'ours/run_joint_training_phase2.sh', root/'unused.json',
                        gpus=2, seconds=3600, workspace_gib=30)
            self.assertEqual(call.call_count, 1)
            summary = controller.budget.summary()
            self.assertEqual(summary['reserved_gpu_hours']['pilot'], '2')
            self.assertIsNone(summary['pending_allocations'][0]['job_id'])
            rows = [json.loads(line) for line in (root/'gpu.jsonl').read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows], ['reserve'])


if __name__ == '__main__':
    unittest.main()
