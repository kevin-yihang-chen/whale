"""Resource failures must consume budget and ambiguous submissions retain it."""
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from ours.research_budget import ExecutionBudget, terminal_allocation


class ResearchBudgetTests(unittest.TestCase):
    def test_failed_job_cost_wrong_identity_and_double_settlement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = ExecutionBudget(root/'ledger.jsonl')
            ident = journal.reserve(stage='pilot', gpus=2, time_limit_seconds=3600, purpose='fixture')
            journal.bind(ident, '123')
            raw = 'JobId=123 JobState=TIMEOUT ExitCode=0:15 RunTime=00:30:00 AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h800=2'
            terminal = root/'terminal.txt'
            terminal.write_text(raw)
            with self.assertRaises(ValueError):
                journal.settle(ident, job_id='124', gpu_hours=1, terminal_path=terminal)
            with self.assertRaises(ValueError):
                journal.settle(ident, job_id='123', gpu_hours=0, terminal_path=terminal)
            journal.settle(ident, job_id='123', gpu_hours=1, terminal_path=terminal)
            self.assertEqual(journal.summary()['charged_gpu_hours']['pilot'], '1')
            with self.assertRaises(ValueError):
                journal.settle(ident, job_id='123', gpu_hours=1, terminal_path=terminal)

    def test_unbound_reservations_survive_restart_and_limit_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ledger.jsonl'
            journal = ExecutionBudget(path)
            ident = journal.reserve(stage='pilot', gpus=4, time_limit_seconds=90000, purpose='fixture')
            journal = ExecutionBudget(path)
            with self.assertRaises(ValueError):
                journal.bind(ident, 'uncertain sbatch response')
            self.assertEqual(journal.summary()['reserved_gpu_hours']['pilot'], '100')
            with self.assertRaises(ValueError):
                journal.reserve(stage='pilot', gpus=1, time_limit_seconds=1, purpose='over budget')
            with self.assertRaises(ValueError):
                journal.reserve(stage='main', gpus=1, time_limit_seconds=1, purpose='before go evidence')

    def test_runtime_with_days_and_pending_allocations(self):
        raw = 'JobId=123 JobState=FAILED RunTime=1-01:30:00 AllocTRES=gres/gpu=4'
        self.assertEqual(terminal_allocation(raw, '123')['gpu_hours'], Decimal('102'))
        self.assertIsNone(terminal_allocation(raw.replace('FAILED','RUNNING'), '123'))
        with self.assertRaises(ValueError):
            terminal_allocation(raw.replace('gres/gpu=4', 'cpu=4'), '123')


if __name__ == '__main__':
    unittest.main()
