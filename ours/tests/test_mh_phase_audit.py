"""Check independent continuation review against native multi-turn CPU fixtures."""
from collections import deque
from copy import deepcopy
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('chess'), 'Requires native Chess runtime')
class MHPhaseAuditTests(unittest.TestCase):
    def test_legal_continuation_and_corrupted_verdicts(self):
        import chess
        from autoharness_chess_puzzle import runner
        from autoharness_chess_puzzle.harness import load_harness
        from ours.audit_mh_phase import board_review
        from ours.local_completion import CompletionResponse
        example = runner.example_from_mapping({'extra_info': {'puzzle_id': 'offline-fixture',
            'start_fen': chess.STARTING_FEN, 'solution_moves': ['e2e4', 'e7e5', 'g1f3']}})
        harness = load_harness('upstream/WHALE/domains/chess_puzzles/environments/chess_puzzle/base_harness.py')
        class ReplyFixture:
            def __init__(self, replies):
                self.replies = deque(replies)
            def complete_response(self, messages, *, max_tokens=None):
                return CompletionResponse(self.replies.popleft(), {'completion_tokens': 1})
        output = runner.run_puzzle_rollout(harness=harness, example=example,
            llm=ReplyFixture(['<move>e2e4</move>', '<move>g1f3</move>']),
            assistant_token_budget=8129, policy_max_tokens=8129)
        reviewed = board_review(example, output)
        self.assertTrue(reviewed['solved'])
        self.assertEqual(len(reviewed['steps']), 2)
        wrong = deepcopy(output)
        wrong['harness_trace'][0]['parsed_action'] = 'd2d4'
        with self.assertRaisesRegex(ValueError, 'legal reference'):
            board_review(example, wrong)
        wrong = deepcopy(output)
        wrong['reward'] = 0.
        with self.assertRaisesRegex(ValueError, 'Recorded reward'):
            board_review(example, wrong)
        malformed = runner.run_puzzle_rollout(harness=harness, example=example,
            llm=ReplyFixture(['<move>a1a1</move>', '<move>a1a1</move>']),
            assistant_token_budget=8129, policy_max_tokens=8129)
        self.assertFalse(board_review(example, malformed)['solved'])
        self.assertEqual(malformed['stop_condition'], 'malformed')


if __name__ == '__main__':
    unittest.main()
