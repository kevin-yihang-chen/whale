"""Check the independent audit's distinction between malformed and illegal UCI."""
import importlib.util
import unittest

from ours.audit_chess_bootstrap import action_properties


@unittest.skipUnless(importlib.util.find_spec('chess'), 'Run in the isolated training runtime with python-chess')
class BootstrapAuditTests(unittest.TestCase):
    def test_same_square_is_malformed_not_an_illegal_move(self):
        import chess
        self.assertEqual(action_properties(chess.Board(), 'a1a1'), (False, False))

    def test_well_formed_move_can_be_illegal(self):
        import chess
        self.assertEqual(action_properties(chess.Board(), 'e2e5'), (True, False))

    def test_legal_does_not_imply_a_reference_solution(self):
        import chess
        self.assertEqual(action_properties(chess.Board(), 'e2e4'), (True, True))
