"""The user-directed route must submit only the bounded visual screen."""
from pathlib import Path
import unittest
from unittest.mock import Mock, call

from ours.visual_priority_controller import execute_visual


class VisualPriorityTest(unittest.TestCase):
    def test_visual_route_has_no_chess_dependency_or_training(self):
        controller = Mock()
        controller.allocation.return_value = ('123', Path('terminal.txt'))
        visual = Path('visual.json')
        self.assertEqual(execute_visual(controller, visual), ('123', Path('terminal.txt')))
        self.assertEqual(controller.mock_calls, [
            call.command('check_visual_service', 'ours.native_visual_service',
                         '--phase', 'check', '--plan', visual, visual=True),
            call.allocation('run_visual_service', 'ours/run_native_visual_service.sh', visual,
                            gpus=1, seconds=2400, workspace_gib=2)])

    def test_failed_visual_preflight_cannot_allocate(self):
        controller = Mock()
        controller.command.side_effect = ValueError('changed model')
        with self.assertRaisesRegex(ValueError, 'changed model'):
            execute_visual(controller, Path('visual.json'))
        controller.allocation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
