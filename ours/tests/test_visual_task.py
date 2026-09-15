import pytest

from ours.visual_task import binary_answer_verifier


@pytest.mark.parametrize("prediction,truth,expected", [
    ("A", "A", True), (" b.\n", "B", True), ("A!", "A", True),
    ("A or B", "A", False), ("The answer is A", "A", False),
    ("", "B", False), ("B", "A", False), ("A. extra", "A", False),
])
def test_engineering_verifier_does_not_extract_a_label_from_explanation(prediction, truth, expected):
    assert binary_answer_verifier(prediction, truth) is expected
