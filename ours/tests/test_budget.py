from decimal import Decimal

import pytest

from ours.budget import opus_cost, proposer_sessions, scenarios


@pytest.mark.parametrize("steps,round_steps,expected", [(256, 13, 95), (26, 13, 5), (13, 13, 0), (1, 13, 0)])
def test_no_search_after_final_weight_update(steps, round_steps, expected):
    assert proposer_sessions(steps, round_steps, 5) == expected


def test_token_categories_are_disjoint():
    assert opus_cost(input_tokens=200000, output_tokens=20000) == Decimal("1.5")
    assert opus_cost(input_tokens=0, output_tokens=0, cache_write_tokens=1000000,
                     cache_read_tokens=1000000) == Decimal("6.75")


def test_main_matrix_excludes_weight_only_proposer_calls():
    result = scenarios()
    assert result["main_matrix_sessions"] == 3900
    assert Decimal(result["rows"][1]["60_run_matrix_if_chess_schedule_usd"]) == 5850
