"""Proposer budget scenarios, not observed usage or spending authorization."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json

PRICE_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICE_CHECKED = "2026-09-09"


def proposer_sessions(total_steps: int, round_steps: int, iterations: int) -> int:
    """Pinned alternate.sh skips MH after the final weight phase."""
    for value in (total_steps, round_steps, iterations):
        if type(value) is not int or value < 1:
            raise ValueError("Budgets must be positive integers")
    cycles = (total_steps + round_steps - 1) // round_steps
    return (cycles - 1) * iterations


def opus_cost(*, input_tokens: int, output_tokens: int, cache_write_tokens: int = 0,
              cache_read_tokens: int = 0) -> Decimal:
    """Opus 4.7 standard global API prices; 5-minute cache writes.

    Categories are disjoint. Include all requests in a proposer session, including
    tool turns and billed thinking. No Batch discount or local GPU cost assumed.
    """
    counts = (input_tokens, output_tokens, cache_write_tokens, cache_read_tokens)
    if any(type(v) is not int or v < 0 for v in counts):
        raise ValueError("Token counts must be nonnegative integers")
    return sum(Decimal(v) * Decimal(p) for v, p in zip(counts, ("5", "25", "6.25", ".50"))) / 1000000


def scenarios() -> dict:
    sessions = proposer_sessions(256, 13, 5)
    matrix = 2 * 2 * 3 * (40 + 3 * sessions)
    rows = []
    for name, tokens_in, tokens_out in (("short", 50000, 5000), ("medium", 200000, 20000),
                                       ("long", 1000000, 50000)):
        cost = opus_cost(input_tokens=tokens_in, output_tokens=tokens_out)
        rows.append({"scenario": name, "session_input_tokens": tokens_in, "session_output_tokens": tokens_out,
                     "session_usd": str(cost), "20_sessions_usd": str(20 * cost),
                     "one_default_chess_whale_run_usd": str(sessions * cost),
                     "60_run_matrix_if_chess_schedule_usd": str(matrix * cost)})
    glm_session = Decimal(".2") * Decimal(".15") + Decimal(".02") * Decimal(".50")
    return {"kind": "budget_scenarios_not_measurements", "model": "claude-opus-4-7",
            "price_checked": PRICE_CHECKED, "price_source": PRICE_SOURCE,
            "currency": "USD", "default_chess_whale_sessions": sessions, "main_matrix_sessions": matrix,
            "matrix_assumption": "2 domains x 2 sizes x 3 seeds; weight-only 0, harness-only 40, "
                                 "FST/WHALE/VETO 95 sessions each; same chess schedule for illustration only",
            "excluded": ["GPU", "tax", "retry overhead", "additional ablations", "baseline reproduction"],
            "caveat": "No cache discount assumed; cache writes can add cost. These are scenarios, not upper bounds.",
            "rows": rows,
            "glm_flash_medium_scenario": {
                "source": "https://docs.z.ai/guides/overview/pricing", "price_checked": PRICE_CHECKED,
                "regular_input_usd_per_million": ".15", "regular_output_usd_per_million": ".50",
                "session_usd": str(glm_session), "20_sessions_usd": str(20 * glm_session),
                "95_sessions_usd": str(sessions * glm_session), "3900_sessions_usd": str(matrix * glm_session),
                "promotion": "50% discount ends 2026-09-09 24:00 UTC+8; future budgets use regular rates",
                "limitation": "Equal tokens assumed; proposer quality, retries and actual tokenizers can differ."},
            "paid_sessions_executed": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(scenarios(), indent=2))
