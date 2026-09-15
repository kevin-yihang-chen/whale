from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json

import pytest

from ours.glm_gateway import BudgetJournal, completed_usage, cost, merge_usage


def test_concurrent_reservations_cannot_exceed_limit(tmp_path):
    path = tmp_path / "journal.jsonl"

    def attempt(_):
        try:
            return BudgetJournal(path).reserve(131072)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(filter(None, pool.map(attempt, range(50))))
    report = BudgetJournal(path).summary()
    assert len(ids) == len(set(ids)) == 28
    assert Decimal(report["held_cny"]) <= Decimal("45")
    assert Decimal(report["available_for_reservation_cny"]) < cost(1048576, 131072)


def test_restart_preserves_unresolved_charge(tmp_path):
    path = tmp_path / "journal.jsonl"
    ident = BudgetJournal(path).reserve(1024)
    journal = BudgetJournal(path)
    assert journal.summary()["unresolved_calls"] == 1
    journal.settle(ident, {"input_tokens": 100, "output_tokens": 30, "cache_read_input_tokens": 20})
    assert journal.summary()["conservative_usage_cny"] == "0.00024"
    assert journal.summary()["unresolved_calls"] == 0
    with pytest.raises(ValueError):
        journal.settle(ident, {"input_tokens": 1, "output_tokens": 1})


def test_unknown_usage_or_server_tools_keep_full_reservation(tmp_path):
    journal = BudgetJournal(tmp_path / "journal.jsonl")
    ident = journal.reserve(256)
    before = journal.summary()["held_cny"]
    with pytest.raises(ValueError):
        journal.settle(ident, {"input_tokens": 4, "output_tokens": 5,
                               "server_tool_use": {"web_search_requests": 1}})
    with pytest.raises(ValueError):
        journal.settle(ident, {"input_tokens": 9999999999, "output_tokens": 1})
    assert journal.summary()["held_cny"] == before


def test_probe_charge_import_is_idempotent(tmp_path):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"status": "CONNECTED", "response": {
        "model": "glm-5.3-flash", "usage": {"input_tokens": 17, "output_tokens": 26}}}))
    journal = BudgetJournal(tmp_path / "journal.jsonl")
    journal.initialize_probe(probe)
    journal.initialize_probe(probe)
    assert journal.summary()["conservative_usage_cny"] == "0.000121"


def test_stream_usage_uses_cumulative_maximum():
    usage = {}
    merge_usage(usage, {"input_tokens": 100, "output_tokens": 1})
    merge_usage(usage, {"output_tokens": 40})
    merge_usage(usage, {"output_tokens": 40})
    assert usage == {"input_tokens": 100, "output_tokens": 40}
    with pytest.raises(ValueError):
        merge_usage(usage, {"output_tokens": -1})
    with pytest.raises(ValueError):
        cost(1, True)


def test_partial_stream_cannot_release_reservation():
    start = b'data: {"type":"message_start","message":{"model":"glm-5.3-flash","usage":{"input_tokens":17,"output_tokens":0}}}\n\n'
    delta = b'data: {"type":"message_delta","usage":{"output_tokens":26}}\n\n'
    with pytest.raises(ValueError):
        completed_usage(start + delta, True)
    assert completed_usage(start + delta + b'data: {"type":"message_stop"}\n\n', True) == {"input_tokens":17,"output_tokens":26}
