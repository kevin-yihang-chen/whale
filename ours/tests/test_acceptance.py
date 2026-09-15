"""Constructed verifier outcomes are unit fixtures, never experiment results."""

import ast
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import pytest

from ours.acceptance import CandidateMetrics, EvidenceConstrainedAcceptance
from ours.adapter import WHALEAcceptanceAdapter
from ours.evidence import AuditReceipt, EvaluationIdentity, PairPrediction, VisualPair, VisualPairEvaluator, fingerprint


def digest(label):
    return fingerprint(label)


IDENTITY = EvaluationIdentity(*(digest(s) for s in ("weights", "H", "C", "decode", "verifier")), phase="1")
CANDIDATES = tuple(CandidateMetrics(n, digest(n), a, t) for n, a, t in
                   (("incoming", .6, 3), ("shortcut", .9, 1), ("grounded", .8, 2)))


def audits():
    return {c.name: AuditReceipt(IDENTITY, c.harness_sha256, "C", ("a", "b"), values)
            for c, values in zip(CANDIDATES, (((1, 1), (0, 0)), ((1, 0), (0, 1)), ((1, 1), (0, 0))))}


def test_pair_success_requires_both_answers():
    pairs = [VisualPair("p", "source", "Which bar?", (digest("img1"), digest("img2")), ("A", "B"))]
    identity = replace(IDENTITY, audit_data_sha256=fingerprint([asdict(p) for p in pairs]))
    result = VisualPairEvaluator(lambda a, b: a == b).evaluate(
        pairs, [PairPrediction("p", ("A", "A"))], identity=identity, harness_sha256=digest("h"), role="C")
    assert result.paired_accuracy == 0
    assert result.marginal_accuracy == .5


@pytest.mark.parametrize("rows", [[], [PairPrediction("unknown", ("A", "B"))],
                                  [PairPrediction("p", ("A", "B"))] * 2])
def test_pair_coverage_fails_closed(rows):
    pair = VisualPair("p", "s", "q", (digest("i"), digest("j")), ("A", "B"))
    identity = replace(IDENTITY, audit_data_sha256=fingerprint([asdict(pair)]))
    with pytest.raises(ValueError):
        VisualPairEvaluator(lambda a, b: a == b).evaluate([pair], rows, identity=identity,
                                                       harness_sha256=digest("h"), role="C")


def test_gate_recovers_candidate_pruned_by_task_pareto():
    decision = EvidenceConstrainedAcceptance().select(CANDIDATES, incoming="incoming", identity=IDENTITY, audits=audits())
    assert decision.selected == "grounded"  # Dominated on A/turns by the rejected shortcut.
    assert decision.rejected == ("shortcut",)


@pytest.mark.parametrize("mode,winner", [("off", "shortcut"), ("paired", "grounded"),
                                        ("marginal_gate", "shortcut"), ("marginal_rank", "shortcut"),
                                        ("soft_pair", "grounded")])
def test_same_data_controls(mode, winner):
    assert EvidenceConstrainedAcceptance(mode).select(CANDIDATES, incoming="incoming", identity=IDENTITY,
                                                     audits=audits()).selected == winner


@pytest.mark.parametrize("field", ["weights_sha256", "search_data_sha256", "audit_data_sha256", "decode_sha256", "verifier_sha256", "phase"])
def test_stale_phase_or_configuration_rejected(field):
    changed = replace(IDENTITY, **{field: "2" if field == "phase" else digest("different")})
    with pytest.raises(ValueError, match="Stale"):
        EvidenceConstrainedAcceptance().select(CANDIDATES, incoming="incoming", identity=changed, audits=audits())


@pytest.mark.parametrize("role", ["V", "T", "engineering"])
def test_no_validation_or_test_selection(role):
    rows = {n: replace(r, role=role) for n, r in audits().items()}
    with pytest.raises(ValueError, match="optimization audit C"):
        EvidenceConstrainedAcceptance().select(CANDIDATES, incoming="incoming", identity=IDENTITY, audits=rows)


@pytest.mark.parametrize("fault", ["missing", "code", "pairs"])
def test_incompatible_audit_rejected(fault):
    rows = audits()
    if fault == "missing":
        del rows["shortcut"]
    elif fault == "code":
        rows["shortcut"] = replace(rows["shortcut"], harness_sha256=digest("new code"))
    else:
        rows["shortcut"] = replace(rows["shortcut"], pair_ids=("a", "other"))
    with pytest.raises(ValueError):
        EvidenceConstrainedAcceptance().select(CANDIDATES, incoming="incoming", identity=IDENTITY, audits=rows)


def test_fixed_phase_reference_prevents_incremental_degradation():
    rows = audits()
    rows["grounded"] = replace(rows["grounded"], correctness=((0, 0), (0, 0)))
    decision = EvidenceConstrainedAcceptance(epsilon=.25).select(CANDIDATES, incoming="incoming", identity=IDENTITY, audits=rows)
    assert decision.selected == "incoming"


def test_zero_and_positive_epsilon():
    assert EvidenceConstrainedAcceptance(epsilon=.5).select(CANDIDATES, incoming="incoming", identity=IDENTITY,
                                                          audits=audits()).selected == "shortcut"


def test_actual_pair_manifest_must_match_identity():
    pair = VisualPair("p", "s", "q", (digest("i"), digest("j")), ("A", "B"))
    with pytest.raises(ValueError, match="Actual pair manifest"):
        VisualPairEvaluator(lambda a, b: a == b).evaluate(
            [pair], [PairPrediction("p", ("A", "B"))], identity=IDENTITY, harness_sha256=digest("h"), role="C")


def test_tolerance_boundary_uses_exact_binary_counts():
    candidates = CANDIDATES[:2]
    rows = {c.name: AuditReceipt(IDENTITY, c.harness_sha256, "C", tuple(str(i) for i in range(10)),
                                ((1, 1),) * n + ((0, 0),) * (10 - n))
            for c, n in zip(candidates, (8, 7))}
    decision = EvidenceConstrainedAcceptance(epsilon=.1).select(candidates, incoming="incoming",
                                                               identity=IDENTITY, audits=rows)
    assert decision.selected == "shortcut"  # 0.7 - 0.8 is exactly -0.1 here.


@pytest.mark.parametrize("epsilon", [-1, float("nan"), float("inf"), 1.01])
def test_invalid_tolerance(epsilon):
    with pytest.raises(ValueError):
        EvidenceConstrainedAcceptance(epsilon=epsilon)


@pytest.mark.parametrize("stage", sorted(WHALEAcceptanceAdapter.STAGES))
def test_off_delegates_without_auditing(stage):
    def forbidden(*_):
        raise AssertionError("Off path must not load audits or change upstream selection")
    adapter = WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance("off"))
    selected, receipt = adapter.decide(stage=stage, original_selector=lambda: "upstream_exact_tie",
                                       incoming="incoming", archive_provider=forbidden, audit_provider=forbidden,
                                       identity=IDENTITY)
    assert selected == "upstream_exact_tie"
    assert receipt["audit_calls"] == 0


@pytest.mark.parametrize("stage", ["initial", "ordinary", "early_stop"])
def test_enabled_paths_use_same_gate(stage):
    adapter = WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance())
    selected, _ = adapter.decide(stage=stage, original_selector=lambda: "shortcut", incoming="incoming",
                                 archive_provider=lambda: CANDIDATES, audit_provider=lambda c: audits()[c.name], identity=IDENTITY)
    assert selected == "grounded"


def test_resume_roundtrip_and_tamper():
    adapter = WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance())
    kwargs = dict(original_selector=lambda: "shortcut", incoming="incoming", archive_provider=lambda: CANDIDATES,
                  audit_provider=lambda c: audits()[c.name], identity=IDENTITY)
    _, receipt = adapter.decide(stage="ordinary", **kwargs)
    saved = json.loads(json.dumps(receipt))
    assert adapter.decide(stage="resume", resume_receipt=saved, **kwargs)[0] == "grounded"
    saved["decision"]["selected"] = "shortcut"
    with pytest.raises(ValueError, match="differs"):
        adapter.decide(stage="resume", resume_receipt=saved, **kwargs)
    with pytest.raises(ValueError, match="matching decision receipt"):
        adapter.decide(stage="resume", **kwargs)


def test_off_delegates_actual_pinned_upstream_selectors(tmp_path):
    """Execute upstream selector function bodies, not its unavailable model runtime."""
    path = Path(__file__).resolve().parents[2] / "upstream/WHALE/domains/chess_puzzles/meta_harness/meta_harness_chess_puzzle.py"
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                 {"pick_accepted", "find_early_stop_candidate"}]
    namespace = {"Path": Path, "Any": Any, "get_accepted_harness": lambda _: "existing"}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    target = tmp_path / "harnesses/h1/harness.py"
    target.parent.mkdir(parents=True)
    target.write_text("# Selector file-existence fixture only\n")
    frontier = {"_best": {"harness": "h1"}}
    adapter = WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance("off"))
    for stage in ("initial", "ordinary", "resume"):
        selected, _ = adapter.decide(stage=stage, original_selector=lambda: namespace["pick_accepted"](frontier, tmp_path),
                                     incoming="existing", archive_provider=lambda: (), audit_provider=lambda _: None,
                                     identity=IDENTITY)
        assert selected == "h1"
    rows = [{"harness": "b", "avg_success_rate": 1., "avg_mean_turn_count": 2},
            {"harness": "a", "avg_success_rate": 1., "avg_mean_turn_count": 2}]
    baseline = lambda: namespace["find_early_stop_candidate"](rows, ["a", "b"], 1.)["harness"]
    selected, _ = adapter.decide(stage="early_stop", original_selector=baseline, incoming="existing",
                                 archive_provider=lambda: (), audit_provider=lambda _: None, identity=IDENTITY)
    assert selected == "a"
