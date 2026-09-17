"""VETO-v3: uncertainty-aware counterfactual safety selection.

The audit is an optimization-set decision rule, not a population guarantee.
It keeps ordinary H accuracy as the objective and uses paired C outcomes only
to decide whether a candidate is sufficiently supported to enter that ranking.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from math import lcm
from typing import Mapping, Sequence, cast

from .acceptance import AcceptanceDecision, CandidateMetrics, TaskAccuracyAcceptance
from .evidence import AuditReceipt, EvaluationIdentity, finite_metric


def evidence_statistics(receipt: AuditReceipt) -> dict[str, Fraction]:
    """Return competence, fragility and paired success with their exact identity."""
    n = len(receipt.correctness)
    marginal = Fraction(sum(a + b for a, b in receipt.correctness), 2 * n)
    fragility = Fraction(sum(a != b for a, b in receipt.correctness), n)
    paired = Fraction(sum(a * b for a, b in receipt.correctness), n)
    if paired != marginal - fragility / 2:
        raise ValueError("Paired evidence decomposition is inconsistent")
    return {"marginal_competence": marginal, "counterfactual_fragility": fragility,
            "paired_accuracy": paired}


def _empirical_bootstrap_interval(values: Sequence[Fraction], tail_probability: float) -> tuple[Fraction, Fraction]:
    """Deterministic percentile bounds for the empirical paired-source bootstrap.

    The distribution is enumerated exactly, so there is no Monte Carlo noise.
    These remain approximate bootstrap bounds and carry no finite-population or
    post-training guarantee.
    """
    if len(values) < 2 or not 0 < tail_probability < .5:
        raise ValueError("Bootstrap bounds require at least two sources and a valid tail probability")
    scale = 1
    for value in values:
        if not isinstance(value, Fraction):
            raise ValueError("Bootstrap inputs must use exact paired-source fractions")
        scale = lcm(scale, value.denominator)
    support = Counter(int(value * scale) for value in values)
    probabilities = {value: count / len(values) for value, count in support.items()}
    distribution = {0: 1.0}
    for _ in values:
        updated: dict[int, float] = {}
        for total, probability in distribution.items():
            for value, mass in probabilities.items():
                updated[total + value] = updated.get(total + value, 0.0) + probability * mass
        distribution = updated

    def quantile(level: float) -> Fraction:
        cumulative = 0.0
        for total, probability in sorted(distribution.items()):
            cumulative += probability
            if cumulative + 1e-15 >= level:
                return Fraction(total, scale * len(values))
        return Fraction(max(distribution), scale * len(values))

    return quantile(tail_probability), quantile(1 - tail_probability)


class CounterfactualSafetyAcceptance(TaskAccuracyAcceptance):
    """E2-v3 safety audit followed by WHALE's original E3 task ranking.

    A candidate passes only when the source-paired bootstrap supports both
    marginal-competence non-inferiority and fragility non-increase. Ambiguous
    candidates are rejected at the fixed maximum audit size; this is VETO's
    abstention behavior. Incoming always remains eligible.
    """

    mode = "counterfactual_safety"

    def __init__(self, *, audit_pairs: int = 256, competence_epsilon: float = .01,
                 fragility_epsilon: float = 0., familywise_alpha: float = .05,
                 candidate_budget: int = 3):
        if type(audit_pairs) is not int or audit_pairs < 2:
            raise ValueError("A fixed audit needs at least two source pairs")
        if type(candidate_budget) is not int or candidate_budget < 1:
            raise ValueError("Candidate budget must be a positive integer")
        for value in (competence_epsilon, fragility_epsilon, familywise_alpha):
            finite_metric(value, probability=True)
        if not 0 < familywise_alpha < 1:
            raise ValueError("Familywise alpha must be strictly between zero and one")
        self.audit_pairs = audit_pairs
        self.competence_epsilon = competence_epsilon
        self.fragility_epsilon = fragility_epsilon
        self.familywise_alpha = familywise_alpha
        self.candidate_budget = candidate_budget

    def receipt_settings(self) -> dict:
        return {"selector_protocol": "fixed_source_pair_safety_v3",
                "mode": self.mode, "audit_pairs": self.audit_pairs,
                "competence_epsilon": self.competence_epsilon,
                "fragility_epsilon": self.fragility_epsilon,
                "familywise_alpha": self.familywise_alpha,
                "candidate_budget": self.candidate_budget,
                "uncertain_policy": "reject_and_retain_incoming"}

    def _validate(self, candidates: Sequence[CandidateMetrics], incoming: str,
                  identity: EvaluationIdentity | None,
                  audits: Mapping[str, AuditReceipt] | None) -> None:
        self.validate_archive(candidates, incoming)
        if identity is None or audits is None or set(audits) != {c.name for c in candidates}:
            raise ValueError("Safety selection requires a complete identity-bound audit archive")
        if len(candidates) - 1 > self.candidate_budget:
            raise ValueError("Candidate archive exceeds the prespecified multiplicity budget")
        pair_ids = audits[incoming].pair_ids
        if len(pair_ids) != self.audit_pairs:
            raise ValueError("Safety selection requires the fixed maximum audit size")
        for candidate in candidates:
            receipt = audits[candidate.name]
            if (receipt.identity != identity or receipt.harness_sha256 != candidate.harness_sha256 or
                    receipt.role != "C" or receipt.pair_ids != pair_ids):
                raise ValueError("Safety selection requires matched, ordered C receipts")

    def select_with_diagnostics(self, candidates: Sequence[CandidateMetrics], *, incoming: str,
                                identity: EvaluationIdentity | None = None,
                                audits: Mapping[str, AuditReceipt] | None = None) -> tuple[AcceptanceDecision, dict]:
        self._validate(candidates, incoming, identity, audits)
        audits = cast(Mapping[str, AuditReceipt], audits)
        baseline = audits[incoming]
        baseline_stats = evidence_statistics(baseline)
        # Two metrics, two tails and all allocated non-incoming candidates.
        tail = self.familywise_alpha / (2 * 2 * self.candidate_budget)
        verdicts = {}
        eligible = []
        eps_m = Fraction(str(self.competence_epsilon))
        eps_f = Fraction(str(self.fragility_epsilon))
        for candidate in candidates:
            receipt = audits[candidate.name]
            stats = evidence_statistics(receipt)
            if candidate.name == incoming:
                competence_interval = fragility_interval = (Fraction(0), Fraction(0))
                verdict = "PASS_INCOMING"
            else:
                competence_delta = tuple(Fraction(a + b - x - y, 2) for (a, b), (x, y) in
                                         zip(receipt.correctness, baseline.correctness))
                fragility_delta = tuple(Fraction(int(a != b) - int(x != y)) for (a, b), (x, y) in
                                        zip(receipt.correctness, baseline.correctness))
                competence_interval = _empirical_bootstrap_interval(competence_delta, tail)
                fragility_interval = _empirical_bootstrap_interval(fragility_delta, tail)
                safe = competence_interval[0] >= -eps_m and fragility_interval[1] <= eps_f
                unsafe = competence_interval[1] < -eps_m or fragility_interval[0] > eps_f
                verdict = "PASS" if safe else ("FAIL" if unsafe else "UNCERTAIN")
            if verdict.startswith("PASS"):
                eligible.append(candidate)
            verdicts[candidate.name] = {
                "verdict": verdict,
                "marginal_competence": float(stats["marginal_competence"]),
                "counterfactual_fragility": float(stats["counterfactual_fragility"]),
                "paired_accuracy": float(stats["paired_accuracy"]),
                "competence_delta": float(stats["marginal_competence"] - baseline_stats["marginal_competence"]),
                "fragility_delta": float(stats["counterfactual_fragility"] - baseline_stats["counterfactual_fragility"]),
                "competence_interval": [float(x) for x in competence_interval],
                "fragility_interval": [float(x) for x in fragility_interval],
            }
        winner = min(eligible, key=lambda c: (-c.accuracy, c.mean_turns))
        names = tuple(c.name for c in eligible)
        decision = AcceptanceDecision(winner.name, names,
                                      tuple(c.name for c in candidates if c.name not in names), self.mode)
        has_uncertain = any(row["verdict"] == "UNCERTAIN" for row in verdicts.values())
        has_candidate_pass = any(name != incoming and row["verdict"] == "PASS"
                                 for name, row in verdicts.items())
        status = ("ABSTAINED_TO_INCOMING" if winner.name == incoming and has_uncertain and not has_candidate_pass else
                  "RETAINED_INCOMING_WITH_UNCERTAIN_EXCLUSIONS" if winner.name == incoming and has_uncertain else
                  "SELECTED_PASS_WITH_UNCERTAIN_EXCLUSIONS" if has_uncertain else "RESOLVED")
        diagnostics = {"schema": 1, **self.receipt_settings(),
            "tail_probability": tail, "bootstrap": "exactly_enumerated_empirical_percentile",
            "decision_status": status,
            "verdicts": verdicts,
            "limitations": "Optimization-set bootstrap decision only; no population or post-RSFT guarantee."}
        return decision, diagnostics

    def select(self, candidates: Sequence[CandidateMetrics], *, incoming: str,
               identity: EvaluationIdentity | None = None,
               audits: Mapping[str, AuditReceipt] | None = None) -> AcceptanceDecision:
        return self.select_with_diagnostics(candidates, incoming=incoming, identity=identity, audits=audits)[0]
