"""Method E2-E3: constrain empirical selection before task-score ranking.

The empirical constraint has no population or post-RSFT non-regression guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from .evidence import AuditReceipt, EvaluationIdentity, finite_metric, validate_digest


@dataclass(frozen=True)
class CandidateMetrics:
    name: str
    harness_sha256: str
    accuracy: float
    mean_turns: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Candidate needs a name")
        validate_digest(self.harness_sha256)
        finite_metric(self.accuracy, probability=True)
        finite_metric(self.mean_turns)


@dataclass(frozen=True)
class AcceptanceDecision:
    selected: str
    eligible: tuple[str, ...]
    rejected: tuple[str, ...]
    mode: str


class TaskAccuracyAcceptance:
    """WHALE's accuracy/turn ordering; preserve archive order for exact ties."""

    @staticmethod
    def validate_archive(candidates: Sequence[CandidateMetrics], incoming: str) -> None:
        names = [c.name for c in candidates]
        if len(set(names)) != len(names) or incoming not in names:
            raise ValueError("Need a unique complete archive including the phase incoming harness")

    def select(self, candidates: Sequence[CandidateMetrics], *, incoming: str) -> AcceptanceDecision:
        self.validate_archive(candidates, incoming)
        winner = min(candidates, key=lambda c: (-c.accuracy, c.mean_turns))
        return AcceptanceDecision(winner.name, tuple(c.name for c in candidates), (), "off")


class EvidenceConstrainedAcceptance(TaskAccuracyAcceptance):
    """E2-E3 Acceptance Gate and prespecified same-data controls.

    `incoming` stays fixed within the weight phase. Candidates must come from the
    full valid archive: task-score Pareto pruning can discard the only eligible
    improvement. Missing or stale audits are errors, not low-scoring candidates.
    """

    RANK_MODES = {"paired_rank", "marginal_rank_floor"}
    MODES = {"off", "paired", "marginal_gate", "marginal_rank", "soft_pair"} | RANK_MODES

    def __init__(self, mode: str = "paired", *, epsilon: float = 0., pair_weight: float = 1.,
                 accuracy_tolerance: float = 0.):
        if mode not in self.MODES:
            raise ValueError(f"Unknown acceptance mode: {mode}")
        finite_metric(epsilon, probability=True)
        finite_metric(pair_weight)
        finite_metric(accuracy_tolerance, probability=True)
        if mode not in self.RANK_MODES and accuracy_tolerance != 0:
            raise ValueError("Accuracy tolerance is only defined for competence-floor ranking")
        if mode in self.RANK_MODES and epsilon != 0:
            raise ValueError("Evidence ranking uses accuracy_tolerance, not a paired epsilon")
        self.mode, self.epsilon, self.pair_weight = mode, epsilon, pair_weight
        self.accuracy_tolerance = accuracy_tolerance

    def select(self, candidates: Sequence[CandidateMetrics], *, incoming: str,
               identity: EvaluationIdentity | None = None,
               audits: Mapping[str, AuditReceipt] | None = None) -> AcceptanceDecision:
        if self.mode == "off":
            return super().select(candidates, incoming=incoming)
        self.validate_archive(candidates, incoming)
        if identity is None or audits is None:
            raise ValueError("Enabled selection requires identity-bound audit receipts")
        if set(audits) != {c.name for c in candidates}:
            raise ValueError("Audit coverage differs from the complete candidate archive")
        pair_ids = set(audits[incoming].pair_ids)
        for c in candidates:
            receipt = audits[c.name]
            if receipt.identity != identity or receipt.harness_sha256 != c.harness_sha256:
                raise ValueError(f"Stale or incompatible audit: {c.name}")
            if receipt.role != "C" or set(receipt.pair_ids) != pair_ids:
                raise ValueError("Acceptance requires the same optimization audit C for every candidate")
        def evidence(name: str) -> Fraction:
            receipt = audits[name]
            if self.mode in {"marginal_gate", "marginal_rank_floor"}:
                return Fraction(sum(a + b for a, b in receipt.correctness), 2 * len(receipt.correctness))
            return Fraction(sum(a * b for a, b in receipt.correctness), len(receipt.correctness))
        constrained = self.mode in {"paired", "marginal_gate"}
        eligible = [c for c in candidates if not constrained or c.name == incoming or
                    evidence(c.name) - evidence(incoming) >= -Fraction(str(self.epsilon))]
        if self.mode in self.RANK_MODES:
            # E2-v2 is an empirical competence floor, not a statistical guarantee.
            baseline = next(c for c in candidates if c.name == incoming)
            eligible = [c for c in candidates if c.name == incoming or
                        Fraction(str(c.accuracy)) - Fraction(str(baseline.accuracy)) >=
                        -Fraction(str(self.accuracy_tolerance))]
        def rank(c: CandidateMetrics) -> tuple:
            if self.mode in self.RANK_MODES:
                # E3-v2: matched control changes only paired versus marginal score.
                return -evidence(c.name), -c.accuracy, c.mean_turns
            if self.mode == "marginal_rank":
                score = audits[c.name].marginal_accuracy
            elif self.mode == "soft_pair":
                score = c.accuracy + self.pair_weight * audits[c.name].paired_accuracy
            else:
                score = c.accuracy
            return -score, c.mean_turns
        winner = min(eligible, key=rank)
        names = tuple(c.name for c in eligible)
        return AcceptanceDecision(winner.name, names,
                                  tuple(c.name for c in candidates if c.name not in names), self.mode)
