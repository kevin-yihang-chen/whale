"""Method E1: correctness under paired, answer-changing visual interventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Callable, Sequence


def fingerprint(value: object) -> str:
    """Canonical content identity; reject non-finite JSON numbers."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_digest(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA-256 digest")


@dataclass(frozen=True)
class EvaluationIdentity:
    """Bind scores to a fixed weight phase and its actual evaluation inputs.

    Callers must hash checkpoint content, not a model name or checkpoint path.
    The identity is a compatibility check, not proof of the serving model's identity.
    """

    weights_sha256: str
    search_data_sha256: str
    audit_data_sha256: str
    decode_sha256: str
    verifier_sha256: str
    phase: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name != "phase":
                validate_digest(value)
        if not isinstance(self.phase, str) or not self.phase:
            raise ValueError("A nonempty weight phase is required")


@dataclass(frozen=True)
class VisualPair:
    """Verifier-only metadata. Do not pass this object to the model or proposer."""

    pair_id: str
    source_id: str
    question: str
    images_sha256: tuple[str, str]
    answers: tuple[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "images_sha256", tuple(self.images_sha256))
        object.__setattr__(self, "answers", tuple(self.answers))
        if any(not isinstance(v, str) or not v.strip() for v in
               (self.pair_id, self.source_id, self.question)):
            raise ValueError("Pair, source and question must be nonempty strings")
        if len(self.images_sha256) != 2 or len(self.answers) != 2:
            raise ValueError("A visual intervention has exactly two sides")
        for value in self.images_sha256:
            validate_digest(value)
        if self.images_sha256[0] == self.images_sha256[1]:
            raise ValueError("Answer-changing pairs require different image content")
        if any(not isinstance(a, str) or not a.strip() for a in self.answers):
            raise ValueError("Answers must be nonempty strings")
        if self.answers[0].strip().casefold() == self.answers[1].strip().casefold():
            raise ValueError("E1 requires different correct answers")


@dataclass(frozen=True)
class PairPrediction:
    pair_id: str
    answers: tuple[str, str]


@dataclass(frozen=True)
class AuditReceipt:
    identity: EvaluationIdentity
    harness_sha256: str
    role: str
    pair_ids: tuple[str, ...]
    correctness: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        validate_digest(self.harness_sha256)
        if not isinstance(self.identity, EvaluationIdentity):
            raise ValueError("Expected an EvaluationIdentity")
        object.__setattr__(self, "pair_ids", tuple(self.pair_ids))
        object.__setattr__(self, "correctness", tuple(tuple(row) for row in self.correctness))
        if self.role not in {"C", "V", "T", "engineering"}:
            raise ValueError("Unknown audit data role")
        if any(not isinstance(value, str) or not value for value in self.pair_ids):
            raise ValueError("Pair IDs must be nonempty strings")
        if not self.pair_ids or len(set(self.pair_ids)) != len(self.pair_ids):
            raise ValueError("Audit requires nonempty, unique pair IDs")
        if len(self.pair_ids) != len(self.correctness):
            raise ValueError("Incomplete paired audit")
        if any(len(row) != 2 or any(type(v) is not int or v not in (0, 1) for v in row)
               for row in self.correctness):
            raise ValueError("Each pair needs two binary verifier outcomes")

    @property
    def paired_accuracy(self) -> float:
        return sum(a * b for a, b in self.correctness) / len(self.correctness)

    @property
    def marginal_accuracy(self) -> float:
        return sum(a + b for a, b in self.correctness) / (2 * len(self.correctness))


class VisualPairEvaluator:
    """E1 Paired Visual Audit; exact pair coverage and a frozen binary verifier."""

    def __init__(self, verifier: Callable[[str, str], bool]):
        self.verifier = verifier

    def evaluate(self, pairs: Sequence[VisualPair], predictions: Sequence[PairPrediction],
                 *, identity: EvaluationIdentity, harness_sha256: str, role: str) -> AuditReceipt:
        if not pairs or len({p.pair_id for p in pairs}) != len(pairs):
            raise ValueError("Expected nonempty, unique manifest pairs")
        if identity.audit_data_sha256 != fingerprint([asdict(pair) for pair in pairs]):
            raise ValueError("Actual pair manifest differs from audit identity")
        lookup = {p.pair_id: p for p in predictions}
        if len(lookup) != len(predictions) or set(lookup) != {p.pair_id for p in pairs}:
            raise ValueError("Predictions must cover every manifest pair exactly once")
        rows = []
        for pair in pairs:
            prediction = lookup[pair.pair_id]
            if len(prediction.answers) != 2 or any(not isinstance(a, str) for a in prediction.answers):
                raise ValueError("Expected two text predictions per pair")
            row = tuple(self.verifier(pred, truth) for pred, truth in zip(prediction.answers, pair.answers))
            if any(type(value) is not bool for value in row):
                raise ValueError("E1 requires a binary verifier, not a fractional score")
            rows.append(tuple(int(value) for value in row))
        return AuditReceipt(identity, harness_sha256, role,
                            tuple(p.pair_id for p in pairs), tuple(rows))


def finite_metric(value: float, *, probability: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Metrics must be finite real numbers")
    if value < 0 or (probability and value > 1):
        raise ValueError("Metric outside its valid range")
