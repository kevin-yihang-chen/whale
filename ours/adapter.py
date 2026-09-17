"""Method E3 integration boundary for WHALE's function-based search.

This adapter consumes a complete evaluated archive; it does not implement a VLM
environment, checkpoint exporter, proposer or a live upstream search launcher.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Mapping, Sequence

from .acceptance import CandidateMetrics, EvidenceConstrainedAcceptance
from .evidence import AuditReceipt, EvaluationIdentity, fingerprint


class WHALEAcceptanceAdapter:
    """Single decision boundary for initial, ordinary, early-stop and resume callers.

    Off delegates to the caller's original selector verbatim, including its
    early-stop/tie semantics, with no audit calls. An enabled caller must use
    the returned decision to update BOTH accepted_harness and stop metadata.
    All four call sites still need wiring in the eventual visual-domain loop.
    """

    STAGES = {"initial", "ordinary", "early_stop", "resume"}

    def __init__(self, acceptance):
        self.acceptance = acceptance

    def decide(self, *, stage: str, original_selector: Callable[[], str], incoming: str,
               archive_provider: Callable[[], Sequence[CandidateMetrics]],
               audit_provider: Callable[[CandidateMetrics], AuditReceipt],
               identity: EvaluationIdentity,
               resume_receipt: Mapping | None = None) -> tuple[str, dict]:
        if stage not in self.STAGES:
            raise ValueError("Unknown acceptance stage")
        if self.acceptance.mode == "off":
            return original_selector(), {"mode": "off", "stage": stage, "audit_calls": 0}
        archive = tuple(archive_provider())
        settings = {"identity": asdict(identity), "incoming": incoming,
                    "archive": [asdict(c) for c in archive]}
        if hasattr(self.acceptance, "receipt_settings"):
            settings.update(self.acceptance.receipt_settings())
        else:
            settings.update(mode=self.acceptance.mode, epsilon=self.acceptance.epsilon,
                            pair_weight=self.acceptance.pair_weight)
            if self.acceptance.mode in self.acceptance.RANK_MODES:
                settings.update(selector_protocol="competence_floor_lexicographic_v2",
                                accuracy_tolerance=self.acceptance.accuracy_tolerance)
        binding = fingerprint(settings)
        if stage == "resume" and (resume_receipt is None or resume_receipt.get("binding") != binding):
            raise ValueError("Resume requires a matching decision receipt; bare accepted_harness.txt is insufficient")
        audits = {c.name: audit_provider(c) for c in archive}
        diagnostics = None
        if hasattr(self.acceptance, "select_with_diagnostics"):
            decision, diagnostics = self.acceptance.select_with_diagnostics(
                archive, incoming=incoming, identity=identity, audits=audits)
        else:
            decision = self.acceptance.select(archive, incoming=incoming, identity=identity, audits=audits)
        receipt = {"schema": 1, "binding": binding, "stage": stage, "identity": asdict(identity),
                   "decision": asdict(decision),
                   "audit_sha256": {name: fingerprint(asdict(audit)) for name, audit in audits.items()}}
        if diagnostics is not None:
            receipt["audit_diagnostics"] = diagnostics
        if stage == "resume" and (fingerprint(resume_receipt.get("decision")) != fingerprint(receipt["decision"]) or
                                  resume_receipt.get("audit_sha256") != receipt["audit_sha256"] or
                                  resume_receipt.get("audit_diagnostics") != receipt.get("audit_diagnostics")):
            raise ValueError("Resumed evidence or decision differs from the saved receipt")
        return decision.selected, receipt
