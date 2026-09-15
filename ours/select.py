"""Apply Method E2-E3 to recorded evaluations; never invent unrun scores.

Input schema: identity, incoming, candidates (CandidateMetrics dictionaries),
audits (harness name to AuditReceipt dictionary). CLI off uses task-score ranking;
live upstream off-path compatibility is provided by WHALEAcceptanceAdapter.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .acceptance import CandidateMetrics, EvidenceConstrainedAcceptance
from .evidence import AuditReceipt, EvaluationIdentity, fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(EvidenceConstrainedAcceptance.MODES), default="paired")
    parser.add_argument("--epsilon", type=float, default=0.)
    parser.add_argument("--pair-weight", type=float, default=1.)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    identity = EvaluationIdentity(**data["identity"]) if args.mode != "off" else None
    audits = None
    if args.mode != "off":
        audits = {name: AuditReceipt(**{**r, "identity": EvaluationIdentity(**r["identity"])})
                  for name, r in data["audits"].items()}
    result = EvidenceConstrainedAcceptance(args.mode, epsilon=args.epsilon, pair_weight=args.pair_weight).select(
        [CandidateMetrics(**c) for c in data["candidates"]], incoming=data["incoming"],
        identity=identity, audits=audits)
    with args.output.open("x") as stream:
        json.dump({"input_sha256": fingerprint(data), "decision": asdict(result),
                   "epsilon": args.epsilon, "pair_weight": args.pair_weight,
                   "kind": "recorded_evaluation_selection"}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
