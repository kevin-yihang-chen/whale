# VETO v2: competence-constrained evidence ranking

2026-09-16. Authorized after review of method suggestions. This amendment adopts
the two recommended changes, not the proposed 15-condition matrix, cleanup,
phase-cap transfer, confidence-bound selector, triplets, or Pair-RSFT.

## Method and fair comparison

E1 is unchanged: P_C(h) = mean_i c_i(h)c'_i(h), using the fixed host scorer.
E2-v2: E = {h: A_H(h) >= A_H(h_in)} union {h_in}. The empirical ordinary
accuracy tolerance is fixed at zero. It is not statistical non-inferiority.
E3-v2: maximize (P_C(h), A_H(h), -turns_H(h)) lexicographically over E;
exact ties preserve complete archive order. Incoming stays fixed at the start
of the phase. Missing/stale audits are errors. E4 remains native RSFT, including
successful-trajectory filtering and the common Adam-reset restoration convention.

`EvidenceConstrainedAcceptance(mode='paired_rank')` implements E2-v2/E3-v2.
`marginal_rank_floor` changes only the primary score to mean_i(c_i+c'_i)/2;
competence floor, tie-breaking, candidate archive and measurement budget match.
`off` preserves upstream selection exactly. Legacy `paired`, `marginal_gate`,
and `marginal_rank` keep their old semantics. Adapter receipts bind the new mode
and accuracy tolerance, including initial, ordinary, early-stop and resume paths.

New `fast_chart_search init` defaults to `--selection-protocol evidence_v2`.
Explicit `gate_v1` is available; plans without a version remain v1. Formal v2
search compares WHALE/off, VETO/paired_rank and marginal_rank_floor. All share
one candidate pool and H/C measurements. Source identity checks remain active.
The old 18-condition schedule is an archived v1 plan, not an executed v2 matrix.
Any later formal schedule must identify its version explicitly.

## Shared feedback path, not a second claimed contribution

`chart_proposal_feedback.summarize` validates all 128 H answers with the shared
parser and scorer, reports correct/parseable-wrong/unparseable by task, and includes
the first two failures per task in sorted sample-id order. Summary rules are fixed
before new proposals. The actual proposer task contains this JSON, with a fingerprint
and an input receipt. Every method shares the same feedback-conditioned proposals.
This establishes delivery, not that the proposer internally used the feedback.
No C/V/T/R outcomes enter the proposer; error causes such as wrong bar reading
are not inferred from final answers. Feedback improvements are infrastructure.

## Immediate independent selection diagnostic

The observed seed42 v1 candidate archive is exploratory. Regrade its complete
H/C records before replay: old WHALE and gate select h3; new paired ranking selects
h1 and the matched marginal ranking selects h3. This motivates testing and is not
a retrospective v2 win. No new candidates or API calls are needed for this probe.

Compare h1 and h3 at exactly the same existing theta1 checkpoint on R512, the first
512 of the reserved 1,024 unique R source charts sorted by source id. Both receive
the same questions, processor, frozen endpoint inference policy, seed42 and full
pass. No V-based filtering of this R subset. T stays sealed; R is not used to tune
thresholds, prompts, learning rates or generate further candidate replacements.
This one-seed diagnostic uses old generic proposals and is not the combined v2
method result. Its outcome is not copied into unexecuted training branches.

Report paired and marginal differences h1 minus h3. Resample source charts jointly
20,000 times, RNG seed20260916, for a percentile95% interval. This interval describes
source variation conditional on the seed and candidate archive, not seed variation.
Continue toward bounded training only if the paired difference is positive, the
interval lower bound is positive and ordinary accuracy loss is at most1 percentage
point. Otherwise stop expansion and record insufficient selection evidence. No
automatic retries, extra candidates or checkpoint training are triggered by the suite.

One single-H800 job executes the two cases sequentially with at most2 GPUh total
(one-hour child deadlines), charged to the existing core40 GPUh phase. Temporary
storage admission reserves12 GiB within the existing400 GiB scope; no new model
checkpoints, deleted files or changed API45yuan/total100GPUh limits. Standard Slurm
ALL state notifications and the existing terminal-cost observer remain enabled.

Formal v2 validation still requires frozen three-seed runs and actual continuation.
Identical first-stage training can be reused with full identity and cost attribution.
The old seed42 results remain unchanged and cannot be called a prospective v2 run.

## Reproduction

CPU archive replay:

```
python -m ours.chart_selection_diagnostic replay --selection <old-selection-veto.json> --output <new-report.json>
```

Prepare/check with the existing offline training runtime, then submit only the
bounded suite through the common budget entrypoint:

```
python -m ours.chart_selection_diagnostic prepare --selection <old-selection-veto.json> --plan <new-plan.json> --output <fresh-compact-directory>
python -m ours.chart_selection_diagnostic check --plan <new-plan.json>
python -m ours.fast_chart_submit --kind selection-probe --phase core --plan <new-plan.json>
```
