# VETO-v3: fixed-budget counterfactual safety audit

## Why V2 stops

The seed42 archive gave h1 one more correct C64 pair than h3 (54 versus 53),
so evidence-v2 selected h1 although ordinary H selected h3. On the independent
R512 diagnostic, h1 was lower than h3 by 2.15 percentage points in paired
accuracy and 1.86 points in marginal accuracy; the paired source-bootstrap
interval included zero. The preregistered continuation gate failed. This is a
negative V2 result and remains unchanged.

R512 is now method-development information. It cannot validate V3, select a V3
threshold or be reported as prospective V3 evidence. V1 and V2 remain available
as versioned historical protocols.

## E1-v3: competence and fragility

For source pair i, let c_i and c'_i be binary correctness. Define

M_C(h) = (1/(2N)) sum_i (c_i + c'_i),

F_C(h) = (1/N) sum_i 1[c_i != c'_i].

The earlier paired score is retained as a reported endpoint and satisfies the
exact identity P_C(h) = M_C(h) - F_C(h)/2. M and F are therefore two views of
the same paired outcomes, not independent evidence.

## E2-v3: uncertainty-aware safety decision

V3 uses one fixed C256 pass drawn from source tables that were not used by the
historical W/H/C/V/T/R split. All candidates and incoming h_in use the same
ordered source pairs. For each candidate, source-paired deltas in M and F are
resampled jointly with an exactly enumerated empirical percentile bootstrap.
The familywise alpha of 0.05 is divided across the three candidate slots, two
metrics and both tails. This is an operational optimization-set uncertainty
rule, not a distribution-free confidence sequence or a population guarantee.

A candidate is PASS when

LCB[M_C(h)-M_C(h_in)] >= -0.01

and

UCB[F_C(h)-F_C(h_in)] <= 0.

It is FAIL when the corresponding opposite bound establishes a violation. All
other candidates are UNCERTAIN. At the fixed audit limit, FAIL and UNCERTAIN do
not enter the task ranking. Incoming is always PASS. This explicit abstention
prevents a one-pair C64 difference from controlling the next training phase.

## E3-v3 and E4

Among PASS candidates and incoming, use WHALE's original ordering: maximize H
accuracy, then minimize H turns, preserving archive order for exact ties. Paired
evidence is never a ranking objective. E4 remains the original successful-
trajectory RSFT with the same restoration convention. Pair-RSFT is excluded
from the main method.

In the paper architecture, E2-v3 is the **Counterfactual Safety Audit** between
candidate evaluation and WHALE selection. `CounterfactualSafetyAcceptance`
implements E1-v3 through E3-v3; native RSFT remains the unchanged E4 block.

## Proposal data and mechanism slots

The proposer receives ordinary H128 outcomes plus answer-changing twins of the
same 128 source-disjoint H charts. Eight paired questions are reassigned by a
model-free minimum-change rule because their original read-value targets did not
have disjoint answer intervals after editing; exact three-task quotas are retained.
Feedback reports only observable both-correct,
base-only, counterfactual-only and both-wrong behavior by task, with bounded
examples. C and all endpoint data remain hidden.

The three candidate slots are fixed to visual recheck, reasoning decomposition
and response control. Each candidate declares its hypothesis, addressed failure
and expected policy/tool cost in machine-readable metadata. Crop/zoom is not
enabled in V3: the existing development screen measured direct at 53/64 pairs
and normalized zoom at 0/64 under the then-current loop. A new tool condition
would require a separate shared calibration before entering the search space.

## Fresh-evidence rule

V3 code and tolerances are frozen before new model calls. Its C256 sources come
from previously unselected source tables. The seen R[0:512] and the unobserved
R remainder are not used to claim V3 success. T remains sealed. A future result
must identify the exact V3 data manifest, candidates, weights, code and decision
receipt, and must retain negative or abstaining outcomes.

Sequential 64-to-128-to-256 peeking is deferred. It may be added only with a
separately frozen repeated-look error allocation or confidence sequence; ordinary
bootstrap intervals must not be repeatedly inspected as if they retained their
single-look coverage.
