# Approved VETO execution protocol, 2026-09-10

Status: APPROVED_FOR_STAGED_EXECUTION; scientific claims remain unvalidated.
Authority: the user's explicit implementation request for the 4–6 week plan in
this session. This prospective amendment supersedes the old requirement to
complete all twelve Chess pilot trials before visual research. The original
Chess protocol and frozen execution records remain intact.

## Objective and schedule

VETO — Visual Evidence Tests for Optimization. Working title: *VETO: Preserving
Visual Evidence during Harness–Weight Co-adaptation*. Investigate whether
accuracy-selected harnesses transfer visual shortcuts into weights through
subsequent RSFT, then test paired visual acceptance at the search boundary.
No assumed improvement percentage or acceptance guarantee.

Complete draft: October 8; submission candidate: October 22. Target CVPR 2027:
registration November 10, paper November 16, supplement November 23, all AoE.
Source: https://cvpr.thecvf.com/Conferences/2027/Dates (checked 2026-09-10).

Finish the active Chess seed42 full search and actual native second weight
phase, including checkpoint handoff verification. Defer remaining Chess seeds
and Chess FST; do not open the Chess test64 split. This closes an engineering
prerequisite, not a numerical reproduction of the published paper.

## Method and implementation boundary

E1: P_C(theta,h) = mean_i 1[answer_i=y_i] 1[answer'_i=y'_i]. Both sides must be
correct under the same question and a valid image change. E2: admit h when
P_C(theta,h) >= P_C(theta,h_in)-epsilon, always retaining incoming h_in. Keep
theta and h_in fixed within a search phase; default epsilon=0. E3: use native
accuracy-first, native-turns-second ordering over the eligible complete valid
archive. E4 reuses native successful-trajectory RSFT, with no new loss.

`VisualPairEvaluator` is E1; `EvidenceConstrainedAcceptance` is E2–E3;
`WHALEAcceptanceAdapter` connects initial/ordinary/early-stop/resume decisions.
Image input, crop tools, reward/verifier, serving, checkpoint transport and
budget records are shared infrastructure. Additions live in ours/, with the
upstream submodule unchanged. Every implementation report identifies its
equation or diagram block. No generalization or post-training non-regression
theorem follows from the empirical gate.

AutoDesign already uses non-regression acceptance and CHILL-Harness already
studies counterfactual harness learning. Contribution must be distinguished by
the visual-to-weight feedback diagnosis and intervention evidence:
https://arxiv.org/html/2608.13560v1
https://arxiv.org/html/2607.25825v1

## Data and experimental controls

Primary domain: derived PlotQA numerical chart pairs, with source tables,
rendering provenance and deterministic truth. Secondary: existing CLEVR images
paired under identical executable questions with opposite truth. Scene pairs
are not called minimal edits unless the individual change is controlled.
External test: original open-answer ChartQA and its original scoring protocol;
binary derivatives are separately named. Links and licenses are inventoried
before acquisition/distribution. Source truth never enters model/tool inputs.

Per domain: W=2048 questions, H=128 questions, C=64 pairs, V=256 pairs,
T=1024 sealed pairs, independent recheck pool=512 pairs expandable to1024.
Group all variants of a source chart/scene in one split; check source/template
near-duplicates. C is optimization data. T is not used to select checkpoints,
hyperparameters, candidate code or narratives. No-image and answer-preserving
style controls remain separately labelled.

Pilot seeds42/43/44: weight-only, harness-only, WHALE in the chart domain.
Select a shared strong h0 using ordinary V accuracy. Choose shared visual LR
from {1e-7,1e-6,1e-5} by V accuracy, ties to smaller LR. Freeze all choices before
scientific sampling. Calibrate {4,8} batches/weight phase and {2,3} rounds/search;
choose the largest common configuration fitting the phase budget, with at least
three weight phases and two searches. Each search round has three proposals.
If the smallest configuration cannot fit, stop expansion and report it.

Formal seeds42/43/44, Qwen3.5-4B main and Qwen3.5-2B scale check:

| Matrix | Runs |
|---|---:|
| two domains x4B x(weight-only,harness-only,FST,WHALE,VETO) x3 seeds | 30 |
| charts x2B x(WHALE,VETO) x3 seeds | 6 |
| charts x4B x(marginal gate,augmentation,ordinary reevaluation,soft pair,random rejection) x3 seeds | 15 |

51 formal condition runs; engineering, calibration and diagnostic forks are
additional and still count against the resource cap. Failure remains failure,
not a zero score or fabricated completed row. FST means Fast–Slow Training.
Soft-score lambda is selected from a prospectively recorded small V grid.
Archive-only epsilon/audit-size sensitivity measures selection stability, not
the unexecuted training performance of alternative decisions.

At the first post-search checkpoint, branch incoming versus selected harness
with identical input/order/resource and native restore behavior. Evaluate both
under common h0; additionally match accepted-token counts for diagnosis. Keep
the native Adam-reset caveat explicit and identical across branches.
Report equal-candidate/trajectory comparisons and equal-total-resource
comparisons; give ordinary WHALE access to saved audit budget for reevaluation
or extra sampling. Include rejected/failed candidates, API, audit, loading,
training and final evaluation costs. Report all seeds, sample std, paired
source-cluster uncertainty, common-h0 and learned-harness performance.

## Go/no-go and resource bounds

First week incl. new Chess completion allocations: <=100 GPUh. Before expansion:
valid visual evidence and oracle/no-leakage checks; at least2/3 seeds show A not
decreasing and P decreasing >=2pp at a pre-registered phase; fresh recheck keeps
direction; evidence of extra degradation relative to fixed-harness training;
throughput and storage support the matrix. An inconclusive interval is not a
positive result. Use the recheck pool within100 GPUh; unsupported after that
stops method expansion and produces a diagnosis/replanning report.

Final method gate: mean paired improvement >=2pp across the three domain/size
cells, consistent aggregate direction across all three seeds, mean ordinary
accuracy degradation <=1pp, and independent value beyond simple controls.
Report each cell, external failures and uncertainty; reduce claims as needed.

| Stage | GPUh cap |
|---|---:|
| pilot (Sept10–17) | 100 |
| main36 (Sept18–Oct1) | 280 |
| ablations and mechanism (Sept25–Oct8) | 160 |
| verification and necessary reruns (Oct9–22) | 60 |

Total600 GPUh is a ceiling. New allocations reserve full GPU count x Slurm time
limit before submission; failures are charged, ambiguous submissions retain
reservations. Previous costs remain in original ledgers. Expansion is disabled
until a verified scientific decision exists. Continue the global45CNY GLM
journal, including unresolved calls; no automatic budget increase or provider
substitution. Networked entry node prepares assets; compute runs offline.

Compare queue/runtime/GPUh for2/4 GPUs under one-job/four-GPU constraints;
evaluate two independent two-GPU lanes where this saves total completion time.
Do not change frozen Chess allocations. Keep execution-state notifications.

No deletion is authorized. First provide exact paths, sizes, reference and
retention analysis, then obtain the user's explicit approval. A storage deficit
pauses expansion. Preserve active/frozen checkpoints and evidence. Changes stay
local; GitHub push requires separate explicit authorization.

## Deliverables and acceptance checks

Real service startup, image-bearing generation, tool pixel lineage, actual
parameter update, checkpoint reload into the next search, and all four selector
paths must pass. CPU fixtures are explicitly distinguished from GPU evidence.
Candidate processes need OS file/network confinement; AST checks are only an
additional layer. Test pair identity/order, masks/rewards, partial failures,
cache invalidation, restore lineage and accounting without fabricating scores.

Three tables: main results, ablations, external generalization/cost. Four figures:
method, A–P phase trajectories, checkpoint/evaluation-harness cross matrix,
performance/cost. Include successful and failed tool/evidence case studies.
Preprocessing/plotting scripts exist to reproduce those artifacts. Final package:
CVPR manuscript/supplement, all51 outcomes or explicit failures, reproducible
launchers, source/data/cost manifests, reviewer-oriented evidence checklist.
