# Approved compact VETO protocol, 2026-09-15

Authority: the user's explicit implementation request for the compact plan.
This prospective revision supersedes the 51-condition schedule for new work;
the original protocol, source snapshots, costs and outcomes remain archived.
No old scientific gate is retroactively passed.

## Question and contribution

VETO — Visual Evidence Tests for Optimization. Working title:
*VETO: Evidence-Constrained Harness Selection for Reliable Chart Reasoning*.
Test whether paired evidence improves harness selection and subsequent RSFT
under a limited budget. Long-horizon shortcut propagation is not a required
claim. AutoDesign's acceptance gates and Chartographer's counterfactual chart
evaluation are prior art, not our inventions.

E1 is mean paired correctness, mean(c_i*c'_i). E2 fixes incoming and current
weights and permits candidates with P >= P_incoming (epsilon=0), always keeping
incoming. E3 ranks the complete eligible archive by accuracy, then fewer native
turns, retaining archive order for exact ties. E4 is native successful-trajectory
RSFT, with the corrected multimodal forward shared by every condition. Native
checkpoint restoration excludes Adam moments as in the existing protocol.

## Frozen experimental choices

- Qwen3.5-4B only. Seeds 42,43,44. Chess/CLEVR/2B/51-run expansion paused.
- Source groups and W2048/H128/C64/V256/T1024/R1024 partitions unchanged.
- Equal-as-possible comparison/read-value/difference tasks per partition,
  assigned deterministically using source hashes and pixel-oracle eligibility.
  Numeric tolerance is 5% relative, absolute 1e-9 at zero. Answer-changing
  numeric intervals must be disjoint. No model results enter assignment.
- Shared host-owned answer parser, ground-truth verifier, tool interface and
  generation bounds. Candidate parse_answer callbacks cannot affect scores.
- Initial prompts: direct, structured chart reading, brief reasoning. Select
  on ordinary full-V accuracy, ties to fewer generated tokens then fixed order.
- Shared LR in {1e-7,1e-6,1e-5}, select on ordinary V accuracy, ties to smaller LR.
- Each training stage: four batches, eight tasks/batch, eight trajectories/task.
  One search round with at most three candidate attempts; failures consume a
  slot, never trigger an automatic extra proposal. epsilon is not tuned.
- Five trained branches share each seed's first-stage checkpoint and candidate
  measurements: weight-only, WHALE, VETO, marginal gate, counterfactual
  augmentation. Harness-only searches independently at theta0. Eighteen logical
  conditions are not eighteen independent first-stage trainings.
- Augmentation replaces half of the second-stage ordinary question slots with
  complete C pairs sampled deterministically without replacement per seed;
  ordinary+paired slots remain32, trajectories256. It uses WHALE's choice.
- Final checkpoint is budget-defined. T is never a selection dataset.
- ChartQA: fixed original open-answer subset,256 human+256 augmented questions,
  source-group selection and overlap review before inference. Label as subset.
- Physical shared work is charged once in the allocation ledger. Comparative
  per-method costs include their share of common training/search and all audit
  work required by the algorithm; sharing never creates a free method.

## Prospective decisions and resource contract

First decision on day7: real selector differences from WHALE and marginal gate,
positive development effects in at least2/3 seeds, and independent confirmation.
No differences or unsupported recheck -> stop expanding method experiments and
produce a diagnosis. No silent addition of seeds, candidates, losses or metrics.
Final criterion: mean paired improvement >=1pp, all seed differences >=0, at
least two >0, mean ordinary difference >=-1pp, source-cluster95% interval with
lower bound >0; demonstrate value beyond marginal gate. Inconclusive intervals,
external regressions and stronger augmentation results remain in the report.

GPU total100h includes the existing pilot ledger's6.8980555556h. Phase caps:
setup20/core40/ablation15/evaluation20/retry5. Reserve full allocation time before
submission and settle against actual Slurm terminal records including failures.
No hidden600h fallback. Project GLM cumulative45CNY ceiling is unchanged.
New disk allocation <=400GiB; preserve40GiB free margin. Main output root is
data/fast-chart-20260915-v1. Reuse immutable images; no deletion authorization
is implied. Before any cleanup provide exact paths, sizes and dependencies and
obtain explicit authorization. B group remains untouched. No GitHub push.
Compute nodes use local assets; candidate callbacks retain OS file/network
confinement. Status-change email remains configured for yihangc@connect.hku.hk.

Dates: draft2026-09-29; candidate2026-10-13. Budget/queue/data failures are recorded
as actual blockers rather than invented completed results. Paper scaffolding
starts immediately; no fabricated numbers or work-volume claims.

## Evidence and deliverables

Three tables: main comparison, necessary ablations, external evaluation/cost.
Three figures: mechanism, selection/training trajectories, performance/cost.
Positive and negative cases with original model outputs. Every headline claim
links to immutable configuration, code/model/data identities, source-cluster
predictions, real allocation receipt and cost entries. Every code delivery maps
changes to E1/E2-E3/E4 or shared infrastructure.

References: https://arxiv.org/html/2608.13560v1 ;
https://arxiv.org/abs/2605.27311 ; https://github.com/vis-nlp/ChartQA ;
https://cvpr.thecvf.com/Conferences/2027/Dates .
