# VETO manuscript evidence checklist

This checklist follows the [September15 compact protocol](../fast_chart_protocol_20260915.md).
It tracks what the manuscript must demonstrate; a checked implementation or a
compiled PDF does not establish a scientific result. Update result-dependent
entries from complete records, including negative results. Operational state is
maintained in [PROJECT_STATUS](../../PROJECT_STATUS.md).

## Contribution and interpretation

| Review question | Required evidence and manuscript treatment | Current status |
|---|---|---|
| What belongs to WHALE? | Attribute alternating optimization, native RSFT, the harness search, and the training infrastructure to the upstream paper. Identify E2–E3 as the studied selection rule. | Described in main.tex and README; upstream tracked files remain unchanged. |
| How does VETO differ from existing acceptance gates and counterfactual evaluation? | Discuss AutoDesign and Chartographer; claim value in evidence-constrained harness selection followed by real training, without claiming the first generic gate or paired metric. | Related-work framing present; empirical distinction pending. |
| Does the safety audit add value beyond ordinary and same-data point selection? | Report M, F, P, all three verdicts, eligible sets and selected harnesses. Use P=M−F/2 as a decomposition, not independent evidence. | V2 failed its independent R512 gate. In the complete V3 C256 archive, h1/h2 are FAIL and h3 is UNCERTAIN; VETO, WHALE and the matched point gate all select h0. No distinct benefit is observed. |
| Is the claimed effect from selection or only from evaluation noise? | Compare WHALE, VETO and the point gate using the same candidate measurements and budgets. Preserve uncertainty and independent rechecks. | h3 gains2 paired C examples and3 single-image answers over h0, but both adjusted difference intervals cross zero and h3 is one H answer lower. All rules retain h0; no selection effect is claimed. |
| Is the scope overstated? | Describe one training–search–training cycle with one4B model and one chart domain. Do not claim long-term forgetting, all visual tasks, scale laws or full-budget WHALE reproduction. | Scope narrowed in protocol and manuscript. |
| Is an empirical constraint being presented as a theorem? | State that E2 constrains the current weights and audit set only. No guarantee of future-training or population non-degradation. | Explicit in main.tex and supplement.tex. |

## Experimental controls and validity

| Review question | Required evidence and manuscript treatment | Current status |
|---|---|---|
| Are h0 and LR common, with no cherry-picking? | Select h0 from three registered prompts by ordinary V accuracy, ties by tokens. Select LR from1e-7/1e-6/1e-5 by ordinary V accuracy, ties smaller. Report every calibration row and explicitly reuse the selected seed42 prefix. | All three prompts and rates complete; fixed rule selects structured h0 and LR1e-6. All calibration declines are retained. |
| Do image pairs have valid changed answers? | Retain source tables, rendering recipes, numerical checks, disjoint answer-tolerance intervals and unchanged source-group roles. Distinguish derived PlotQA-EvidencePairs from official PlotQA scores. | V3 H-pair128 and fresh C256 passed pixel and answer checks. Fresh C overlaps neither historical roles nor H-pair. Eight H-pair tasks were reassigned by a model-free minimum-change rule. |
| Can a candidate change grading or see audit/test labels? | Host-managed parser/scorer; H/H-pair-only proposer feedback; restricted candidate processes. Preserve source identities and actual request/response evidence. | The V3 proposer ran once using H/H-pair feedback; the three candidates were confined and measured with the host scorer. C/V/T/R labels remained excluded from proposal input. |
| Does real visual training happen? | Match accepted trajectories to actual image forwards and finite nonzero visual gradients; verify the full native parameter update and exact BF16 export. | Three full4-batch calibration stages audited; formal continuations pending. |
| Are the next search and continuation using the updated model? | Record model/processor/decode/harness identities, loaded coordinates, checkpoint restoration, actual sampler state and theta1→theta2 parameter differences. | V3 fixed-weight search used the verified theta1 identity and completed h0--h3. All rules selected h0, so the preregistered distinct-selection gate stopped continuation before theta2. |
| Is the off condition the original selection behavior? | Exercise initial, normal, early-stop and resumed paths with mode=off. No bypass of the unified acceptance adapter; retain failed candidate slots. | CPU native-loop paths tested. Complete real seed42 final/resume choices independently reconstructed and identical; this real run did not encounter an early stop or candidate failure. |
| Is augmentation a fair simple alternative? | Start at the same theta1 and WHALE-selected harness; replace50% of32 second-stage question slots with complete pairs, preserving256 trajectory attempts and restoration order. | Sampling control implemented and CPU checked; actual augmentation training pending. |
| Are six conditions really independent runs? | Report18 logical condition outcomes, three independent seeds, shared physical prefixes and archives. Train all required branches; do not count shared work as independent evidence. | Protocol specifies sharing; formal condition outcomes pending. |
| Is test data kept out of selection? | Freeze final checkpoints by the registered budget, before T/R/ChartQA endpoints. Do not tune prompts, LR, epsilon, batching or checkpoints on T. | No compact T/R/ChartQA model calls yet; shared batch32 endpoint policy frozen from complete speed measurements. |
| Is ChartQA being reported correctly? | Use the preselected256 human and256 augmented original open-answer questions, official scoring, separate subset results and exact-overlap checks. Label it a fixed subset, not the full benchmark; acknowledge unverified pretraining/semantic overlap. | Dataset and shared scorer prepared; real external scores pending. |

## Results, resources and submission readiness

| Review question | Required evidence and manuscript treatment | Current status |
|---|---|---|
| Is the primary endpoint supported across seeds? | Report paired and ordinary accuracy for42/43/44, mean and sample SD, and source-chart clustered95% intervals. Explain that the interval is conditional on the three trained seed pairs. | Statistics implementation checked; final model evidence absent. |
| Are ordinary accuracy, tokens and negative outcomes retained? | Include ordinary losses, generation tokens, failed candidates/runs and external degradation. No best-seed-only reporting or predetermined contribution percentage. | Negative calibration measurements retained; formal results pending. |
| Do final weights improve under a common harness? | Evaluate final WHALE/VETO weights under the same h0 as well as their selected harnesses. Interpret a single-cycle observation within its limited scope. | Endpoint interface prepared; real observations pending. |
| Are cost comparisons complete and fair? | Include loading, failed jobs/candidates, audits, training, evaluation and API usage. Charge shared computation once to the project and explicitly attribute it to beneficiaries for method comparisons. | V3 successful candidate archive used2.300556GPUh; the separate0.105GPUh h1 interface failure is retained. Project ledger is19.557222/100GPUh with no reservation; no final method-cost comparison exists because expansion stopped. |
| Does the full plan fit the actual limits? | Forecast all18 conditions using completed LR allocations and measured common endpoint throughput;100GPUh total, phase caps,400GiB new disk and45CNY API. Obtain exact-path approval before any cleanup. | Expansion stopped at19.557222GPUh after the distinct-selection gate failed. No phase transfer or cleanup is needed for V3 closure; any V4 attempt requires a new bounded protocol. |
| Has the Day7 continuation gate passed? | Distinct actual selections plus positive development evidence in at least two seeds under fixed scoring and equal budgets; independent recheck. Stop expanding if rules always coincide or benefit lacks support. | No. The complete seed42 V3 archive gives h0 for WHALE, VETO and the matched point gate. Expansion and downstream V3 continuation stopped as preregistered; seeds43/44 are not used to rescue the failed gate. |
| Does the candidate manuscript meet the prospective criteria? | Paired mean improvement≥1pp, all three seeds nonnegative and at least two positive, ordinary mean loss≤1pp, main grouped95% interval strictly positive, and value beyond the marginal gate. Retain inconclusive/contrary evidence. | Not evaluated. These criteria cannot be claimed passed from calibration. |
| Can each headline claim be traced? | Three complete result tables, three evidence figures, success/failure cases, raw-result identities, reproducible commands, main/supplement sources and claim-to-evidence index. | The complete V3 optimization archive and stop decision are hash-indexed. Final method tables/curves remain incomplete because no V3 continuation was scientifically admissible. h2's comparison failure is a task-type mismatch observation, not demonstrated visual non-use. |
| Is the submission package current and authorized? | Recheck official2027 formatting before submission. Exclude secrets, raw model/data artifacts and runtime caches from publication; obtain separate approval for GitHub push or submission. | Local `paper-build-v28` contains the V3 negative diagnostic and hash-bound evidence index. The public GitHub snapshot remains older; no push or submission is authorized by this update. |

## Evidence entrypoints

- [Complete V3 candidate archive and equivalent selections](../../data/fast-chart-20260915-v1/v3-seed42-search-report-v1/result.json)
- [V3 stop decision, costs and evidence hashes](../../results/veto-v3-seed42-complete-diagnostic-20260917-v1.json)

- [Complete first native cycle and raw-answer reconstruction](../../data/fast-chart-20260915-v1/first-cycle-report-v2/result.json)
- [First-cycle diagnostic and limits](../first_cycle_diagnostic_20260916.md)
- [Explicit state of all18 conditions](../../results/fast-chart-matrix-status-20260916-v1.json)

- [First complete selection reconstruction](../../data/fast-chart-20260915-v1/first-search-report-v1/result.json)
- [Post-hoc case with unchanged images and real responses](../../data/fast-chart-20260915-v1/illustrative-case-v1/case.json)
- [Six-page draft visual review](../../results/fast-chart-paper-v15-visual-review-20260915-v1.json)
- [Actual proposer tool-use limitation](../../results/fast-chart-first-proposal-tool-use-review-20260915-v1.json)

- [Complete three-rate raw-answer review](../../results/fast-chart-complete-calibration-review-20260915-v1.json)

- [Complete first-LR development review](../../results/fast-chart-lr-1e-07-development-review-20260915-v1.json)
- [Complete second-LR development review](../../results/fast-chart-lr-1e-06-development-review-20260915-v1.json)
- [Matched three-LR configuration and third-job submission](../../results/fast-chart-third-lr-start-20260915-v1.json)
- [Actual image/parameter handoff for the second LR](../../results/fast-chart-second-lr-parameter-handoff-20260915-v1.json)
- [Actual image/parameter handoff for the third LR](../../results/fast-chart-third-lr-parameter-handoff-20260915-v1.json)
- [Fixed-processor export compatibility](../../results/fast-chart-real-processor-contract-20260915-v1.json)
- [Official GLM price verification](../../results/fast-chart-glm-public-pricing-20260915-v1.json)
- [Runbook, endpoint freeze and full-cost forecast commands](../fast_chart_runbook.md)

The current paper build is identified in the runbook. Its generated
claim-evidence-index.json is authoritative for the evidence actually included
in that particular PDF; this checklist does not silently upgrade its claims.
