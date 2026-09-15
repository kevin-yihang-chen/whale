# Compact VETO execution and evidence map

This is the authorized2026-09-15 scope. The51-condition schedule, Chess, CLEVR,
and2B remain historical and paused. The scientific VETO claim is unvalidated.

## Current automatic continuation

Job231666 completed the three registered h0 prompts on all256 V pairs.
Structured was selected by ordinary V accuracy (452/512; paired214/256),
with0.996111GPUh charged. The other scores and complete raw evidence remain.
The first LR1e-7 trial is native training job232265 (two H800 GPUs).
The single controller at `data/fast-chart-20260915-v1/lr-calibration-v1/`
waits for its complete Slurm terminal record, then prepares three seed42
four-batch LR trials and their verified export/V followups, one job at a time.
Do not start another controller. Read `controller-status.json`, `controller.log`
and the corresponding allocation receipts. A failed job or cap violation stops
the controller; it never silently retries, chooses a partial score, or opens T.
It ends at shared h0/LR selection. Formal expansion requires the measured cost
and storage forecast; the controller does not launch the18-condition matrix.
One CPU watcher now waits for that result and writes
`lr-calibration-v1/expansion-forecast.json`. Its process receipt is
`forecast-watcher-process.json`; do not start a duplicate. It has a24h wait bound,
preserves calibration failure, and never submits, cleans up, or changes caps.

GPU accounting reuses the historical visual ledger and enforces the new100h
total including6.8980555556h previously charged. Setup20h, core40h, ablation15h,
evaluation20h, retry5h. Every Slurm limit is reserved before submission; actual
terminal allocation cost is charged even after failure. Mail-type ALL remains
set on every wrapper. Notification configuration is not proof of email delivery.

## Reproduction environment

The native Python is `data/training-runtime-v1/bin/python`; set PYTHONPATH to
`data/visual-runtime-overlay-v1:ours/compat:upstream/WHALE/domains/chess_puzzles:.`.
Export the trial's WHALE_TRIAL_SEED and PYTHONHASHSEED before launching training
Python. Use local assets and the supplied Slurm wrappers. Public-source
acquisition and GLM proposals run from the networked entrance; compute jobs
remain offline. Paper plotting uses the existing system Python with matplotlib.

## Modules and equations

| Module | Method / architecture block | Evidence status |
|---|---|---|
| chart_answer_protocol | Shared host parser/verifier for E1/E4 | Unit checks and real V evaluation |
| plotqa_multitask | Shared task construction, unchanged source groups | Complete rendered-data and pixel checks |
| visual_sampler_reference | E4 state restoration reference | Actual corrected historical sampler verified |
| fast_chart_training + hooks/worker | E4 four-batch native RSFT, restore and received weights | CPU checks; real compact training232265 started |
| fast_chart_followup | E4 full native parameter comparison and serving handoff | Reuses verified primitives; new compact run pending |
| fast_chart_search_evaluation | Shared H score and E1 C audit | CPU adapter; real compact search pending |
| fast_chart_search | Native one-round3-slot search, E2–E3 and resumed handoffs | Native-loop CPU tests; no paid compact proposal yet |
| fast_chart_augmentation | Same-budget data augmentation control | CPU sampling checks; real continuation pending |
| chartqa_subset/open_answer + endpoint evaluation | Shared independent evaluation |512questions acquired; scorer hook checked; no model test score |
| fast_chart_statistics/figures/paper | Intervals, evidence figures and manuscript | Statistics checked; manuscript scaffold compiled |
| fast_chart_admission/forecast + endpoint integrity | Engineering/resource admission and fixed endpoint evidence |5CPU checks; real evaluation-throughput preview |

The acceptance rule remains the existing EvidenceConstrainedAcceptance. No
extra loss or attention architecture is introduced. The generated candidate's
parse_answer is ignored by the host. OS callback confinement remains active.

## One seed's training/search/continuation sequence

1. Use `fast_chart_training prepare/check/run` for stage1. The chosen LR
   calibration trial can be the seed42 prefix only when its complete settings
   match. Seeds43/44 require their own real stage1.
2. Use `fast_chart_followup prepare/check/run` to compare every native active
   parameter, export canonical BF16, and measure V. A changed receiver sample
   is used when BF16 updates provide at least eight changed coordinates.
3. `fast_chart_search_evaluation prepare` prepares h0 H128+C64 at the verified
   weights. Submit through `fast_chart_submit --kind search --phase core`.
4. `fast_chart_search init/advance` records the fixed incoming reference and
   requests one3-slot proposal. `propose` uses the existing GLM journal with
   cumulative45CNY cap and a positive H-only feedback allowlist. No C/V/T/R
   labels or scores enter the proposer workspace.
5. Evaluate each real candidate once. `attach` accepts complete matching
   allocation evidence; `failed-evaluation` records failed candidates without
   inventing zero scores or replacement slots. `advance` resumes the native
   search and writes whale/veto/marginal_gate selection receipts. All ordinary,
   early-stop and resume boundaries use the same acceptance adapter.
6. Prefer `fast_chart_condition --condition ... --configuration ... --parent-plan ...`
   to bind the shared h0/LR freeze, archive and same-budget augmentation receipt.
   Its underlying stage2 `fast_chart_training prepare --parent-plan ... --selection ...`
   restores native step4 and its real data.pt, checks the incoming-harness
   independent sampler reference, then trains steps5–8. All branches retain
   the original Adam-reset behavior. No missing check is bypassed.
7. For augmentation, build `fast_chart_augmentation` using the WHALE receipt,
   then add `--augmentation <manifest>` to stage2. Two complete C pairs replace
   four ordinary slots in each batch; total32questions/256trajectories stays
   fixed. It is explicitly a payload replacement in the same sampler positions.
8. Followup compares theta2 directly with native theta1 as well as with theta0,
   preserving BF16 rounding distinctions. Register the fixed endpoint before
   preparing T/R/ChartQA or the WHALE/VETO common-h0 probe.

Harness-only performs its own search at theta0; its candidate archive is not
reused from theta1. Weight-only keeps the frozen h0 for both real weight stages.
Shared physical computations must remain explicit in result and cost tables.
An unchanged winning LR stops expansion; do not substitute a runner-up selected
by its larger update. Completed zero-update formal branches remain reportable.
The current automatic controller covers calibration; these later steps are
explicit entrypoints and have not yet run as a compact scientific experiment.

Run `python -m ours.fast_chart_forecast --configuration <complete LR result.json>
--output <fresh report.json>` after calibration. The early h0-only projection is
`results/fast-chart-throughput-preview-20260915-v1.json`:33.84GPUh for all24
T passes and18external passes, above the20h final-evaluation allocation. This
is a measured-throughput proxy, not an incurred cost or an approved phase
transfer. Validate a common throughput improvement and complete training/storage
measurements before expanding. The forecast never launches or deletes anything.

## Data, storage and paper

New data: `data/fast-chart-20260915-v1/dataset/`; original source roles remain
W2048/H128/C64/V256/T1024/R1024. Each role is balanced across comparison,
single-value and difference questions. C is optimization data. R retains a V
evaluator role with explicit partition R; it is not the ordinary development V.

External subset: `chartqa512-v1/`,256human+256augmented questions on485 unique
images. Exact PNG and normalized labeled-table overlap checks against W/C
found no matches. Semantic near-duplicates and model pretraining overlap are
not exhaustively established. It is never labeled the full ChartQA score.

All generated runtime artifacts use the compact output root; maximum new
allocation400GiB and free margin40GiB. Native first-stage and final export
retention must be forecast from actual files before expanding. Keeping every
native final checkpoint plus every serving copy can exceed400GiB; no deletion
is automatic. Any reclamation needs exact paths, sizes, references and a new
explicit authorization. B-group data and old project files stay protected.

`python -m ours.fast_chart_paper --output <fresh compact build directory>`
compiles the manuscript and supplement with the local Tectonic tool/cache.
The latest complete build is `paper-build-v8`, with its source/evidence index,
actual h0 calibration table and corrected font encoding/reference back links.
The official author-kit checkout currently says2026; it is a provisional
drafting layout, not certification of the2027 submission format.
`fast_chart_figures` requires complete real records before drawing stage and
cost plots. Empty result cells are intentional. Claims map to raw sample
records and immutable paths/hashes through the build's claim-evidence index.
