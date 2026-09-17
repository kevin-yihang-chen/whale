# Compact VETO execution and evidence map

## First native cycle completed (2026-09-16 10:08 HKT)

Jobs233990/234097 completed; controller2574402 exited after its bounded task.
No allocation remains active. Reconstructed evidence is in
`data/fast-chart-20260915-v1/first-cycle-report-v2/result.json`.
The V trajectory is452/512 and214/256 initially;448/512 and207/256 after stage1;
447/512 and212/256 after stage2 with h3. All three gates selected h3. This is
not an independent gate comparison, and unexecuted branches are not imputed.

Build with the existing calibration, search and case options plus
`--cycle-report data/fast-chart-20260915-v1/first-cycle-report-v2`.
Current local draft: `paper-build-v19`. The complete cycle reporter is
`python -m ours.fast_chart_cycle_report --cycle data/fast-chart-20260915-v1/first-cycle-controller-v1/result.json --configuration data/fast-chart-20260915-v1/lr-calibration-v3/result.json --output <fresh-output-under-compact-root>`.
This is read-only E4 evidence reconstruction; it performs no training or API call.

Remaining full scope and exact deletion/phase questions are in
`ours/fast_chart_resource_decisions_20260916.md`; no approval is inferred from
elapsed time or the earlier GitHub-publication request. Current settled GPU
cost15.938333h, no reservation. Full remaining costs count completed work once.

This is the authorized2026-09-15 scope. The51-condition schedule, Chess, CLEVR,
and2B remain historical and paused. The scientific VETO claim is unvalidated.

## Current first-cycle handoff (2026-09-15 23:17 HKT)

All candidate jobs completed. WHALE, VETO and marginal gate all selected h3;
the real reconstruction is `data/fast-chart-20260915-v1/first-search-report-v1/`.
Controller2574402 is preparing the prospectively fixed VETO continuation with
child2651574. h3's complete2048-input filter passed; it next verifies the actual
incoming-h0 loader and saved data.pt. CPU preparation takes time; do not submit
another stage2 plan while this process runs. There is currently no GPU job.

`paper-build-v15` includes complete optimization counts and the post-hoc case
from `illustrative-case-v1`, with unchanged PNGs and real responses. Six pages
were visually inspected. Main method/endpoint tables remain placeholders;
equivalent choices and the fact that paired failure does not diagnose visual
non-use are explicit. A separate case float page remains a draft layout issue.
Use `--search-report data/fast-chart-20260915-v1/first-search-report-v1 --case
data/fast-chart-20260915-v1/illustrative-case-v1` with the ordinary paper builder
and the complete configuration. The precise included claims are bound by the
v15 claim-evidence-index; no formal method completion is asserted.

## Earlier first-cycle handoff (2026-09-15 22:40 HKT)

22:50 update: h1 completed233589,111/128 H,54/64 C pairs,115/128 C images;
both paired and marginal gates admit it. h2 now runs as233705 on1H800.
After all slots complete, use `python -m ours.fast_chart_search_report --search
data/fast-chart-20260915-v1/formal-v1/seed42/first-search --output
data/fast-chart-20260915-v1/first-search-report-v1` with the native runtime and
existing offline PYTHONPATH. This only reads/certifies evidence; it does not
regenerate candidates, call an API or retrain. Then supply that directory to
`ours.fast_chart_paper --search-report` for the evidence-bound supplement table.
Do not run the report before the complete search result exists. Four CPU tests
passed in `results/fast-chart-search-report-tests-20260915-v3.log`; real output
and a new PDF build are still pending.

Controller2574402 under `first-cycle-controller-v1` reuses active candidate job233589,
then completes h2/h3, native selection, the prospectively fixed seed42 VETO stage2,
and full followup. Do not manually submit these same plans while it runs. Scope
is exactly the existing six-GPU-hour worst-case setup admission; it cannot make
API calls, delete files, run other conditions or evaluate sealed endpoints.
Launch/test receipts are `results/fast-chart-first-cycle-controller-launch-20260915-v1.json`
and `results/fast-chart-first-cycle-controller-tests-20260915-v2.log`.

H/C baseline233540 completed:107/128 H,49/64 C pairs,110/128 C images. The paid
proposal completed once with3 generic candidates; the actual GLM trace did not
read the available H feedback. Preserve the tool-use limitation and actual pool.
The earlier automatic transfer rejection was resolved with existing explicit
thread consent and exact H-only transfer evidence, not a bypass or new permission.
No independent VETO result exists yet. Full-matrix retention and phase transfers
remain pending; the bounded first cycle fits without either change.

## Earlier live handoff (2026-09-15 22:00 HKT)

233485 completed both throughput passes and settled0.508056GPUh. The shared
endpoint policy is frozen at32 by speed, with every answer change retained.
All1536 baseline/new raw answers were rescored; see
`results/fast-chart-throughput-completion-review-20260915-v1.json`.
Controller2431747 submitted setup H/C job233540 at21:57:19; do not duplicate it.

The complete forecast is in `setup-hc-controller-v1/measured-expansion-forecast.json`: 
80.66GPUh overall,23.52GPUh in final evaluation. The [phase reallocation proposal](fast_chart_phase_reallocation_20260915.md)
asks for core35/evaluation25 with the total100 unchanged. Both it and exact-path
retention consent are pending; the current setup measurement can continue.
Later sections preserve earlier chronology and entrypoint guidance.

## Current automatic continuation

The active goal now supervises recovery from job232265's post-SFT transport OOM.
The old `lr-calibration-v1` controller stopped and is preserved. Use the single
new controller under `lr-calibration-v3`; its first trial is a diagnosed retry
charged to the5GPUh retry allocation. It compares all experimental settings with
the failed plan before submission. Remaining new LR trials use the setup phase.
The preparation repair only releases idle allocator caches before native NCCL
buffer allocation; it preserves live allocations, tensor values and bucket size.
If the job chain fails again, retain its terminal evidence and let the active
goal diagnose it rather than treating a failed script as goal completion.

The v2 CPU preflight was stopped before GPU submission because its inherited
thread environment created72 threads. Its complete logs are preserved. The v3
launcher matches the Slurm wrapper's one-thread CPU settings and repeats the
actual dataset/sampler checks; no training setting or source was changed.
Full2048-row and actual four-batch preflight passed. Job232741 completed all
four batches with191 accepted trajectories and25 native updates, charged1.27GPUh
to retry. Every post-SFT transfer returned, and global_step_4 is preserved.
Followup233025 verified all191 consumed images and nonzero visual gradients,
the complete native update and its exact BF16 export. Its complete V512 result
is443/512 single-image and212/256 paired correct, below the common initial
452/512 and214/256. Retain this calibration outcome; it is not a VETO result.
233025 completed at19:08:39 for0.418056GPUh. Job233172 completed LR1e-6
on2H800 with the same2048-row population,32 sampled questions, h0 and source
identities; only LR and output paths differ. It used1.257778GPUh from setup,
with192 accepted trajectories and25 updates. Followup233258 completed its full
image/parameter/export audit and V512:448/512 ordinary,207/256 paired correct,
below the initial452/512 and214/256. It used0.331944GPUh.
The controller saw its COMPLETED allocation before NFS exposed followup/result.json.
The original result subsequently appeared and all identities passed; it was not
reconstructed. Bounded300s artifact visibility waits and explicit --resume now
preserve old failure/start receipts. Five checks pass. Resumed PID2231441
submitted1e-5 as job233361 at20:31:05 on2H800 after its complete preflight;
all three input/source/configuration identities match except LR and output paths.
233361 completed at21:01:41 for1.02GPUh, with144 accepted trajectories and20
native updates. Followup233399 started at21:04:43 on1H800; its full144-image
consumption/gradient audit and complete parameter comparison passed. Its V512
plan binds the verified export; complete V512 is280/512 ordinary and89/256 paired.
All512 raw answers were rescored; the three-rate rule selects1e-6.
All three native stages and their full parameter audits have now finished.
It uses the same v3 directory; no GPU work is repeated and no partial LR
winner is selected. The original forecast watcher stopped on the preserved
failure; run the fresh full forecast after all three trials and throughput finish.
Controller2231441 has completed and exited. Throughput job233485 started at21:26:00 on1H800. Do not duplicate it.

Canonical exports reorganize tokenizer/processor files. Endpoint binding keeps
both raw asset identities and requires equal loaded model/generation settings,
complete tokenizer backend, image/video processors and templates. Real base/export
CPU pixel/grid/token checks agree at three sizes;9 related tests pass. Raw file
drift or a behavioral difference still fails; no identity check is skipped.

The paper builder accepts `--configuration` only after all three LR trials are
complete. It traces every row to its result, training plan and native transition,
rechecks the shared selection rule, and identifies the reused seed42 prefix.
The complete LR configuration is now available for a fresh manuscript build;
`paper-build-v14` includes all three actual LR rows and the selected1e-6 prefix.
The v13 build retained its missing-font failure; v14 fetched the public cmr9
font into the existing TeX cache and compiled both PDFs.
The E1 explanation now includes P=2M-1+q0, a constructed correctness example,
and its limitations. Different-weight V decompositions do not prove different
gate selections on a fixed-weight C archive.

The independently prepared `results/fast-chart-throughput-plan-20260915-v2.json`
measures two full V passes at batch16 and32, against the existing batch8 record.
Job233485 is running after calibration completion, submitted through
`fast_chart_submit --kind throughput --phase setup`; it reserves one GPU hour.
The unsubmitted v1 plan is preserved. v2 adds separate materialization/dataset
timing; the full forecast scales preparation above512 images rather than treating
it as constant model loading. Eleven targeted checks pass. Selection uses seconds per image, not the best score, and all answer
changes are retained. This is shared E1 execution infrastructure, not a VETO
mechanism. No T/R data is opened or h0/LR reselected by this measurement.

After the throughput job reaches COMPLETED, freeze its measured execution
policy before any independent endpoint is submitted:

```bash
python -m ours.fast_chart_inference_policy --result data/fast-chart-20260915-v1/throughput-v2/result.json --suite results/fast-chart-throughput-plan-20260915-v2.json
python -m ours.fast_chart_forecast --configuration data/fast-chart-20260915-v1/lr-calibration-v3/result.json --measured-endpoint-policy --output results/fast-chart-measured-expansion-forecast-20260915-v1.json
```

This fresh forecast also charges the slowest observed base/updated-model loading
to every future evaluation allocation. The first updated model loaded in280.66s,
against251.69s for the slowest h0 case; source-matched completed LR records are
required. Three related checks pass. The existing watcher loaded an older module;
its preview is not the final admission forecast.

The full forecast also requires all files in the recorded native/export model
inventories to remain present and nonempty. Missing checkpoints must not be
measured as zero-sized models. Five related checks pass, and both completed LR
inventories passed the actual presence/size measurement. If cleanup is later
authorized, retain the pre-cleanup measured forecast and use an explicit retention
calculation; do not rerun raw model-size estimation on removed checkpoints.

The single `shared-inference-policy-v1.json` binds all methods/seeds on T/R
and ChartQA. Endpoint preparation requires it and cannot select a separate
batch size. W training, H/C search, V calibration and their token budgets stay
unchanged. The policy keeps real answer changes from the batch comparison;
faster batching is not assumed to preserve answers bit for bit. No policy is
currently frozen because the throughput measurement has not run.

The CPU forecast watcher was started before this endpoint-policy extension;
its initial report remains conservative at batch8. The explicit second command
above consumes the completed measurement and replaces only endpoint/R throughput
estimates. It still reports all loading, training, storage and budget gaps, and
never grants formal scientific admission or starts jobs.

`python -m ours.fast_chart_costs --output <fresh-report.json>` verifies settled
GPU amounts against stored Slurm terminal records and includes every global API
ledger entry. Unsettled reservations remain explicit bounds, not zero charges.
With `--manifest <attribution.json>`, each resource must appear exactly once with
named beneficiaries and a reason; shared resources are divided equally among
those beneficiaries. The shares sum exactly to the project total, including
failures. This is attributed cost for the executed shared study, not standalone
algorithm cost. Keep historical project overhead explicitly named; do not hide
it or mix it into an unlabelled method figure. Final beneficiary assignments
depend on the actual completed branches and are not yet frozen.

The following paragraph records the original calibration launch history:

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
| fast_chart_training + hooks/worker | E4 four-batch native RSFT, restore and received weights | Three complete real LR stages; stage2 continuation pending |
| fast_chart_followup | E4 full native parameter comparison and serving handoff | Three complete image/parameter/export audits and V512 passes |
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
The latest complete local build is `paper-build-v28`. It includes all three real
LR rows and the complete seed42 V3 optimization archive. Its evidence index binds
the h0--h3 report, the versioned V3 report source and the equivalent h0 decisions
for WHALE, VETO and the point gate. The manuscript explicitly records the failed
distinct-selection gate and contains no fabricated continuation result. Main and
supplement PDFs contain3 and2 pages, respectively. `paper-build-v27` is an
incomplete local build attempt from an environment without matplotlib and is not
a manuscript artifact.
The official author-kit checkout currently says2026; it is a provisional
drafting layout, not certification of the2027 submission format.
`fast_chart_figures` requires complete real records before drawing stage and
cost plots. Empty result cells are intentional. Claims map to raw sample
records and immutable paths/hashes through the build's claim-evidence index.

## Prepared next step and pending retention consent (2026-09-15)

`results/fast-chart-seed42-first-search-h0-plan-20260915-v1.json` is prepared,
not submitted. It binds the selected1e-6 updated model and common structured h0
to complete H128+C64-pair measurement. It will initialize the E2–E3 fixed-weight
candidate archive only after resource admission. No proposer has been called.

[The retention proposal](fast_chart_retention_20260915.md) identifies four existing
nonselected calibration model directories (56.741GiB) and15 proposed final
native-checkpoint paths. Consent is pending; nothing has been deleted. Selected
common prefixes, all final serving models and complete scientific evidence remain
retained. The pre-throughput forecast is
`results/fast-chart-complete-calibration-forecast-before-throughput-20260915-v1.json`;
it still exceeds the20GPUh final-evaluation phase at batch8 and cannot admit expansion.
Use the completed throughput-v2 measurement for the final forecast.

The first seed42 H/C measurement is part of the explicitly authorized days1–3
corrected visual-chain closure (setup20GPUh). Its bounded admission is
`results/fast-chart-seed42-setup-hc-admission-20260915-v1.json`. After233485 and
its observer finish, submit that prepared h0 plan with `--kind search --phase setup`,
subject to fresh per-phase/total reservation checks. It needs no new checkpoint
or cleanup. Broad18-condition expansion remains gated; completed shared work
must be counted once and removed from remaining-work forecasts.

The bounded setup handoff is now supervised by PID2431747 under
`setup-hc-controller-v1`. Four tests cover failed/pending prerequisites, NFS
visibility, no duplicate submission and rejection of larger/formal scope. It
freezes throughput-v2 and writes its measured forecast in its own output directory,
then submits only the one prepared setup H/C job and certifies its complete result.
No paid proposer or stage2 is launched by this controller. Do not submit its job
manually while it is live.
