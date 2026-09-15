# VETO compact execution,2026-09-15

Current authority and commands: [compact runbook](fast_chart_runbook.md),
[frozen protocol](fast_chart_protocol_20260915.md).
The18 logical conditions share real prefixes;51-run expansion, Chess, CLEVR
and2B are paused. Budget100GPUh includes historical visual costs; new storage
cap400GiB and cumulative GLM cap45CNY. No automatic cleanup or GitHub push.

Real full-V calibration is running. New native4batch/search/continuation entrypoints
are implemented with CPU checks; the compact scientific loop and VETO benefit
remain unvalidated. The calibration controller ends at common h0/LR selection
and does not automatically launch the formal matrix. Main and supplemental
LaTeX compile with explicit pending results; see PROJECT_STATUS.md for evidence.

The material below records the earlier2026-09-10 implementation state. Later
explicit user authorizations and the current compact protocol take precedence.

# VETO implementation and integration status

**VETO — Visual Evidence Tests for Optimization** is an unvalidated research
hypothesis. This directory contains the acceptance module, recorded-evaluation
CLI, shared visual input/reward/tool adapters, provider configuration and reproduction checks.
Real VLM evaluation and a native RSFT batch have completed. Visual search,
acceptance and resumed training have not yet formed an operational closed loop.

`visual_search_bridge.py` now stages the original `run_evolve` across real H/C
evaluation and a networked proposer. The actual h0 archive has reached initial
acceptance; a real h1 proposal awaits explicit external-transfer scope approval
after automatic review rejection. `visual_search_contract.md` defines that
proposer's visible inputs. Four CPU tests cover native ordinary/early-stop
control flow, off delegation, receipt resume/tampering and H-only feedback.
They do not prove an executed h1 or subsequent training. The proposed external
transfer is listed in `results/visual-proposer-transfer-review-20260910-v1.json`.

The original protocol is [veto_execution_protocol_20260910.md](veto_execution_protocol_20260910.md).
The later [visual priority amendment](visual_priority_plan_20260910.md) takes
precedence: Chess continuation is deferred and the 51 conditions are a historical
design, not the current execution queue. The old milestone controller is stopped.
`research_budget.py` reserves full allocation limits before submission and charges
terminal failures. Formal expansion remains disabled. Current work uses existing
storage for small visual diagnostics and one native training checkpoint. Further
allocations are individually bounded; the original 51-condition matrix is inactive.

`native_visual_service.py`, `native_visual_agent_observation.py` and
`native_visual_model_worker.py` connect real serving startup to E1 with image,
decoding, token and loaded-parameter observations. The current frozen plan uses
engineering inputs; actual GPU serving has completed. Initial tool-coordinate and
answer-format failures remain archived separately from the subsequent successful
direct-answer screen. The shared visual harness supports `execution=isolated`
through `isolated_visual_harness.py`, `visual_callback_worker.py` and
`callback_confinement.py`. The worker applies kernel filesystem/network/process
restrictions before loading candidate code; there is no weaker automatic fallback.
These are shared experimental execution blocks, not additional VETO mechanisms.

`acquire_visual_sources.py` preserves bounded official downloads and provenance.
`visual_source_inventory.py` explicitly excludes nonfinite source records.
`plotqa_evidence_pairs.py` prepares source-table-disjoint W/H/C/V/T/R recipes,
re-renders the charts and checks saved bar pixels against independent arithmetic.
Its derived binary comparison task is named **PlotQA-EvidencePairs**. W/H retain
original values; paired roles exchange two numerical cells.
This preprocessing supplies E1/E4 inputs; it does not modify the loss or acceptance.
The 2026-09-10 render has since completed: 4,544 source tables and all 6,912 PNGs
pass the saved-pixel numerical checks. `visual_dataset_materialization.py` now
provides native W/H/C/V Parquet inputs (2,048/128/128/512 images), with model
messages limited to questions and pixels. It retains C's optimization role and
does not materialize T or R. The first64 frozen V pairs now support a bounded
direct/blind/normalized-zoom screen; this is not a VETO efficacy experiment.

```bash
python -m ours.visual_source_inventory --receipt COMPLETED_SOURCE_RECEIPT --output NEW_INVENTORY
python -m ours.plotqa_evidence_pairs --phase prepare --inventory COMPLETED_INVENTORY_RESULT --output NEW_DATASET
python -m ours.plotqa_evidence_pairs --phase render --output PREPARED_DATASET
```

The first16-image screen and its coordinate diagnostic have completed. The next
64-pair development screen is source-bound by
`results/visual-development-screen-plan-20260910-v1.json`; current outcomes are
in [PROJECT_STATUS](../PROJECT_STATUS.md). It runs three diagnostic conditions
sequentially using one existing4B model, without training or proposer calls.
`canonical_answer_harness` accepts an unambiguous standalone final A/B line;
every new condition shares it. Earlier strict scores are retained unchanged.
Before broader allocations, review [storage_review_20260910.md](storage_review_20260910.md).
The user explicitly approved group A and declined group B on 2026-09-10.
Group A is complete: listed old model/download caches were removed and identical
Chess weights share one inode with both paths retained. Available space rose to
219.14 GiB; MIMIC images remain in place. Further cleanup requires separate
approval, and the full formal matrix still lacks a validated storage plan.

After the development job has completed and its terminal allocation is archived,
recompute E1 counts and source-paired descriptive intervals without model calls:

```bash
python -m ours.summarize_visual_development_screen \
  --plan results/visual-development-screen-plan-20260910-v1.json \
  --output results/visual-development-screen-rescore-NEW.json
```

## Method correspondence

| Implementation | Equation / architecture block | Current scope |
|---|---|---|
| `evidence.py:VisualPairEvaluator` | E1: \(P_C=\frac1n\sum_i v(\hat y_i,y_i)v(\hat y'_i,y'_i)\); Paired Visual Audit | Computes paired and marginal accuracy from complete, manifest-bound predictions |
| `normalized_visual_zoom.py`, `normalized_visual_tools.yaml` | Shared image tool: pixel box = normalized box × (width,height,width,height)/1000 | Reuses the native crop and rewards; two actual-pixel CPU checks pass; not an acceptance mechanism |
| `visual_harnesses/canonical_answer_harness.py` | Shared answer normalization before E1/E4 verification | Explicit final-line parsing without truth access; archived strict scores are not overwritten |
| `visual_blind_control.py` | Diagnostic removal of image input before E1 | Native processing and actual requests verify image absence; labels remain private grading inputs |
| `visual_coordinate_diagnostic.py`, `visual_development_screen.py` | E1 diagnostic measurement with fixed model and data | Real model screens; no E2/E3 selection or E4 update |
| `summarize_visual_development_screen.py` | E1 recomputation and matched descriptive contrasts | Requires complete role-V results and terminal allocation; source-paired intervals do not replace independent training seeds |
| `native_visual_training.py`, `visual_training_bootstrap.py` | E4 native successful-trajectory RSFT and full multimodal evidence | Job224041 completed64 fresh trajectories on8 W questions,47 accepted,6 reported optimizer steps; original loss/filter unchanged |
| `visual_training_identity.py`, `visual_training_agents.yaml` | E4 sample provenance through native reward-worker collation | Inherits the shared visual loop and preserves only the missing sample identity in extra fields; two native recording/collation checks pass |
| `audit_native_visual_training.py` | E4 input/reward/success-subset verification | Replays all64 actual W inputs, model-visible pixels, tokens and replies;47-row success subset matches exactly; not a weight-change or performance proof |
| `native_visual_followup.py`, `run_native_visual_followup.sh` | E4 → E1: canonical checkpoint export and fixed-harness development reevaluation | Reuses the original merger and full active-tensor comparison; checks identical actual input/decoding multisets and paired before/after answers; no search or new training |
| `audit_visual_followup_inputs.py` | E1 input comparability across E4 checkpoints | All128 archived native-agent pixel, grid, prompt and position tensors agree; actual inference repeatability is separate |
| `visual_followup_repeatability.py` | E1 repeated fixed-model measurement | Reuses completed initial/trained checkpoints on the full same64 V pairs; keeps every replicate, no training or checkpoint selection |
| `acceptance.py:EvidenceConstrainedAcceptance` | E2: \(P_C(g)-P_C(g_{in})\ge-\epsilon\); E3: maximize task accuracy, then minimize turns | Inherits task-score selector; supports `off`, `paired`, `marginal_gate`, `marginal_rank`, `soft_pair` |
| `adapter.py:WHALEAcceptanceAdapter`, `visual_search_bridge.py` | E2–E3: Acceptance Gate → next training harness | Native-loop boundary implemented for initial/ordinary/early-stop/resume; actual h0 initial acceptance completed, other paths CPU verified, real candidate pending |
| `visual_search_evaluation.py` | Shared H ranking and E1 optimization audit C | Actual h0 H/C passes and repeats complete; remaining inference variability is recorded, first pass controls selection |
| `visual_resume_data_probe.py` | E4 saved-data continuation | Actual native CPU sampler restore and next eight W inputs pass; no GPU actor restore or training |
| `visual_training_resume.py`, `visual_resume_bootstrap.py`, `visual_resume_model_worker.py` | E2–E3 decision → E4 native restore and weight transport | Complete search receipt required; six CPU checks and actual W input preview pass; no real candidate selection, GPU resume or second update yet |
| `visual_evidence_dataset.py:VisualEvidenceDataset` | Shared observation channel `(I,q) → pi_theta` before E1/E4 | Native subclass preserves images across prompt overrides and counts expanded image tokens; actual direct-answer visual training completed |
| `visual_evidence_reward.py:compute_score` | E4 shared task verifier `r_i=v(answer_i,y_i)` before original success filtering | Actual visual RSFT accepted47/64 freshly generated trajectories; unchanged binary reward and native filtering |
| `visual_evidence_tool_loop.py`, `visual_evidence_agents.yaml`, `visual_evidence_tools.yaml` | Shared observation `I_next=crop(I_initial,b)` and E4 final-answer verification/masks | Real crop evaluation and direct-answer actor updates completed separately; training on successful crop trajectories remains unverified |
| `visual_harness.py`, `visual_harness_dataset.py`, `visual_harness_loop.py`, `visual_harnesses/base_harness.py` | Shared E3 search space `h=(observation, tool_arguments, feedback, parser, continuation)` → E4 committed-answer reward | Five callbacks execute in the native dataset/tool loop; source identity, prompt subspace, crop trace and native batch collation pass CPU checks; actual visual orchestration and VETO selection remain pending |
| `visual_native_evaluation.py` | E1 `P_C=mean(v(answer,y) v(answer_prime,y_prime))` from the shared native E4 rollout path | Actual evaluation of64 V pairs completed; full coverage, parser/reward/masks and sample identity checked; live VETO selection remains pending |
| WHALE upstream RSFT | E4: existing success-filtered weight update | Attributed to WHALE; unchanged; no new loss implemented |
| `chart_pairs.py` | Inputs to E1 | Eight engineering pairs, independently checked from rendered pixels; not a benchmark |
| `visual_task.py:VisualTaskAdapter` | Image and question → target → E1 | Shared input formatting; excludes labels, pair IDs and paths from model messages |
| `run_visual_smoke.py`, `run_visual_smoke.sh` | Measurement of the E1 input connection | Fixed cached model, raw output ledger, source/weight/input hashes; no training or search |
| `local_completion.py`, `compat/autoharness_textarena/` | Shared target execution before the E1–E4 loop | New local HF implementation of the missing Chess completion interface; original runner/verifier unchanged |
| `prepare_chess_runtime.py`, `run_chess_smoke.py`, `.sh` | Baseline execution checks | Pinned inputs, checkpoint identity, real token ledger and guarded submission; not a Method contribution |
| `vllm_runtime.py`, `run_vllm_runtime.sh`, `audit_vllm_runtime.py` | E4 shared execution and independent measurement | Local vLLM HTTP calls through the original Chess runner; complete token traces and independent move replay |
| `native_search.py`, `run_native_search.sh`, `audit_native_search.py` | Shared candidate generation → evaluation → E3 selection | Calls the original Meta-Harness loop with GLM and a frozen local target; VETO off; rejects incomplete evaluations and imports only declared proposer artifacts |
| `prepare_chess_splits.py` | Shared experimental data protocol | Pins Lichess, delegates the original converter/splitter and excludes all inspected engineering IDs and position groups |
| `training_bootstrap.py`, `run_native_rsft.sh`, `config/data/legacy_data.yaml` | E4 native training execution | Explicit bundled-package/config restoration; native dataset/configuration and one GPU optimizer step verified |
| `chess_bootstrap.py`, `audit_chess_bootstrap.py`, `replay_chess_agent_loop.py` | E4 successful-trajectory prerequisite | Audited bootstrap sampling and original AgentLoop reward/mask replay; optimizer evidence is recorded separately |
| `run_native_rsft_pilot.sh`, `rsft_pilot_gate.py`, `check_checkpoint_transition.py` | E4 optimizer and checkpoint handoff measurement | Frozen native execution and dependency preflight; step222156 completed, reviewed canonical value audit222165 passed |
| `native_training_trace.py`, `recorded_training_bootstrap.py` | E4 execution evidence | Native subclasses preserve chat requests, errors, exact training tensors/masks and events; CPU tests and complete64-row live capture passed in221967 |
| `probe_native_optimizer.py`, `run_native_optimizer_probe.sh` | E4 optimizer execution check | Native ZeRO-1 CPU comparison and bounded synthetic GPU memory probe; no real checkpoint update or model forward/backward |
| `audit_native_training_batch.py` | E4 recorded training evidence | Original AgentLoop replay against request journals and exact tensors, plus independent board checks;8 targeted tests and full live64-row request/tensor/board audit pass |
| `audit_alternation_training.py` | E3 selected harness → E4 accepted-trajectory verification | Four CPU checks and complete222292 batch pass:64 trajectories,11 accepted on two puzzles; subsequent actor forward OOM is separate |
| `verify_alternation_transition.py` | E4 incoming canonical checkpoint → next native/exported parameters | Checks the same active graph and separates FP32 updates from BF16-surviving updates; serialized-tensor CPU fixture and actual723-tensor checkpoint audit222703 pass |
| `tests/test_native_fused_rsft.py` | E4 original objective under a shared execution backend | Released Torch fused-output backend passes small hybrid-model FP32/BF16 CPU checks; BF16 has rounding differences and requires matched controls;4B/FSDP execution completed222640; dense/fused gradient comparison remains CPU-only |
| `alternation_recovery.py`, `recovered_alternation_bootstrap.py`, `run_alternation_recovery.sh` | E4 interrupted-phase replay and native backend selection | Full frozen config, one-use audited batch, unchanged original filter/minibatches/optimizer/transport; three CPU integration checks pass; GPU222512/222615 backward OOM;222640 completed two original updates and saved a checkpoint |
| `native_transport_memory.py` | E4 weight transport → training memory lifecycle | Subclasses original NCCL finalize to release idle CuPy blocks; two native CPU boundary fixtures pass;222640 real GPU fixture passes; initial/final native sync each releases6GiB+512 bytes; full training completes |
| `compact_native_batch.py`, `recovered_training_bootstrap.py`, `recovery_gate.py` | E4 interruption recovery | Reuse the audited64-row batch once; retain original successful-row objective; only remove masked response suffix |
| `alternation_checkpoint_export.py`, `run_alternation_checkpoint_export.sh` | E4 completed update → canonical export and full parameter audit | Rejects real failed recovery evidence; 222703 completed canonical export and full723-tensor audit;14,479,791 changed values survive BF16 export |
| `canonical_native_export.py`, `verify_native_transition.py` | E4 checkpoint serialization and parameter measurement | Preserve canonical active names/base inference settings; audit all values and separate FP32 update from BF16 serialization |
| `probe_updated_vllm.py`, `vllm_weight_probe_worker.py`, `run_updated_vllm_probe.sh` | E4 checkpoint → actual inference worker | Official worker extension samples eight distinguishing embedding values; one bounded engineering generation follows |
| `probe_alternation_vllm.py`, `run_alternation_vllm_probe.sh` | E4 next exported checkpoint → actual inference worker | 222779 completed exit0; eight actual values match new/differ from old; one request returns2 tokens; no task-accuracy claim |
| `mh_phase.py`, `phase_vllm_worker.py`, `run_mh_phase.sh` | E4 updated checkpoint → E3 search → next E4 | Original search split across32-case GPU evaluation/login GLM proposal/GPU candidate evaluation; native summaries from two fixed shards, per-worker numeric weight proof |
| `experiment_randomization.py`, `controlled_training_bootstrap.py` | E3/E4 shared trial randomness | Process-start hash/Python/NumPy/Torch initialization; native data sampler and rollout engine seed, continuous unseeded training request stream; three Hydra configurations checked |
| `controlled_conditions.py` | E4 theta updates and E3 h search coordinates | Original loop starts from incoming h; full harness for WHALE/harness-only, prompt subspace for FST, no search for weight-only; native CPU fixtures pass |
| `probe_trial_randomization.py`, `vllm_randomization_worker.py` | E4 sampler-state measurement | 222792: actual seeds42/43/44 and changed CUDA RNG;24 toy requests/64 tokens; standalone vLLM, not full Ray training |
| `canonical_initialization.py`, `finalize_canonical_initialization.py` | Shared theta0=Q_BF16(theta_pretrained) | No learning; exact cast and complete active graph audit; corrupt-value/alias/graph fixture passes; 222794 generation-asset check failed;222797 exact723-tensor audit and inference-contract check pass;2791 FP32-origin elements round, no learning |
| `controlled_pilot.py`, `run_controlled_pilot.sh` | E4 weight-only experimental coordinate | 222801 completed two continuous batches from common theta0; preflight order deviation preserved; future plans use actual DataLoader |
| `audit_controlled_pilot.py` | E4 two-batch measurement | Complete222801 two-batch audit passes under the explicit schedule-deviation auditor; original expected-ID discrepancy remains disclosed |
| `scoped_proposer.py` | E3 native multi-candidate proposal boundary | Actual iteration/allocated names preserved and prior evidence protected; seed42 harness-only search completed, and the independent joint search is running after its own phase-one update/export |
| `preflight.py`, `budget.py`, `proposer_profiles.py` | Shared experimental infrastructure | Reproduction readiness, budget scenarios, provider settings; not scientific contributions |

The gate uses the **full valid archive before task-score Pareto pruning** and a
fixed phase-incoming reference. Identity includes checkpoint content, H/C data,
decoding and verifier hashes. Enabled selection accepts only optimization role C;
V/T/engineering receipts cannot enter selection. These checks catch incompatible
records; they cannot prove a serving endpoint actually loaded the claimed weights.
Fresh-worker value and generation verification222176 passed via the official numeric RPC extension; this does not inspect the original in-place NCCL receiver.

The current vLLM runtime check completed8 calls on one H100 (job221663), with all
8129-token budgets exhausted and no solved Chess trajectories. The independent
audit verifies input/output tokens and move decisions; it does not establish
scientific accuracy or produce an RSFT update. See the current
[status](../PROJECT_STATUS.md) and [engineering log](../EXPERIMENTS.md).
The released default4B also completed the same eight-input diagnostic (job221725),
with eight length stops and zero successful trajectories. The2B frozen launcher
must be replayed at commit`476baed`; the current optional-plan launcher is bound
by the4B plan. The original AgentLoop text-formatting helpers passed the4B input
token check. Subsequent native update222156 and canonical export222165 pass;
fresh-worker inference verification222176 passed.

`off` at the adapter delegates directly to the original selector without reading
audits. The regression test executes the pinned upstream selector bodies, including
early-stop tie semantics. This is **not** an end-to-end WHALE equivalence test.

## Reproduce the offline checks

From the project root, using the existing base Python with pytest and Pillow:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q ours/tests
python -m ours.preflight --output results/preflight-new.json
python -m ours.budget
python -m ours.chart_pairs --output data/engineering-new --count 8 --seed 17
python -m ours.proposer_profiles --profile glm_flash
```

Preflight deliberately exits 2 when readiness is incomplete. Its output is a
diagnostic, not a failed model benchmark. Output paths for preflight, selection
and chart generation must be new; existing evidence is not overwritten.

For saved, actual evaluations use `python -m ours.select INPUT.json --output
DECISION.json --mode paired`. The input schema is in `select.py`; a CLI selection
does not execute candidate code or a model. Unit-test outcomes are fixtures and
must never be included in result tables.

## Native training bootstrap

The [bootstrap audit](../results/chess-bootstrap-221898-summary.json) passed all64
training trajectories and found two complete successes on different puzzles.
These are not held-out scores or an improvement comparison. The next native
single-step job collects64 fresh trajectories; cached feasibility responses are
not silently substituted for online training. Job221906 failed during initialization
because CuPy was absent; job221907 then exposed an undersized transport buffer.
Initial weight sync then passed in221912, which was proactively stopped to
correct a communication deadline that included queue waiting. Its partial
generation counts remain unknown. Job221921 completed64 fresh trajectories and
accepted one, but its first AdamW foreach step failed with CUDA OOM. The native
scalar override passed CPU numerical and synthetic GPU memory checks. Job221967
uses the [v5 plan](../results/native-rsft-pilot-plan-20260909-v5.json), scalar AdamW
and the recording hook. [PROJECT_STATUS](../PROJECT_STATUS.md) records all costs
and limitations. Recovery222156 completed one real optimizer step on the fully
audited221967 batch, with no resampling. Canonical export222165 passed all active
parameter values; fresh vLLM handoff verification222176 passed, followed by explicit allocation
release because exit cleanup retained the worker processes.

The [execution plan](chess_reproduction_plan.md) explains the paper's initial
truncation behavior, the separate assistant and total-response limits, stopping
rules and subsequent weight-handoff checks. The actual training entrypoint only
accepts a frozen plan whose sources, model and full resolved configuration pass
`rsft_pilot_gate.py`. Historical plans must be checked against their archived
source snapshot when later fixes change source identities.

After a complete batch receipt exists, audit the recorded run **before changing
any source bound by its plan**:

```bash
PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD" \
  data/training-runtime-v1/bin/python -m ours.audit_native_training_batch \
  --directory data/native-rsft/pilot-221967/audit \
  --plan results/native-rsft-pilot-plan-20260909-v5.json \
  --output results/native-batch-audit-new.json
```

The command requires a complete batch and accounted requests. Its unit checks
and full live invocation pass for221967. Native imports may require host IPC access.
The audit cannot establish optimizer success or updated-checkpoint inference.

For future native Chess runs, `recorded_training_bootstrap.prepare_worker` adds
recording subclasses while preserving the native method arguments and returns.
It was **not enabled in job221921**; job221967 selects it explicitly. The CPU configuration check is:

```bash
bash ours/run_native_rsft.sh \
  ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=ours.recorded_training_bootstrap.prepare_worker \
  --cfg job --resolve
```

Overrides must precede Hydra option flags. A subsequent GPU run needs a new
frozen plan and reviewed launcher configuration. Seven recorder tests and CPU hook
checks and full distributed capture221967 pass. The chat journal records
starts before calls, preserving unknown usage on interruption. Batch files retain
actual padded tensors/masks and native per-turn events. Receipts certify file
completion, not optimization or performance. `summarize_requests` reports a
snapshot of recorded calls, not completion of a live job. These private records
contain reference metadata and require a separate redaction path before proposer
access; the token-ID `generate` interface is outside the request recorder's scope.

## Native search and disjoint pilot data

One logical native search completed across jobs221733/221743/221754. Both h0 and
the actual GLM h1 solved0/8, each using65032 output tokens. h1 was accepted on an
accuracy/call-count tie determined by filesystem traversal; this is not a gain.
The [independent audit](../results/native-search-221754-summary.json) verifies
token evidence, Chess verdicts and tied-optimum membership. The failed connection,
metadata import and initial audit assumption are preserved in the
[experiment log](../EXPERIMENTS.md). No RSFT update or F0–F5 pass is claimed.

`native_search.py` delegates the pinned `run_evolve`: fresh h0 evaluation, a
GLM-generated h1, native validation/evaluation, frontier and selection. The
candidate is not a prescribed edit. It uses the original full-harness skill,
one iteration/candidate, eight engineering inputs and seed42. The shared adapter
raises on failed/incomplete native sweeps because upstream otherwise records
evaluation exceptions without raising. It does not change successful scores.

The proposer receives a disposable copy of the answer-redacted native archive,
confined with Landlock. Only h1, pending metadata and its report can return;
changes to existing evidence or unexpected files abort import. The real key and
dataset remain outside that view. Native CLI logs and exact SSE responses are
retained; CLI USD estimates are not authoritative GLM charges. The persistent
45 CNY journal, 12 API requests, 12 CLI turns and 300-second proposer deadline
bound this first engineering search. No outer retry or fallback model is added.

Completed h0 evidence can be reused only after exact data/seed/sampling/weights
and archive-hash checks. A completed GLM proposal can likewise be imported without
another API call. The unambiguous metadata alias `slot: h1` is normalized to
`name: h1`; conflicting names, paths or parents are rejected. Original metadata
is archived and candidate code remains byte-identical. A restarted target service
is disclosed; this is not a controlled study of service-state numerical effects.

Use the frozen plan only with its matching source snapshot. A successful
check-only exit must precede submission:

```bash
export PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
data/training-runtime-v1/bin/python -m ours.native_search \
  --plan results/native-search-plan-20260909-v3.json --check-only &&
sbatch ours/run_native_search.sh results/native-search-plan-20260909-v3.json
```

`prepare_chess_splits.py` prepared128 train/32 MH-validation/64 test examples,
with no model calls or score-based filtering. Preparation and an independent
Python-chess pass verify disjoint IDs and position groups, excluding
all64 previously inspected engineering cases. The audit replays790 legal
solution plies. Dataset bytes, IDs, position groups, pin and source hashes are in
the [pilot manifest](../results/chess-pilot-manifest-20260909.json) and
[independent audit](../results/chess-pilot-audit-20260909.json). These are future
pilot inputs; they are not substituted into the eight-case search, original paper
splits or a completed F0 experiment.

## Real local visual input check

Job221552 completed 32 generations on one H800 in 44 allocated seconds
(0.01222 GPUh), using the cached Qwen2.5-VL-3B-Instruct revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`. All 16 image inputs had 154 visual
tokens with matching grids and pixel tensors; all 16 no-image controls had none.
Visual accuracy was 16/16 and paired accuracy 8/8; no-image accuracy was 8/16 and
paired accuracy 0/8. These eight easy pairs test input handling; their ceiling
performance cannot establish VETO effectiveness. This is not the original WHALE
model family, a proposer coding test, or an F0–F5 scientific pass.

The pre-run plan, raw generations, full result and independent rescore are in
`results/visual-smoke-*`. See [the experiment log](../EXPERIMENTS.md).
Source files used by the run remain unchanged and their hashes were verified.
At this milestone the offline suite had62 passing tests; later milestones add further checks.

On this cluster, inspect current resources before repeating the GPU run:

```bash
sbatch --test-only --partition=q-h800 --gres=gpu:h800:1 ours/run_visual_smoke.sh
sbatch --partition=q-h800 --gres=gpu:h800:1 ours/run_visual_smoke.sh
```

The script uses the existing `qwen-vl` environment and cached checkpoint, a
10-minute hard limit, ALL email notifications, no requeue and a new job-specific
output directory. CPU-only processor checking is available through
`python -m ours.run_visual_smoke --preflight-only --manifest MANIFEST --model
LOCAL_SNAPSHOT --output NEW_DIRECTORY` in that environment.

## Original Chess runner with a local target

The explicit `ours/compat` PYTHONPATH overlay supplies only `LLMConfig`,
`ChatMessage` and `LLMClient` from `ours.local_completion`. It does not install an
imitation upstream package globally. The local client rejects remote-provider
configs, binds requests to the actual checkpoint bytes, records output token IDs
and usage, and loads a fresh Qwen3.5 model instance. It is our compatibility
implementation; the missing original client remains unavailable.

Job221577 ran the unmodified `runner.evaluate_harness` and original base harness
against8 real Lichess engineering positions, with Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. One H800,25 allocated seconds,
8 generations,98 output tokens, no truncation. All moves were legal and none
matched the first solution move. Greedy decoding, thinking disabled and reduced
token limits make this an input/runtime check, not an original-paper score.
All cases terminated on the first wrong move: real multi-turn inference and
success-filtered training have not been validated.

The preparation fixes a reproduced Arrow scanner shutdown error using synchronous
row-group reads. All64 source records are identical to the earlier attempts;
the original converter/verifier and sample selection are unchanged. The manifest
lists engineering IDs to exclude from future scientific splits. This exclusion
still needs enforcement when those splits are constructed.

Prerequisites on this machine are the existing qwen-vl environment, the pinned
checkpoint in `data/models/`, and isolated chess1.11.2 in `data/runtime-deps/`.
The package installation report and model SHA verification are in `results/`.
The job script sets the overlay only in its child process and requests ALL email
notifications and no requeue. Before another submission, refresh queue/resources
and require a successful plan check:

```bash
python -m ours.run_chess_smoke --check-only --plan results/chess-smoke-plan-20260909.json &&
sbatch ours/run_chess_smoke.sh
```

This historical command requires the source snapshot at `60b836c` or `a542169`.
The current driver supports fixed shards and intentionally fails the old source
hash check. The plan freezes runtime sources and inputs, so changing them requires a new plan
and matching job command. The failed job221576 is preserved: an earlier orchestration
step submitted despite a failed CPU assertion, before a plan existed. It made no
model calls. The documented command now gates submission on the precheck's exit
status. See [the complete engineering log](../EXPERIMENTS.md).

## Official domestic GLM proposer

The user authorized GLM-5.3-Flash with a45 CNY first-round project budget. The key
is stored outside the repository in a0600 private file. `proposer_profiles` also
supports `--profile glm_flash_cn`; configuration-only output never prints a key.
`glm_gateway.py` reserves the full published context plus requested output before
each call and settles complete provider usage. It counts cached tokens at the
same conservative input rate. Rates1/4 CNY per million input/output are budgeting
assumptions, not the provider invoice; unknown outcomes retain reservations.
The journal under `data/glm-budget/` persists across processes and must not be
deleted/reset between searches. The45 CNY limit covers this project's requests,
not other users of the account. No fallback model or paid server tools are enabled.

`run_proposer_probe.py` exercised the original wrapper with actual Read/Edit tools
on a verified public harness copy. `confined_exec.py` enforces Landlock filesystem
restrictions; ABI1 does not provide network or metadata-operation confinement.
The relay owns the real key, while the child gets an ephemeral loopback token.
Read→Edit→DONE passed on attempt5 after correcting HTTP response delimitation.
The complete upstream SSE body is preserved but buffered to set Content-Length;
this affects streaming latency, not model text. Earlier timeouts remain archived.
The offline protocol replay is a fixture with0 external calls, regardless of the
native CLI's displayed estimated dollar cost. See [the API/tool evidence](../results/glm-proposer-tool-probe-20260909-attempt5.json).
This is a specified-edit transport probe. A formal proposer search has not run.

## Full decoding diagnostic and candidate training runtime

Job221597 retained all8 engineering inputs across two fixed4-example H100 shards.
All8 outputs hit8129 tokens; total65032 tokens,0 solved. An independent audit
verified coverage, raw token decoding, source/checkpoint/input hashes and each
native step. It rejects multi-call runs because this diagnosis does not validate
multi-turn trajectories. No successful samples are available from this run.
See [the audit](../results/chess-decode-221597-summary.json).

The new `data/training-runtime-v1` environment is separate from both existing
environments. `training_requirements.txt` declares the NumPy2 compatibility change;
`training_resolved.lock` fixes the original237 resolved packages by URL/hash.
Install `training_checkpoint_requirements.lock` afterward with `--require-hashes
--no-deps` for the original NCCL engine's CuPy14.2.0 dependency. This additive
wheel leaves the previous dependency versions unchanged; pip check and explicit
original NCCL import pass. `training_bootstrap.restore_bundled_tools` loads only the missing
`verl.tools` namespace from the same pinned WHALE Math directory. Chess trainer,
actor and rollout still come from Chess; their actual imports passed after this
declared repair. Native actor initialization and one real GPU optimizer step now pass in222156.
The original post-update NCCL call returned, but its receiver values were not
independently sampled. Canonical export/full parameter audit pass in222165;
fresh vLLM worker-value and request verification222176 passed.
All of this belongs to shared implementation details, outside VETO's E1–E3.

## Remaining integration work

Restore the next MH-to-RSFT alternation after the successful fresh-server
weight check, then connect the shared visual task adapter and all four
acceptance call sites to the live search/training loop. Extend per-request logging
to that loop and connect the validated proposer transport to formal search. The completed frozen-model
input check does not recover the upstream baseline or demonstrate visual failure
under alternation. F0–F5 remain required before any scientific conclusions.

Ordinary accuracy gates and counterfactual training have prior art. E2–E3 are an
application of empirical constrained selection, not a new optimization theorem.
See the [thesis](../notes/thesis.md) and [novelty screen](../notes/novelty_screen.md).


### 恢复完整采样后中断的E4步骤

`run_native_rsft_recovery.sh`绑定`results/native-rsft-recovery-plan-20260909-v3.json`，
从同一base恢复221967的64行已审计批次，禁止新模型调用且只消费一次。
`RecoveredDisaggregatedTrainer`沿用原筛选/actor/optimizer/checkpoint/权重同步；
`compact_batch`仅裁全无效响应尾部，保留全部提示与有效token/位置/观测/样本。
Method对应E4：`L = -sum(m_t log p_theta(y_t | x,y_<t)) / sum(m_t)`，分子分母不变。
该适配属于所有对比条件共享的执行部分。对混合循环注意力不假设左补齐不变性；
当前路径不裁左侧，CPU原actor等价测试不能取代真实4B GPU更新验证。


`verify_native_transition.py`实现E4的实际参数变化测量：以活动参数的
`Delta_theta = theta_after - theta_before`计算变化元素数与L2范数，严格限制MTP/共享权重映射。
FP32训练变化和BF16导出变化分别报告，并验证导出逐值等于原生参数的BF16转换。
`run_native_checkpoint_export.sh SOURCE_JOB RECOVERY_PLAN.json`在CPU上调用原merger，
然后运行该核验。真实4B checkpoint已在222165通过全部723个独立活动张量核验。
`canonical_native_export.py`继承原merger，以`save_original_format=False`保留视觉参数名，
并沿用经审查的base推理配置，防止训练EOS/pad配置进入性能对比。

`probe_updated_vllm.py`对应E4架构图的checkpoint→推理worker连接：对预先从磁盘确定的
8个BF16 embedding坐标，验证`theta_worker[j] = theta_export[j] != theta_base[j]`，
然后执行一条最多16token的工程请求。222176实测8坐标一致、返回2token；
输出`Yes`而非要求的`ready`，不记指令遵循成功。此证明与原NCCL接收端指纹分开。
结果完成后清理进程未退出，主动释放分配，Slurm记CANCELLED；不是正常结束的作业。
源码随后补`managed_inference_engine`显式关闭（10秒期限），尚未另外GPU验证清理行为；
不重复已通过的请求。v2计划回放需使用222176归档源码，后续运行须冻结新计划。
导出中原始FP32的3840个元素发生精度转换，必须为正式baseline统一精度或加入cast-only
control；该检查和参数差异不能解释为准确率提升。

## 新checkpoint上的MH阶段

`mh_phase.py --plan PLAN --phase baseline|propose|candidate`保留原搜索循环，GPU阶段使用
`run_mh_phase.sh PLAN baseline|candidate`；propose只在登录节点运行，要求完整baseline审计。
32题MH按固定ID分片，私有顺序记录用于合并，原匿名结果保持；合并后原统计目标仍为
`A(h;theta)=sum_i success_i/32`，原frontier其次比较policy calls，固定读入次序处理普通平局。
这是E4到E3的共享执行连接，不新增方法公式。原early-stop行为保留。
每个服务worker继承原Worker，仅在load后核验8个训练变化坐标；图中对应checkpoint箭头。
训练/test数据不进入proposer视图，原h0评分缓存仅在同一冻结阶段复用。
当前3项原生CPU集成检查通过，真实32题MH基线222196正常完成，两worker权重核验通过。
`audit_mh_phase.py --plan PLAN --harness h0|h1`在整批完成后检查精确回复重放、独立棋盘
续着及全部请求计量；h0实际审计PASS，结果0/32、35次调用、260128输出token。
GLM提案在用户明确授权后完成，h1原样交给222212做同32题评测；科学门保持原要求。

`compact_recorded_training_bootstrap.prepare_worker`用于下一阶段的新在线采样，继承记录器
并在原actor分发前使用已有response-tail裁剪；不安装缓存恢复manager。Method仍为E4的
原成功筛选SFT损失。`RSFT_HARNESS_PATH`将E3选中的源码路径送入原生Ray环境；临时h0
完整CPU配置验证通过，正式训练计划必须等待完整搜索/独立审计并绑定最终选择。

`alternation_training.py --prepare --search-plan SEARCH_PLAN --data-report DATA_REPORT --plan PLAN`
在完整MH及双方审计后冻结下一批数据/模型/harness/配置，默认检查而不提交；
`sbatch ours/run_alternation_rsft.sh PLAN`才启动下一次原生训练。实际配置逐字段比对，
目标选择另外与原frontier/early-stop比较。工程游标推进至下一8题；HF初始化不声称
恢复了之前的optimizer/scheduler/RNG/sampler状态。Method对应E3→E4的harness箭头，
原SFT公式不变，空成功集不能被记录为已完成优化器更新。

## 完整搜索的离线计算节点调度

`native_loader_schedule`通过原生DataLoader生命周期推导训练题序。旧预检只遍历
RandomSampler，遗漏初始化/reset推进随机流；222801的原ID列表保留为偏差证据。
`audit_controlled_schedule_deviation`在完整原轨迹审计中采用真实DataLoader推导的期望
矩阵，并输出带偏差的独立顶层状态；不修改计划、数据或生成轨迹。
`controlled_training_result`核对Slurm终态、两批日志与独立审计并封存checkpoint哈希。

`controlled_data_order`通过真实RLHFDataset、原生collate_fn及RayPPOTrainer DataLoader
生成前两批，交叉核验独立索引探针的题目ID和checkpoint sampler状态，并恢复父进程
Python/NumPy/Torch CPU随机状态。`controlled_pilot_continuation`复用原两批配置和启动
逻辑，为尚未开始的seed43/44冻结这些真实题序及TorchData版本/源码哈希。
Method对应E4训练数据抽样与复现设置，原success-filter SFT公式不变，不引入新损失。
seed42原计划/偏差保留，新入口不能替换它；完整轨迹审计仍用原audit_controlled_pilot。

```bash
python -m ours.controlled_pilot_continuation --phase prepare --seed 43 --plan NEW_PLAN
python -m ours.controlled_pilot_continuation --phase check --plan NEW_PLAN
# 等候当前搜索释放提交名额；运行前再次核对活跃作业。
sbatch ours/run_controlled_pilot_continuation.sh NEW_PLAN
```

`controlled_checkpoint_export`绑定上述受控训练终态、全部批次证据、原生checkpoint及
共同theta0，用原FSDP合并器导出规范HF参数名。Method对应E4到评测模型的保存/交接箭头：
对每个活动参数验证`theta_export[i] = BF16(theta_native[i])`，分别报告
`||theta_native - theta0||_2`及`||theta_export - theta0||_2`，不要求非零变化。
原生model/extra/data.pt保留，不将HF导出冒充训练状态恢复。若Transformers保存时自动
新增generation_config，仅接受已审查的两个诊断输出默认值差异，归档文件后恢复源资产
缺省状态；其他解码差异拒绝。随后完整核对配置、词表、控制token和prompt token。
seed42受控导出223620已完成723活动张量的真实全参数转换检查；最终结果保留训练题序预检偏差，不声明性能收益。

```bash
python -m ours.controlled_checkpoint_export --phase prepare --plan NEW_PLAN --training-plan TRAINING_PLAN --result COMPLETED_RESULT --step 2
python -m ours.controlled_checkpoint_export --phase check --plan NEW_PLAN
# 先确认完整搜索控制器已释放唯一提交名额，再提交CPU导出分配。
sbatch ours/run_controlled_checkpoint_export.sh NEW_PLAN
```

`native_phase_resume.phase_overrides`对应E3→E4原状态恢复箭头：累计target1/2，恢复
model/extra及data.pt，保留原Adam不恢复的行为。原loader对缺失data.pt只警告，此处
提前拒绝，避免重新从第一批开始。CPU恢复探针已用真实数据及保存状态验证下一批，
actor RPC只记录，GPU加载需后续联合试验。不得将这些基础设施称为新的优化目标。

`joint_training_phase1`与`audit_joint_training_phase1`实现联合条件的第一批启动及完整
64轨迹审计。Method对应`theta1 = RSFT(theta0, h0; B1)`，然后保存原生状态交给E3搜索；
累计target=1、online iterations=0，model/extra/data.pt保留，Adam不保存。它不把
weight-only的已采样前缀换名，也不声明第二阶段GPU恢复已完成。WHALE与FST只在后续
MH搜索空间不同，这里的第一批都使用原h0、共同theta0和同一原训练算法。

```bash
python -m ours.joint_training_phase1 --phase prepare --condition whale --seed 42 --plan NEW_PLAN
python -m ours.joint_training_phase1 --phase check --plan NEW_PLAN
# 待完整搜索释放唯一提交名额后启动；每个计划只能开始一个原始试验。
sbatch ours/run_joint_training_phase1.sh NEW_PLAN
python -m ours.audit_joint_training_phase1 --plan NEW_PLAN --directory NATIVE_RUN/audit --output NEW_AUDIT
```

原生循环fixture验证0→1与已处于1时的1→2边界只生成一批；空accepted仍保存/同步。
第二种fixture并未调用checkpoint loader。新第一阶段审计器另用历史64条轨迹/75次请求
完整重放通过，复制fixture的来源元数据被显式改写，不能算作新联合条件实验。

`joint_training_result.py`封存独立WHALE/FST第一阶段：完整64行与请求记账、实际
SFT更新数、Slurm终态、单一global_step_1、model/extra/data.pt及实际预检sampler状态。
每非空批调度器仅推进一次，空更新仍可完成。`joint_checkpoint_export.py`只接受该
联合来源，复核全部完成证据后才生成冻结导出计划；保留原生恢复文件，对每个活动参数
验证`theta_export[i] = BF16(theta_native[i])`，分别报告原生及BF16变化，包含不变结果。
Method对应原E4权重更新到E3搜索的交接。实际worker加载和第二阶段GPU恢复另行验证。

```bash
# 在原生环境、仓库根目录运行；需要真实完成的联合phase1，不能使用weight-only结果。
python -m ours.joint_training_result --plan JOINT_PLAN --audit FULL_BATCH_AUDIT --slurm TERMINAL_RECORD --log TRAIN_LOG --output NEW_TRAIN_RESULT
python -m ours.joint_checkpoint_export --phase prepare --plan NEW_EXPORT_PLAN --training-plan JOINT_PLAN --result NEW_TRAIN_RESULT
python -m ours.joint_checkpoint_export --phase check --plan NEW_EXPORT_PLAN
# 队列名额释放且无搜索控制器争用时才提交；准备计划不调用模型。
sbatch ours/run_joint_checkpoint_export.sh NEW_EXPORT_PLAN
python -m ours.joint_checkpoint_export --phase close --plan NEW_EXPORT_PLAN --slurm EXPORT_TERMINAL_RECORD --output NEW_EXPORT_RESULT
```

3项原生完成/来源绑定测试和2项导出测试通过；后者真实执行原生merger子进程，检查
小VLM全部34张量与42576元素。模型与Slurm来源是合成fixture，没有真实联合训练或
导出作业；缺失image processor警告保留，此测试不覆盖视觉输入。所有历史原始数据
保持原样；最终v4记录包含64行覆盖要求，临时fixture路径修正前的失败日志单独保留。

`staged_search.py`在登录节点保留原生5轮×3候选循环，通过`staged_evaluation.py`
在独立Slurm作业完成每个候选的两个固定ID分片。`audit_staged_evaluation.py`要求
完整32题、全部实际请求/token、原生回复重放及独立棋盘验证；控制器再确认Slurm
COMPLETED/exit0，才复制原生匿名结果并继续原候选选择。名额未释放时只查询队列。

Method对应E3的`h_next = MH(theta_fixed, h_incoming)`执行箭头，不新增损失、目标或
VETO机制。API候选视图仍由`scoped_proposer`隔离，所有题目均属于已授权MH32；权重
训练数据和留出测试内容不进入视图。完整真实多轮搜索尚待运行。

```bash
# 在现有native环境及ours/compat PYTHONPATH下；prepare/check不调用模型或GLM。
python -m ours.staged_search --phase check --plan results/controlled-harness-only-seed42-plan-20260910-v1.json
python -u -m ours.staged_search --phase coordinate --submit-evaluations --plan results/controlled-harness-only-seed42-plan-20260910-v1.json
```

同一试验目录只允许首次启动；先核对`state.json`中的PID和实际进程，不能因为等待
就再次执行coordinate。`--submit-evaluations`会按固定资源自动逐次提交，等待账号
已有作业结束；省略该开关则需要外部按request.json手工提交评测。每次最多40分钟/
2H800，最多16次；预算、源码、数据和共同模型绑定冻结计划。当前计划生成器仅接受
harness-only共同初始化；联合条件必须使用下述独立来源验证入口。

`joint_staged_search.py`继承原StagedSearch，只覆写Slurm/worker入口和提案格式适配。
正式计划需要同条件/seed的独立第一阶段训练、规范导出及Slurm终态；它验证完整来源后
把导出的theta1作为实际target，保存原生checkpoint给后续E4恢复。原计划生成器仅提供
共同MH32/5×3/推理/预算设置，不提供联合权重资格或可复用分数；联合资格单独核验。
Method对应`h1 = MH(theta1, h0)`。FST使用同一输入、预算和权重，保留原提示词子空间。
控制器及评测主进程核验完整文件；worker重验轻量来源并抽查实际加载的8个区分坐标，
不把该抽查称为GPU全参数证明。评测器继续重放全部32题请求/token并独立检查棋盘。

```bash
# 必须已有真实完成的joint export；不会接受weight-only或工程替代品。
python -m ours.joint_staged_search --phase prepare --plan NEW_JOINT_SEARCH_PLAN --root NEW_JOINT_ROOT --export-plan JOINT_EXPORT_PLAN --export-result COMPLETED_JOINT_EXPORT
python -m ours.joint_staged_search --phase check --plan NEW_JOINT_SEARCH_PLAN
python -u -m ours.joint_staged_search --phase coordinate --submit-evaluations --plan NEW_JOINT_SEARCH_PLAN
```

三项原生测试覆盖两条件五轮选择与FST拒绝、满分停止、两个worker模块与CUDA分配路由、
旧32题/34次回复完整重放和来源/配置拒绝。等价测试评分及来源终态是合成fixture，
旧回复属工程档案，不是联合对照的新数据。第一版fixture误向原propose_claude索取
next_names参数而失败；仅改用原任务提示和原生槽分配，v2通过，失败日志保留。
正式whale/seed42联合搜索由独立223622训练及223757导出开始；第一轮h0–h3完整MH32分别解出1、1、5、3题，原生选择器保留h2。第二轮h4/h5均已核验6/32，控制器1413742正在评测h6作业223909；完整五轮搜索和GPU第二阶段恢复尚未完成。h2固定回复在h0 parser下31/32动作不同，但跨运行32个请求/输入token相同而仅3个输出序列相同，不能把总分差全部归因于parser。原harness-only控制器已完成退出；两个独立搜索的同名候选不是同一代码。

6项原生集成检查涵盖选择历史一致性、错误边界、两分片合并与已有真实轨迹审计，
不生成新回复。这里的真实旧h1分数6/32不能复用于新theta0条件。

`staged_search_recovery.py`是首轮报告路径中断的显式恢复入口。原作者允许
`logs/iteration_*/report.md`，本地原检查只接受三位补零路径；新入口只规范当前轮
未补零报告，归档原字节，再执行原完整性检查。它冻结原失败现场与新入口哈希，
确认旧控制器退出及只有完整h0，然后在原`run_evolve`内重用已审计h0及原首轮提案。
后续付费边界、候选校验、评测与原选择逻辑不变。不会自动恢复其他中断。
Method对应E3执行/复现设置，不构成新的优化公式。三项原生测试覆盖五轮历史、
满分提前停止、h0/首轮不重复生成，以及篡改/额外文件拒绝；评分均为合成数据。

恢复前用`--phase prepare --plan ORIGINAL --amendment NEW`冻结修订；
`--phase check --amendment NEW`只做校验；
`--phase coordinate --amendment NEW --submit-evaluations`执行经核验的恢复。
同一修订只启动一次，不能对已恢复目录重复运行。

`staged_metadata_recovery.py`只处理已完成首轮、第二轮清单解析失败的现场。
独立修订将裸列表包成candidates、把一致的name/slot/harness/id映射为name，保留
原字段和全部候选字节。它要求旧进程退出、四个评测审计、首轮选择完整及第二提案
已完成；从原生start_iteration=2、iterations=4接续，不重复已完成的首轮。
原提案、失败现场及第一恢复档案保持不变；原生perfect停止保留，原计划的patience
停止关闭。后续付费提案同样采用这一有限格式兼容，冲突/越界/保护文件改动仍拒绝。
使用与首轮恢复相同的prepare/check/coordinate参数接口，必须使用新的独立修订路径。
三项原生测试比较两次中断与连续运行的完整五轮、满分提前停止及演化记录；评分
均为合成数据，零API调用。Method对应E3执行边界，不是新的研究机制或优化目标。

`recorded_parser_comparison`对完成审计的原始回复逐条比较incoming h0和候选parser，
记录动作/合法性分歧、thinking结束和截断情况；源码与私有日志哈希绑定报告。
它对应E3中prompt、parser等组件的诊断，不会把固定回复分析当作重新生成的受控消融。
222891的32条回复新旧动作/合法性完全一致，8个成功中6个正常结束thinking、2个截断。

## WHALE-FST的提示词搜索空间

`prompt_subspace(native)`供将来的独立FST阶段围绕原`run_evolve`使用。它装配公开完整
Chess任务说明与`prompts/chess_prompt_subspace.md`的重建限制，并启用原prompt-only
AST校验。额外在loader执行前要求两个提示词为纯字符串，避免被屏蔽的赋值表达式
改变程序或随机状态；完整自然语言文本仍可搜索。该额外检查和重建说明均明确归属本项目。

Method对应`H_prompt = {h: nonprompt_AST(h)=nonprompt_AST(h0), prompt RHS为字符串}`，
以此限制原E3候选空间，E4/数据/预算保持共同设置。4项原生CPU检查覆盖原wrapper
收到完整约束、原循环接受纯提示词/拒绝parser或预算等变化、执行前拒绝副作用。
此组件未接入222212，未完成真实FST搜索/训练，也不是独立FST论文的完整复现。


## 联合搜索结束与第二阶段恢复

`search_completion.verify_completed_search`只接受完整五轮或原生满分提前停止的结束档案。
它逐轮重算候选校验/FST拒绝、summary、Pareto、选择与停止，核对每个私有审计和公开
文件哈希；临时目录承接原frontier写入。保留已冻结共享上下文的排序与并列顺序。
`joint_training_phase2`将该证明与同条件/同seed的第一阶段训练/导出来源绑定，使用选中
harness以及原生global_step_1/model/extra/data.pt，累计target=2且只新增第二批64条轨迹。

Method对应`h_star = MH(theta1; H)`之后的`theta2 = RSFT(restore(C1), h_star; B2)`。
其中C1保存模型、scheduler、RNG和sampler；按原作者约定不保存/恢复Adam moments。
`native_resume_observation`对应这条恢复箭头上的观测点：原loader加载后，在actor仍位于
GPU时完整比较FP32状态与保存值，并核对scheduler和所有RNG；driver只读取pending
loader状态，不调用会创建iterator的state_dict()。CPU测试模式不会冒充GPU证明。
原生fit在加载后同步权重再生成，观察器不替代原NCCL同步，也不声称逐值验证vLLM接收端。

```bash
python -m ours.search_completion --plan COMPLETED_JOINT_SEARCH_PLAN --output NEW_COMPLETION_PROOF
python -m ours.joint_training_phase2 --phase prepare --plan NEW_PHASE2_PLAN --search-plan COMPLETED_JOINT_SEARCH_PLAN --search-completion NEW_COMPLETION_PROOF
python -m ours.joint_training_phase2 --phase check --plan NEW_PHASE2_PLAN
```

第二阶段的`audit_joint_training_phase2`先核验实际恢复记录，再完整重放64条回复/token/mask与
棋盘reward。`joint_training_phase2_result`要求Slurm COMPLETED/exit0、唯一global_step_2、
正确B2 sampler和恢复后的scheduler偏移；分别报告本批与累计更新，允许空批保存。
`joint_checkpoint_export`支持两阶段，Method对应`theta_eval = BF16(theta_native)`；
逐值检查转换并保留原生model/extra/data.pt，零变化与BF16舍入后不变均是有效输出。

```bash
python -m ours.audit_joint_training_phase2 --plan PHASE2_PLAN --directory TRAJECTORY_DIRECTORY --output NEW_BATCH_AUDIT
python -m ours.joint_training_phase2_result --plan PHASE2_PLAN --audit NEW_BATCH_AUDIT --slurm TERMINAL_RECORD --log TRAINING_LOG --output NEW_RESULT
python -m ours.joint_checkpoint_export --phase prepare --plan NEW_EXPORT_PLAN --training-plan PHASE2_PLAN --result NEW_RESULT
```

12项原生CPU集成通过，包括旧B2的64条明确标记的重放fixture、空/非空scheduler计数与
两阶段实际小VLM导出。合成GPU/Slurm记录不构成真实联合实验；小导出fixture无image processor。
正式whale/seed42的phase1及导出已完成，自己的联合搜索尚在第一轮；只有该搜索完整结束并
核验后才能准备合法phase2计划。真实GPU恢复、第二批执行及留出集结果尚未完成。

第三轮已付费提案的显式恢复入口为`staged_candidate_recovery`，接口与前两次相同。
它保留原七个评测与前两轮历史，只为一致的`candidate`别名增加映射；从原iteration3
继续，重放原第三提案时0次API调用。独立修订记录所有来源和改动，不自动重启其他失败。
未来尚未准备的联合搜索预先采用该别名兼容；当前共同theta0搜索的原冻结源不改。

## 最终留出评测协议

[controlled_heldout_protocol.md](controlled_heldout_protocol.md)固定四条件、三种子、64题、
原生解码与评分及四个逻辑分片。Method对应训练/搜索后固定模型与harness的最终测量
`A[c,s] = mean_i exact_solved(theta[c,s], h[c,s], x_test[i])`；不将test用于候选选择。
[冻结记录](../results/controlled-heldout-protocol-20260910-v1.json)只基于既有split ID及不透明
文件SHA，未加载test任务或答案。它不是执行计划；全部trial身份/状态封存、来源核验和
最终评测入口现已实现并通过CPU检查，正式trial矩阵尚未封存。相同逻辑分片支持4/2/1卡的预声明资源配置，不改变评分规则。

[原生语义说明](native_chess_semantics.md)记录两个需保留的细节：原frontier会保留同调用数
的低分候选，但最高分优先接受不变；retry环境值是默认，选中harness属性继续优先。
相关只读核验不修改当前搜索或既有协议，也没有加载test任务。

## 连续weight-only的多种子完成记录

`controlled_continuation_result`处理未训练的seed43/44计划，复用原完整128条轨迹审计，
再核验两批的实际sampler/scheduler、空批计数、启动配置及Slurm终态。Method是连续的
`theta1 = RSFT(theta0,h0;B1)`、`theta2 = RSFT(theta1,h0;B2)`，不插入恢复或复用联合前缀。
`controlled_continuation_export`在单进程上下文复用原生规范导出及逐参数比较，保留独立的
weight_only条件、seed、kind和回执目录；默认joint行为及异常后的上下文恢复均有回归。

在原生运行环境中，实际完成训练并获得审计/Slurm终态后使用：

```bash
python -m ours.controlled_continuation_result --plan TRAINING_PLAN --audit COMPLETE_BATCH_AUDIT --slurm TRAINING_TERMINAL --log TRAINING_LOG --output NEW_TRAINING_RESULT
python -m ours.controlled_continuation_export --phase prepare --plan NEW_EXPORT_PLAN --training-plan TRAINING_PLAN --result NEW_TRAINING_RESULT
python -m ours.controlled_continuation_export --phase check --plan NEW_EXPORT_PLAN
python -m ours.controlled_continuation_export --phase close --plan NEW_EXPORT_PLAN --slurm EXPORT_TERMINAL --output NEW_EXPORT_RESULT
```

原seed42使用保留的`run_controlled_checkpoint_export.sh`；其真实导出结束后，由新只读入口
封存终态，保持原题序偏差字段不变：

```bash
python -m ours.controlled_checkpoint_export_result --plan FROZEN_SEED42_EXPORT_PLAN --slurm EXPORT_TERMINAL --output NEW_EXPORT_RESULT
```

11项原生CPU检查通过；旧指标/state及合成来源仅作fixture，未执行真实seed43/44训练或
任何新的导出作业。正式trial矩阵与实际多种子结果仍待完成；64题留出执行器已通过下面的CPU检查。

## 最终身份封存、执行与汇总

`heldout_cohort`要求输入恰好四条件×三种子，每项明确COMPLETE或INCOMPLETE。
COMPLETE harness_only提供`search_plan`；其余提供`export_plan`、`export_result`，重新调用
各自原生完成检查。INCOMPLETE只提供`reason`及`evidence_paths`，没有模型或分数。
所有12项来源先固定，才可为完整条目准备最终评测；准备过程仅核验test文件不透明SHA。

```bash
python -m ours.heldout_cohort --input TWELVE_TRIAL_INPUTS --output NEW_SEALED_COHORT
python -m ours.controlled_heldout --phase prepare --plan NEW_HELDOUT_PLAN --cohort SEALED_COHORT --condition harness_only --seed 42 --root NEW_ATTEMPT_DIRECTORY --gpus 4
python -m ours.controlled_heldout --phase check --plan HELDOUT_PLAN
sbatch ours/run_controlled_heldout.sh HELDOUT_PLAN
python -m ours.controlled_heldout --phase close --plan HELDOUT_PLAN --slurm TERMINAL_CONTROLLER_RECORD --output NEW_FINAL_RESULT
python -m ours.heldout_comparison --cohort SEALED_COHORT --index TWELVE_EVALUATION_OUTCOMES --output NEW_COMPARISON_DIRECTORY
```

这些是原生运行环境中的模板命令，不是已提交作业。默认4H800/32CPU/256G/40分钟；
若预先选2卡计划，提交须同时覆盖`--gres=gpu:h800:2 --cpus-per-task=16 --mem=128G --time=01:20:00`；
1卡则覆盖1H800/8CPU/64G/02:40:00。固定四逻辑分片各起新进程，失败保留现场，后续波不启动。
实际分配、启动身份、分波与完整回复/worker来源在终态一并核验，不能覆盖已有attempt。

汇总输入对象有`evaluations`与`allocation_registry`。前者恰好12项：完整项为
`condition/seed/status=COMPLETE/result`；缺失项为`condition/seed/status=INCOMPLETE/reason/evidence_paths`。
后者每项为`condition/seed/stage/slurm_terminal`，stage取training/search/export/heldout；
成功留出成本从完整结果自动加入，同一job不重复计费，冲突归属拒绝，失败终态也计入。
阶段成本仅是已列记录小计，未列来源为null，不能冒称完整总成本；API与共享初始化成本另报。

JSON保留完整来源、逐种子结果、配对差值与成本；CSV保留12行；Markdown展示三种子和
mean/sample SD。缺种子时均值与SD为空。Method仍为最终测量`A[c,s]`，配对描述量是
`D[c,b,s]=A[c,s]-A[b,s]`，没有显著性、VETO因果贡献或90%归因结论。
6项原生CPU测试及2项汇总测试通过；fixture将旧32个MH题各复制两次验证64个位置，
没有打开test任务、生成新回复或创建正式cohort/计划。1/2/4卡调度测试用stub进程，
实际GPU上的留出运行仍待完成；推理有token记录，没有RSFT loss mask。

## 尚未运行的harness-only独立种子

`independent_harness_search`专用于seed43/44，继承完整原生循环与两分片评测，在计划中
预先固定已经检查的元数据别名/报告路径兼容。候选代码、原选择器、五轮/每轮三候选及
GLM预算不变。Method仍为`h_star[s] = MH(theta0,h0;M_H,s)`；这是提案接口接线。

```bash
python -m ours.independent_harness_search --phase prepare --seed 43 --plan NEW_PLAN --root NEW_SEARCH_ROOT
python -m ours.independent_harness_search --phase check --plan PLAN
python -m ours.independent_harness_search --phase coordinate --plan PLAN --submit-evaluations
```

这些命令须在原生环境及网络登录节点执行；当前实际seed43/44计划已生成，不能覆盖或
在现有控制器占用提交名额时启动。两项原生测试用合成分数/Slurm跑过两种子完整五轮和
满分停止，0实际模型/API调用。WHALE/FST另外四份seed43/44第一批计划也已通过真实
DataLoader/Hydra CPU预检；六份新增计划及完整检查见remaining-trial-plans-implementation-check。

`finish_seed42_search_then_export`的一次性排程PID1330451已退出；它在完整搜索结束后提交223620，导出现已完成。不要重复启动。它要求原搜索完整证明、旧导出plan/模型检查及空闲名额，才提交一次
`run_controlled_checkpoint_export.sh`。若状态为SUBMITTING或STOPPED_FOR_INSPECTION，
先查真实squeue和日志，不能推断没有提交而重试。排程不改变四条件的模型/harness依赖。
当前空间可继续下一阶段；完整矩阵的存储估算与缺口见storage-projection记录。

## Shared visual evidence through the native dataset

`VisualEvidenceDataset` subclasses the released `RLHFDataset`. The original
prompt override can replace image-bearing user content with plain text, and the
original builder pops image payloads before the length filter reads them. The
adapter copies rows before the original builder, restores the single image after
a text override, and binds the captured prompt environment to dataset-cache
identity. It rejects changed prompt environments after construction; instantiate
a new dataset for each harness. No-image controls use the same class. This is a
shared input contract for every visual condition, outside VETO E1–E3.

Method correspondence: preserve `o=(I,q)` in `p_theta(a | o,h)` and compute prompt
length after the processor expands image tokens. The existing E4 response mask
and success-filtered objective are unchanged. The adapter currently accepts one
visual user question, at most one bytes-only image, and no video. It removes only
duplicate encoded image metadata after native row construction; image content
remains in `raw_prompt`.

A future visual data configuration selects `custom_cls.path=pkg://ours.visual_evidence_dataset`
and `custom_cls.name=VisualEvidenceDataset`. Use this only in a separately frozen
visual plan; the current Chess plans retain their original dataset. The shared
single-turn binary reward now passes the native integration check below. The full
executable visual harness is now checked below; actual actor update and acceptance/search orchestration remain pending.

```bash
PYTHONPATH="$PWD/data/visual-runtime-overlay-v1:$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
HF_DATASETS_CACHE="$PWD/data/visual-native-interface-probe-20260910-v1/hf-datasets-cache" \
data/training-runtime-v1/bin/python -m unittest -v ours.tests.test_visual_evidence_dataset
```

The opt-in overlay contains pinned `qwen-vl-utils==0.0.14` and `av==18.1.0`;
wheel digests and installed files are recorded in the visual-runtime-overlay
report. Existing Chess runtime packages were not changed. The failed missing-
package probe is retained before the successful installation.

Two native CPU tests pass on one existing engineering image pair:120 image tokens
per image, correct length filtering and prompt-cache identities, no-image control,
pixel-bearing SingleTurn requests, native mRoPE and response-mask collation into
DataProto. Model replies and rewards are synthetic fixtures; no model weights are
loaded and no visual task accuracy or actor forward/backward is measured. Eight
additional comparisons show common/223620-exported processors produce identical
visual/no-image inputs and mRoPE with thinking on/off. Full evidence:
[implementation record](../results/visual-native-interface-implementation-check-20260910.json).

## Shared visual reward and native success selection

`visual_evidence_reward.compute_score` uses the existing strict A/B verifier for
all visual conditions. Method correspondence is E4: `r_i=v(answer_i,y_i)` and the
released trainer's `B+={tau_i:r_i>0.5}`. It adds no loss or reward shaping and is
outside VETO E1–E3. Malformed dataset labels or domains fail; ordinary incorrect,
empty or invalid-format replies receive zero. Claimed rewards in row metadata
cannot override the dataset truth. Correct answers alone do not prove image use.

The separately frozen visual configuration must use dataset source
`visual_evidence_binary` and the following native reward settings:

```yaml
reward:
  custom_reward_function:
    path: pkg://ours.visual_evidence_reward
    name: compute_score
    reward_kwargs: {}
  reward_model:
    enable: false
  reward_manager:
    source: register
    name: naive
```

Native package loading requires `pkg://`; a bare module string is interpreted as
a file path. The first failing reward probe is retained, and the dataset example
above has the same prefix correction. Under the visual test environment shown
above, run `data/training-runtime-v1/bin/python -m unittest -v
ours.tests.test_visual_evidence_reward`.

Two checks pass in21.047seconds. The native dataset factory, image processor,
SingleTurn loop, reward loader/manager/worker, score collation and original
`_fit_online_rsft` selection execute on CPU. Four scripted mixed replies score
`[1,0,1,0]`; only rows0/2 reach the actor stub, preserving pixel tensors and
response masks. Four invalid replies score zero and skip the update. Reward is
placed at the final valid response token, and model requests exclude truth and
reward metadata. Ray transport, model replies, actor/checkpoint operations are
stubs: zero model/API calls, zero optimizer steps, no distributed service or
visual model accuracy claim. The complete base suite is125passed/95dependency
skipped. Sixteen frozen records and153 unique code/configuration inputs retain
their hashes; large checkpoint weights were not rehashed by this check.
[Full record, including failed probe](../results/visual-reward-integration-check-20260910.json).

## Shared native visual tool execution

`VisualEvidenceToolAgentLoop` subclasses the pinned `ToolAgentLoop`, preserving
its parser, state machine, assistant-token limits and tool/image response masks.
`VisualEvidenceZoomTool` subclasses the pinned Math `ImageZoomInTool`, already
available through the existing shared bootstrap. Native bounding-box handling,
PIL crop, feedback and instance release remain in use. The adapter supplies the
initial model-visible PIL image directly, preserving its coordinate system;
candidate paths/URLs and row tool metadata cannot supply another image. The
upstream crop's unused Ray pool is not created for this local PIL operation.

Method correspondence: `I_next=native_crop(I_initial_visible,bbox)`, followed by
`r=v(last_assistant_span,y)`. The last contiguous generated span is identified by
the native response mask, so prior tool calls and feedback do not contaminate
the strict final A/B verifier. The loop supplies `reward_score` and
`reward_extra_info` to the existing native worker. Negative tool-error feedback
remains tool metadata and does not become task reward shaping. All visual
conditions must share these semantics; this is outside VETO E1–E3.

The separately frozen visual configuration will need both schema paths:

```yaml
data:
  custom_cls:
    path: pkg://ours.visual_evidence_dataset
    name: VisualEvidenceDataset
  tool_config_path: ours/visual_evidence_tools.yaml
actor_rollout_ref:
  rollout:
    agent:
      agent_loop_config_path: ours/visual_evidence_agents.yaml
      default_agent_loop: visual_evidence_tool_agent
    multi_turn:
      tool_config_path: ours/visual_evidence_tools.yaml
      format: qwen3_coder
```

Rows retain the shared `visual_evidence_binary` reward contract. This excerpt
specifies the integration fields, not a complete training plan or frozen visual
budget. Dataset and rollout must use the same tool schema. These tests clear
the existing Chess harness environment. The separate VisualHarnessAgentLoop below
now provides executable candidate dispatch; evaluation/search launchers and VETO's
acceptance call sites remain unwired.

Under the visual test environment above, run the native test module
`ours.tests.test_visual_evidence_tool_loop`. The final agent-YAML/Hydra test
passes in20.384seconds with six scenarios: crop/correct, crop/wrong, invalid
box, no-image, no final answer and direct answer. It executes the actual native
tool parser/state machine, inherited crop, processor and DataProto collation.
Ten scripted requests yield the expected rewards; crop observations have107
zero-mask tokens in each tested successful tool exchange, while its44 assistant
tokens retain loss mask1. Pixel grids, image-token counts and mRoPE agree.
No-image requests remain image-free even with an actual valid PIL image in tool metadata, and
unanswered tool calls score zero. Tool instance state is released after use.

The model server is a fixture. No actual model, actor, optimizer, distributed Ray
service or full trainer executes here; this does not demonstrate visual model
accuracy or train/eval parity in a running experiment. The base suite is now
125passed/96dependency skipped. Sixteen existing frozen records and153 unique
code/configuration inputs remain unchanged; large weights were not rehashed.
[Final integration record](../results/visual-tool-integration-check-20260910-v3.json)
preserves the earlier four-scenario and six-scenario checks as well.

## Executable visual harness in the native dataset and tool loop

`VisualHarnessDataset` and `VisualHarnessAgentLoop` inherit the shared visual
adapters. Method correspondence is the shared E3 search space
`h=(observation, tool_arguments, feedback, parser, continuation)`, followed by the
original E4 update on successful trajectories. This defines what a candidate can
change; VETO's paired evidence and acceptance constraints are separate.

| Callback | Visible inputs | Effect |
|---|---|---|
| `format_observation` | Question text | Formats user text while retaining the image before native prompt-length filtering |
| `prepare_tool` | Requested arguments and initial visible image dimensions | Changes crop arguments before the original tool executes |
| `format_feedback` | Tool response text | Formats the next observation within the shared response limit |
| `parse_answer` | Final generated assistant text | Commits an answer to the same binary verifier |
| `nudge` | Latest generated text and assistant turn count | Appends a user request only when native turn and token limits permit |

Labels, pair identities, reward metadata and hidden image paths are not callback
arguments. Both raw and committed answers are retained. Actual requested/executed
crop arguments, returned pixel hashes and continuation counts reach DataProto;
image/tool/nudge observations retain native zero loss masks. Custom counters use
`extra_fields`, because the original `AgentLoopMetrics` schema drops extra fields.
The first failed probe and subsequent correction remain in the evidence record.

The source SHA enters dataset caching and each row. Rollout rejects mismatched
row identity before generation, and callbacks reject source changes after loading.
Validation restricts module initialization, utility imports and mutable global or
default state. It is not OS isolation or proof that arbitrary Python has no side
effects. Real proposer/evaluation launchers must provide their own execution boundary.
`load_visual_harness(candidate, prompt_reference=reference)` enforces literal
`SYSTEM_PROMPT`/`USER_PROMPT` changes with identical nonprompt AST for prompt-only
search. The eventual FST controller must invoke that validation explicitly.

```yaml
data:
  custom_cls:
    path: pkg://ours.visual_harness_dataset
    name: VisualHarnessDataset
  visual_harness_path: ours/visual_harnesses/base_harness.py
  tool_config_path: ours/visual_evidence_tools.yaml
actor_rollout_ref:
  rollout:
    agent:
      agent_loop_config_path: ours/visual_harness_agents.yaml
      default_agent_loop: visual_harness_agent
    multi_turn:
      tool_config_path: ours/visual_evidence_tools.yaml
      format: qwen3_coder
```

This is an integration excerpt, not a complete training plan. Both components use
the same fixed candidate. Legacy Chess harness/prompt environment variables must
be cleared. Reward/data contracts and budgets remain shared across visual conditions.

Under the visual test environment above, run
`ours.tests.test_visual_harness`. Two native CPU tests use one existing engineering
pair in25.575seconds, with eight scripted requests across default parity, five-hook execution and
turn-limit scenarios. They exercise the actual keyword-based dataset factory,
Hydra agent loading, a160-to128-pixel crop change, feedback and continuation,
pixels/grid/mRoPE, reward/masks, source identity and native concatenation of tool
and no-tool batches. Mutable positional and keyword defaults are rejected in full
harness mode as well as the prompt subspace.

There are zero actual model/API calls or optimizer steps. This does not measure
visual task accuracy, full trainer execution or distributed service behavior.
`base_harness.py` matches the engineering prompt and loop; it is not yet a validated
strong visual baseline. Actual visual proposer/evaluation orchestration, VETO
acceptance and the scientific visual data protocol remain to be established.
The final base suite is125passed/98dependency skipped in1.12seconds. All16 frozen
records retain their hashes, with155 resolved code/configuration files checked
across both source maps. The earlier153 counted literal main-map paths, which
resolve to149 files; the expanded check records this distinction explicitly.
[Final source and test evidence](../results/visual-harness-integration-check-20260910-v2.json).

## Paired evaluation through native manager and worker batches

`visual_native_evaluation.write_pair_parquet` converts an explicit paired manifest
into the same bytes-only visual dataset used by training. The manifest declares
its role, question, two different image hashes and opposite A/B labels. Image
paths stay inside the manifest directory and bytes must match their hashes;
labels and pairing information stay outside model messages and candidate inputs.
The function refuses to overwrite an existing Parquet file.

`evaluate_pairs` consumes an initialized `AgentLoopManager` and a
`VisualHarnessDataset`, with a fixed `EvaluationIdentity`. Before generation it
checks every raw image/question/label against the pair manifest, full coverage
after native filtering, the actual verifier source digest and candidate identity.
Each batch must divide evenly across the manager's workers. It uses native
collation and `validate=True`, invoking the same visual loop as E4 training.

Method correspondence is E1, computed from committed answers:
`P_C=mean_i[v(answer_i,y_i) * v(answer_prime_i,y_prime_i)]`.
Ordinary marginal correctness is reported separately. Native output order is
joined using sample identities. The module replays the fixed parser, checks
binary rewards at their native final-token position and validates response masks.
Complete DataProto batches and per-example records are archived; an interruption
or invalid output preserves partial evidence without creating a result or
aggregate score. No missing prediction is assigned a zero to complete the table.

`visual_policy_calls` counts actual generating-state responses;
`visual_generated_tokens` counts returned assistant tokens before native final
batch truncation. Retained assistant tokens and observation-inclusive tokens are
separate. `mean_native_turns` preserves upstream's user/tool/assistant turn metric;
it is not the model-call count and this addition does not change E3 ranking.

```python
from ours.training_bootstrap import prepare_worker
prepare_worker()  # Before importing native agent/manager modules in a fresh process.
from ours.visual_native_evaluation import write_pair_parquet, evaluate_pairs, verifier_identity
from verl.trainer.main_ppo import create_rl_dataset

write_pair_parquet(manifest_path, new_parquet_path)
dataset = create_rl_dataset(str(new_parquet_path), config.data, tokenizer, processor, is_train=False)
# manager is an initialized native AgentLoopManager using the same fixed candidate.
# identity must bind the frozen phase, data, decoding and verifier_identity().
receipt, report = await evaluate_pairs(manager, dataset, manifest_path=manifest_path,
    identity=identity, output=new_output_directory, batch_size=32)
```

This is a callable evaluation entry, not a frozen GPU launcher. Serving weight
identity, actual decoding, Ray startup and resource/budget lifecycle require the
separate run wrapper. The module explicitly reports that it has not certified
weights or decoding. The returned audit retains the manifest's role; engineering
receipts cannot enter the existing VETO selector.

The CPU integration test executes actual native manager sharding, worker
`generate_sequences`, Hydra agent creation, crop, processor and DataProto
serialization. Manager/worker startup is bypassed; Ray RPC and model generation
are explicit fixtures. One pair yields three scripted requests across a direct
answer and a crop exchange. Returning the batch in reverse order still produces
the expected pair. Filtered coverage, uneven worker batches, corrupted reward,
duplicated output identities and a second-batch interruption are checked. There
are no real model/API calls or optimizer steps, and no scientific data or test
answers are loaded. The first cold-import failure and its bootstrap correction
are retained in the [integration record](../results/visual-evaluation-integration-check-20260910.json).

## Complete native search trajectory

`plot_search_trajectory.py` measures the existing E3 candidate-evaluation and
acceptance history; it adds no Method component. Extraction reruns the original
complete-search verifier and requires equality with the saved completion proof.
The figure shows every evaluated candidate and changes its accepted-harness
curve only at actual round boundaries. Failed/rejected candidates remain in the
complete source proof; candidates without a completed evaluation are not assigned
invented scores. The plotted cost covers candidate evaluation allocations only.

```bash
# Native runtime/PYTHONPATH: complete archive validation, no generation.
data/training-runtime-v1/bin/python -m ours.plot_search_trajectory --phase extract \
  --plan results/controlled-harness-only-seed42-plan-20260910-v1.json \
  --proof results/controlled-harness-only-seed42-completion-20260910.json \
  --data NEW_TRAJECTORY_JSON
# Existing base plotting environment: does not install into the native runtime.
python -m ours.plot_search_trajectory --phase render \
  --data NEW_TRAJECTORY_JSON --output NEW_FIGURE_STEM
```

The first real[PNG figure](../results/controlled-harness-only-seed42-search-trajectory-20260910-v1.png)
and[PDF](../results/controlled-harness-only-seed42-search-trajectory-20260910-v1.pdf)
contain all16 verified seed42 evaluations and9.165556 cumulative evaluationGPUh.
CSV and JSON are saved beside them. The same32 MH optimization tasks were reused;
there are no heldout results, independent sample pooling, error bars or ablation
claims. The native selector first retained h1 after round1 and retained it through
round5; later candidate scores did not exceed8/32. This diagnostic does not justify
changing the frozen search budget after inspecting one trial.


### 接受后视觉续训的固定h0复测（2026-09-10）

`visual_resumed_followup.py`只在真实续训、完整批次/参数审计和作业完成后准备计划。
它复用`native_visual_followup`的导出、真实评测和匹配输入比较；初始与续训后的评测
都使用同一h0。Method对应E4→E1，不是新增算法组件。真实FP32更新与BF16保存时的
舍入残差分别记录，不能以数据格式变化代替学习证据。当前仅2项数值CPU测试通过，
真实theta2尚未产生；首次复测也不能替代重复性检查、受控分叉或独立测试。

### Qwen3.5图像参与E4的核验（2026-09-10纠正）

`qwen35_visual_fused.py`修复上游通用文本fused接口丢弃图像参数的问题。它对应E4中
条件概率`πθ(a_t | I,q,h,a_<t)`的图像输入I；沿用原fused likelihood、成功过滤及
response mask，不增加损失或方法组件。`visual_pixel_bootstrap.py`记录真实视觉
backbone收到的像素和输出特征梯度，`audit_visual_backbone.py`再将其与成功轨迹匹配，
包括原生FSDP/BF16数据类型转换；全参数变化另行核验。

旧224041 checkpoint来自未用像素计算损失的训练，不能作为视觉训练链的起点。
`visual_training_resume.py`要求修正后首批的图像/梯度审计，并在续训中保留修复及观察器。
5项真实小型模型前向/梯度测试与7项恢复路径CPU测试通过；实际GPU完成以对应作业
结果为准，不能用CPU测试替代。所有失败记录和原代码快照保留。
