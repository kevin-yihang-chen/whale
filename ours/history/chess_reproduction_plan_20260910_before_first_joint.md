# Chess bootstrap diagnosis and reproduction plan

Updated 2026-09-10. Status: **WEIGHT_ONLY_SEED42_COMPLETED_WITH_DEVIATION; JOINT_PHASE1_PREFLIGHT_PASS; HARNESS_ONLY_H5_AUDITED_H6_RUNNING**.

222801 completed exit0 in4098 seconds/2H800/2.276667GPUh. The full128-trajectory
audit records3/64 then0/64 accepted, one optimizer update,139 completed calls and
1039118 output tokens. The original expected ID matrix was wrong: direct sampler
iteration omitted native DataLoader initialization/reset RNG advancement. Actual
native configuration/seed-derived order, both recorded batches and saved sampler
state agree. Preserve the explicit deviation, original plan and failed probes;
future preflights must use the actual DataLoader. Native CPU model/extra loading
and real saved-data continuation pass; joint GPU resume is still pending.

Common-theta0 h0 evaluation222889 completed exit0 at02:28:54:1/32 solved,
33 calls,260128 output tokens,1053 seconds/2H800/0.585GPUh. Both workers passed
eight-coordinate weight checks and recorded seed42. At02:31:10 controller930536
failed after six GLM requests because iteration_1/report.md was not zero padded.
All three generated candidates pass native validation; none was GPU-evaluated.
An explicit frozen recovery amendment preserves this failed proposal and h0,
normalizes only the current report filename, and replays without additional calls.
Three native tests verify identical five-round histories and perfect stopping;
the actual complete search remains pending. No old engineering scores are reused.
At02:46:33 recovery controller1019931 replayed the exact first proposal inside
the native loop with zero new API calls and submitted h1 job222891. Both actual
workers passed initialization-coordinate checks and entered inference. The
failed controller/proposal and recovery amendment remain separately archived.
At03:03:49 h1 completed exit0:8/32 solved,32 calls,245056 output tokens,
1036seconds/2H800/0.575556GPUh. h2 job222958 started at03:04:04. The original
selector still awaits all first-round candidates. Fixed-reply diagnostics show
zero new/old parser action or legality disagreements on all32 calls; all8
successes retain their actions under the old parser. Six end thinking normally,
two contain length truncation. This is not a counterfactual generation ablation.

The unstarted seed43/44 weight-only plans now pass complete Hydra and actual
RLHFDataset/StatefulDataLoader preflights. Their first two batch IDs and saved
sampler states match the separate index probe. They preserve common theta0,
h0 and continuous two-batch training. Neither trial has been GPU-submitted;
the original seed42 plan/deviation remains intact. Use the new continuation
entrypoint, whose source and TorchData runtime are frozen in each new plan.

h2 completed exit0 at03:21:06:5/32,32 calls,256392 output tokens,
1022seconds/2H800/0.567778GPUh. h3/222969 started03:21:35; the original
first-round selector awaited all candidates at that snapshot. h3 completed at03:38:37:
0/32,33 calls,260128 output tokens,1022seconds/0.567778GPUh. The native selector
accepted h1 at03:39:05. After eight GLM calls, the second proposal failed local
metadata parsing at03:40:45: a bare list with harness aliases instead of the
expected candidates/name schema. No second-round candidate was evaluated.

The separately frozen second recovery maps only agreeing metadata aliases and
wraps the original list, retaining all fields and exact candidate bytes. It reuses
all four audited evaluations and proposal2 inside the original loop at iteration2
for four remaining rounds. Three native tests compare two interruptions against
uninterrupted five-round/perfect-stop histories, including identical evolution
rows and no repeated generation. Controller1101994 resumed at03:50:35 with zero new API calls; h4/223135 is
running on2H800 and both worker weight checks pass. The full search is still incomplete. API
conservative usage at that snapshot was0.200845CNY/35 settled calls; see the latest update below.

The independent WHALE and WHALE-FST seed42 phase-one plans are frozen and pass
actual Hydra/native-task-loader preflight. Each uses common theta0/h0 and a
cumulative target of1, with64 new trajectories and model/extra/data.pt saving.
The native loop fixture verifies exactly one generated batch for0→1 and1→2,
including save/sync for empty updates; the second fixture does not execute a
checkpoint loader. The new first-phase auditor replays all64 rows/75 calls in
a clearly synthetic provenance copy of the archived weight-only batch. That
fixture is not a joint-condition result; the original source remains unchanged.
Both first-phase jobs remain unsubmitted. Joint search provenance is now
implemented in joint_staged_search; second-phase launch, completed-search verification and passive restore observation
now pass CPU tests; actual GPU restore and phase-two trajectory/completion audit remain pending. First-phase terminal and canonical
export closure are now implemented in joint_training_result/joint_checkpoint_export.
Three native state/completion fixtures pass, with complete64-row/request coverage
and actual saved sampler/scheduler checks. Two export tests execute the real native
merger subprocess on a34-tensor/42576-element small VLM, checking every exported
value and inference assets, including unchanged/sub-BF16 updates. Fixtures have
explicitly synthetic provenance; no real joint checkpoint or export exists yet.

The final weight-only checkpoint export plan is frozen at
results/controlled-weight-only-seed42-export-plan-20260910-v1.json. It binds
completed training evidence and the common theta0, retains the preflight-order
deviation, and measures all active native/BF16 values independently. Five native
CPU tests pass, including source generation-asset restoration, decode-drift
rejection and unchanged/sub-BF16-delta cases. Actual export has not run; do not
race the resident search controller for the account's sole submitted-job slot.

Historical startup:222801 started the first genuine controlled weight-only pilot from the common
untrained theta0: two continuous native batches,128 fresh trajectories,90-minute
limit, no proposer call. Plan: results/controlled-weight-only-seed42-plan-20260910-v2.json.
At01:47:33,139 requests started,91 completed with648926 returned tokens and zero
recorded errors. Batch1 native logs report3/64 accepted and one SFT update;
batch2 and the complete independent audit remain pending. Shared
Ray worker setup and initial CuPy release executed. See controlled_pilot_protocol.md
for the fixed four-condition/three-seed pilot and remaining full-scale differences.

The new resident staged_search controller passes six native CPU tests, including
five original iterations x3 slots, perfect stopping, FST rejection, failed-job
proposer suppression, and old real h0/h1 reply replay. No fresh inference/API was
performed by those tests. The common-theta0 harness-only seed42 plan is frozen;
its two-shard GPU evaluations wait for the account's sole submit slot. Existing
seed42/single-candidate helpers stay unchanged for earlier evidence. Joint-arm
controlled checkpoint provenance/resume still requires implementation.

222792 completed exit0: actual vLLM model/CPU/CUDA seeds42/43/44, distinct initial
RNG states, eight verified embedding coordinates per engine,24 toy requests and
64 tokens.114 seconds/2H800/0.063333GPUh. Six native CPU seed/condition fixtures and
all three native Hydra configurations pass. The complete fresh worker setup also
passes in a standalone CPU process; it is not yet a Ray-spawned training trial.

The original-checkpoint cast222794 saved all active weights, then failed a strict
GenerationConfig contract: save_pretrained added a file absent from the source,
changing output_hidden_states/output_attentions from False to None.77 seconds/
1RTX4090/0.021389GPUh are charged; the failed source/log/asset are archived. Recovery
222797 moves only that known extra asset into its receipt, rechecks every active
value against the source BF16 cast and verifies the original inference contract.
It completed exit0 in95s: all723 active tensors/4,539,265,536 elements equal the
source BF16 cast;2791 FP32-origin elements round, all originally BF16 values stay
unchanged. Weight files are unchanged during recovery, inference contracts pass,
and no optimizer/inference/API call occurs. Shared theta0 manifest SHA:
260337087e4d739db98c2744cf429678cd2f87674e55cad9f22d3cfbe60c3d1a.
Known allocation is9.273056GPUh. No F0 condition or visual gate has passed.

Next execution requirements:

- Start all four controls from the verified, untrained common BF16 theta0. Do not
  substitute the engineering theta1/theta2 checkpoints or reuse their trajectories.
- Use the new shared seed/backend/transport hooks. Existing mh_phase helpers bind
  seed42 and one candidate; independent43/44 and multiple native search iterations
  require a new staged adapter, not a renamed copy of the seed42 result.
- Native online_rsft.iterations overrides trainer.total_training_steps whenever
  positive. The engineering wrapper fixes it to1; scheduled trials must explicitly
  set it to0 and use a declared cumulative target, or explicitly record a bounded
  per-phase batch budget. A native batch step can contain several optimizer steps.
- Released defaults are train16384, train_batch_size256, rollout_n8, total256
  native batch steps, round13, MH256,5 search iterations and3 proposals each.
  Our128/32/64 split and batch8 are a pilot protocol, not those paper-scale defaults.
- Native alternation resumes model/extra and the dataloader, but the released
  launcher excludes optimizer state. A fresh HF initialization is not that resume.
  Do not share a training prefix across methods if doing so changes Adam, scheduler,
  dataloader or RNG continuation. Freeze and report any matched phase-reset control
  separately from the released continuous weight-only baseline.

Earlier completed handoff evidence follows.

222640 completed exit0 at23:34:00 HKT after368s on2H800 (0.204444GPUh). The original
actor completed two optimizer minibatches of8 and3 rows, with30392 loss tokens.
Checkpoint SHA f42e2219040b036e0b926fdd5c2ad0d0d0c7082d715c0177cc069c3433b68ada.
Native loss1.2237049 sums two mini-batch token means; grad norm8.16329 is the reduced
two-step mean. Initial/post-update NCCL finalize each released6GiB+512 idle CuPy
bytes, with active allocation accounting unchanged. Real GPU fixture passed.
Torch allocated/reserved peaks75.803/77.352GiB. Native save/sync completed; shutdown
logged a killed DataLoader worker, preserved alongside authoritative exit0 evidence.

Both prior222512/222615 failures remain archived. Known project GPU allocation through training222640 was
9.064167 hours; training added no GLM or trajectory-generation calls. Canonical export222703 completed exit0:723 active tensors,3,845,402,884 native
FP32 elements changed and14,479,791 changes survive BF16 export. Every exported
value equals the native BF16 cast. Fresh serving222779 completed exit0 in80s on oneH800:all eight actual embedding
values match the new export and differ from incoming weights. One request returned
two tokens(Yes), not the requested ready; this is execution proof only. The engine
logged shutdown complete, plus a destroy_process_group warning retained in evidence.
Known project allocation is9.161944GPUh. No GLM call was added. These are shared
E4 execution checks; F0-F5 and any VETO/heldout improvement remain unestablished.

Earlier source failure and recovery preparation follow.

222292 failed at22:33:45 HKT after1992 seconds/2H800/1.106667GPUh. All64 fresh
trajectories and69 calls are recorded:452201 output tokens,11 accepted trajectories
on two training puzzles,30392 loss tokens. Exact native request/config/token/mask
replay and independent board continuation pass. Native lm_head forward then OOMed
requesting4.82GiB with3.68GiB available. No final actor metrics/checkpoint exist;
whether an earlier mini-batch update ran is unknown. Preserve the complete batch
for recovery without resampling. The released Torch fused-output backend passes
FP32/BF16 small hybrid-model CPU fixtures; relative gradient L2 differences are
1.17e-7 and0.00410. BF16 is not bitwise identical, so formal controls must share
the backend. No new GPU recovery or real4B/FSDP validation is submitted yet.

The preceding live snapshots follow.

Job222196 completed the full32-case MH baseline on the new checkpoint:0/32,
35 calls,260128 output tokens, with exact native replay and independent board/token
audits passing. User-authorized GLM search then produced h1 in7 API requests;
job222212 completed under the same frozen contract. Its h1 scored6/32 versus0/32
for h0; independent native replay/board/token audits passed and the original
search selected h1. Four successes ended thinking normally; two were parsed
from truncated reasoning under the original rule. This is MH optimization-data
evidence, not heldout or VETO performance. The candidate's
last-tag-wins description is inconsistent with its actual parser; preserve this
discrepancy remains documented and the generated code was evaluated unchanged.
The next8 training IDs are preselected by the existing engineering order, not by
success; the final RSFT plan now binds the completed search and both audits,
and its actual native Hydra CPU preflight passed. Job222292 started on two H800
at22:00:33 HKT; the22:07 snapshot has64 request starts and8 completed calls,
63083 output tokens, no recorded errors. No second update is claimed yet.

The22:27:57 HKT snapshot remains the same running job:69 starts,56 completed,
376769 returned output tokens,13 pending and no recorded errors. A separate
selected-harness batch auditor now passes four native CPU checks and verifies
the live request configuration hash against the frozen native configuration.
It distinguishes a solved board from a response-length rejection of feedback;
it fails closed if discarded replies cannot be matched to recorded events.
The next-checkpoint comparator passes a serialized-tensor fixture, separating
actual FP32 updates and BF16-surviving updates. Actual batch/parameter audits
remain pending; these are shared E4 measurements, not a new research mechanism.

Offline application of both parsers to the recorded replies finds identical
actions for all six successful h1 transcripts. The parser change cannot explain
those successful move extractions; actual prompt/observation ablations still
need new controlled sampling. This diagnostic does not replace the FST control.

The missing WHALE-FST proposer contract now has an explicitly reconstructed
component in `prompt_subspace.py`, retaining the native AST rule and adding an
execution-before-load literal-string guard. Four native CPU tests pass. This is
not an unpublished author skill, an integrated GPU FST run, or a completed F0.

Earlier preparation and failure evidence follows:
Job221898 completed the first64 trajectories on two H100 replicas. The exact
[sampling plan](../results/chess-bootstrap-plan-20260909.json) was frozen before
submission. The complete independent audit passed:2 successes on two distinct
training puzzles,70 calls and520237 output tokens. Do not expand to256.
Job221906 failed before generation because the original NCCL engine lacked CuPy.
The exact dependency is now installed and its explicit CPU import passes. Job221907
then exposed an undersized transport bucket before generation; the original3072MiB
default is restored and checked against actual FP32 tensor sizes. Job221912 passed
initial weight sync, then was proactively stopped because the communication timeout
did not account for native queue waiting. No timeout was observed, and its partial
generation counts remain unknown. Job221921 uses the declared4800s deadline and
completed64 fresh online trajectories, accepted1, then failed on Adam foreach
temporary memory during its first step. No completed update/checkpoint exists.
A CPU native-optimizer comparison and a bounded synthetic single-H100 memory
probe passed with foreach=false. Job221967 completed64 new trajectories, accepted2,
then failed during dense response log_softmax allocation. Its complete native
request/tensor/mask audit passed. Job222151 failed before GPU initialization due to
a worker cwd path mismatch; that failure and0.041111GPUh are preserved. Recovery222154 restored the64 rows but rejected the original dataset placeholder
field before actor execution. The explicit zero uint8[N,1] field is now preserved,
and the real dataset-to-original-trainer-to-actor-entry CPU fixture passes. GPU
recovery222156 uses the same audited batch with no new calls, original filtering
and optimizer, and verified response-suffix compaction. Left padding is preserved.
Job222156 completed one native optimizer step, save and the original NCCL call.
Job222165 completed canonical export and the full723-independent-tensor value audit.
The export retains measurable training changes in original BF16 tensors. Its3840
originally FP32 elements also undergo BF16 rounding; formal comparisons must match
serialization precision or include a cast-only control. Job222175 loaded the canonical export but failed before the worker callback
because callable serialization is disabled; its206s/0.057222GPUh are archived.
The official worker extension and a named RPC now pass native CPU serialization
and BF16-value checks. The revised probe222176 verified eight distinguishing worker values and completed
one request with2 generated tokens. Its output was Yes, not the requested ready;
this is generation/weight evidence only. Children remained alive after the saved
result, so the allocation was explicitly cancelled after222s/0.061667GPUh. The
measurement passed; a normal process exit did not. Explicit engine shutdown has
since been added but not rerun on GPU. Original in-place NCCL receiver values
were not independently sampled. No held-out improvement or F0 result is established.
This document replaces the immediate transition from eight-case checks to the
full four-condition/three-seed pilot. Upstream remains pinned at
`fbe125eb7abea7f760c99ab9acc1a6261e708fc6`.

## Evidence and interpretation

The authors report the same initial failure pattern: 2,025/2,048 base trajectories
hit the 8,129-token cap. Their successful Chess example combines trained weights
with structured observations, history and commitment parsing. These are reported
results, not our reproduction: [paper Appendix D.3](https://arxiv.org/html/2609.00196v1#A4.SS3).

Our [offline termination audit](../results/chess-termination-diagnosis-20260909.json)
checks archived token IDs as well as text: all 16 h0/h1 calls lack a thinking-end
token and a concrete move tag. A parser cannot recover an absent final answer.
The [verdict audit](../results/native-search-221754-summary.json) records 15
ambiguous outputs and one legal wrong move. This does not isolate model ability.

The release has two different limits, not one interchangeable budget:

- [Training launcher](../upstream/WHALE/domains/chess_puzzles/scripts/train_chess_puzzle_multinode_disagg.sh):
  `MAX_TOTAL_RESPONSE_LENGTH=16384`, `ASSISTANT_TOKEN_BUDGET=8129`.
- [Agent loop](../upstream/WHALE/domains/chess_puzzles/verl/experimental/agent_loop/chess_puzzle_agent_loop.py):
  the response region includes appended nonassistant messages with loss mask zero;
  assistant generation also obeys its separate cumulative and per-call caps.
- [MH configuration](../upstream/WHALE/domains/chess_puzzles/meta_harness/config-chess-puzzle.json):
  assistant and per-call budgets are both 8129. Changing these to 16384 would be
  a declared diagnostic intervention, not an established correction to the paper.

The [release reproduction notes](../upstream/WHALE/docs/reproducing.md) disclose
that the cleaned launchers have not undergone an end-to-end GPU run. Our existing
compatibility changes and missing checkpoint handoff therefore remain relevant.

## Stage A: test the success-sampling prerequisite

Use Qwen3.5-4B at the existing pinned revision, unchanged h0, thinking enabled,
temperature 1, top-p 1, top-k 20, and the separate release token limits.
VETO remains off. No proposer API is needed for this stage.

1. Select the first 32 IDs in lexicographic order from the existing 128-case
   pilot **training** split, before inspecting model results. Preserve the 32 MH
   and 64 test roles. Record the selected IDs and hashes in a frozen execution
   plan before submission; do not choose easier cases after failures.
2. Generate eight trajectories per selected prompt, with recorded seeds 42–49.
   The first diagnostic chunk covers eight prompts (64 trajectories). If it has
   no complete successes, extend once to the remaining 24 prompts, for a maximum
   of 256 trajectories and 2,081,024 generated assistant tokens across Stage A.
   These are trajectory limits, not HTTP-call counts or measured GPU costs.
3. Report all outcomes: complete success, correct first move, actual legal move,
   ambiguous/missing output, truncation, thinking termination, calls and tokens.
   Do not substitute native `legal_rate` for independently checked legality.
4. At a completed chunk with accepted trajectories, verify the native training
   interface, then run one native RSFT step. The prepared original trainer
   freshly samples64 trajectories from the same eight training prompts; only
   its own accepted set feeds its optimizer. This additional sampling is counted
   separately from the Stage A feasibility budget. Reusing the MH sampler's
   archived tokens would require another replay integration, so cached calls
   are not silently relabelled as live native training. Verify actual parameter changes, checkpoint
   export, service weight synchronization and inference from the new checkpoint.
   A saved checkpoint with an empty/skipped update does not pass this gate.
5. A nonempty update establishes engineering execution only. It does not prove
   improved held-out accuracy, a stable training regime or scientific F0 success.
   Zero successes at the 256-trajectory cap ends Stage A without claiming an
   optimizer update. Do not automatically multiply the sampling budget.

Before a future submission, use live queue/quota evidence to compare one GPU
with independent replicas on two GPUs, and freeze resource/time limits and
merge identities. No wall-clock estimate or allocation is claimed here.

## Stage B: a bounded diagnosis if Stage A remains empty

On the same declared training subset, compare h0 versus a hand-implemented,
paper-informed harness, crossed with thinking on/off. Preserve weights, sampling,
verifier, seeds and cumulative token budget across the relevant contrasts.
The harness intervention jointly changes observation presentation and explicit
commitment handling; this alone cannot attribute gains to either component.

The structured harness may reorganize visible board/legal-move information and
retain accepted history. It must not consult reference continuations or use an
engine to choose moves. Parsing must account for the prompt's implicit open
thinking segment, including outputs with no closing delimiter. Arbitrarily
choosing one move from unfinished reasoning is not an output-format repair.

Declare this a **manual reproduction control**, not a GLM-discovered candidate
or our VETO contribution. Freeze its exact code and diagnostic budget before
running. Reuse prior calls only with complete input/configuration identity.
Disabling thinking is a diagnostic variant, not the original reported baseline.

## Stage C: restore alternation after the gate

Verify the chain `accepted on-policy trajectories -> native RSFT -> updated
checkpoint -> MH evaluation of that checkpoint -> next RSFT phase under the
accepted harness`. GLM may remain the proposer, explicitly a replacement for
the release default; first require trace inspection and compare distinct changes
to the permitted harness components. One prompt-only candidate is insufficient
evidence about full-harness search. Use a shared deterministic tie policy.

Only then scale the four-condition/three-seed pilot, keeping the final test set
out of debugging and candidate selection. The current 128/32/64 dataset is a
pilot; its result cannot be labelled the original paper's test score.

## Method correspondence

These steps validate the existing E4 success-filtered training and execution
path. E1 paired correctness and E2–E3 acceptance remain unchanged. Observation
formatting and commitment parsing belong to shared harness controls; no new
scientific mechanism or claimed improvement is introduced by this plan.

## Execution artifacts

- [Sampler](chess_bootstrap.py), [frozen input builder](prepare_chess_bootstrap.py),
  [Slurm launcher](run_chess_bootstrap.sh), and [independent replay](audit_chess_bootstrap.py).
- [Native AgentLoop offline replay](replay_chess_agent_loop.py) checks exact
  archived requests, rewards and response-token masks without calling a model.
  It reports native reencoding versus original generated token IDs separately.
- [Native training pilot launcher](run_native_rsft_pilot.sh) retains the original
  trainer, actor and online sampling. Its CPU configuration/dataset check passes;
  job221967 uses the [revised frozen plan](../results/native-rsft-pilot-plan-20260909-v5.json).
- [Startup gate](rsft_pilot_gate.py) verifies the complete bootstrap, source/runtime
  identities and actual resolved configuration before native training begins.
- [Tensor transition audit](check_checkpoint_transition.py) checks real parameter
  values after export; it does not infer better task performance from an update.

Latest search update at04:09HKT: h4/223135 completed exit0 at04:07:51,6/32 solved,
32 calls,255996 output tokens,1036seconds/2H800/0.575556GPUh. Its full audit and
terminal evidence are archived. The resident controller automatically started
h5/223199 at04:08:06; both actual worker checks pass. Round2 still awaits h5/h6;
no new proposer calls and no second-round selection yet. Known terminal project
allocations total14.421389GPUh; h5 remains pending terminal accounting.

Joint search implementation: joint_staged_search certifies the independent joint
phase-one training/export lineage and resumes the original h0-based five-round
search at the actually exported theta1. The base writer supplies only common
search settings; its untrained model qualification and scores are not reused.
WHALE-FST keeps the native prompt restriction. Three native tests pass: five-round
and perfect-stop equivalence, FST rejection, full archived32-case/34-call replay
through the joint worker route, and provenance/config mismatch rejection. The
lineage fixture uses synthetic terminal certification; no actual joint plan or
checkpoint is claimed. Export closure now has canonical terminal/artifact paths,
with real small-VLM tests verifying relative/absolute argument equivalence.

Latest search update at04:27HKT: h5/223199 completed exit0 at04:25:19,7/32 solved,
32 calls,246051 output tokens,1033seconds/2H800/0.573889GPUh. Its full audit and
terminal evidence are archived. h6/223204 started04:25:36; both actual worker
checks pass. Round2 has not selected a candidate; h1 remains the incoming accepted
harness. Known terminal project allocations total14.995278GPUh, excluding live h6.

The zero-inference h5 diagnostic compares all32 recorded replies with the original
h0 parser: zero action/legality disagreements and all7 successful actions remain.
Five successes end thinking normally; two include a length stop. This supplies no
observed action-level benefit from the parser extension on this archive, and is
not a causal prompt/parser ablation or an equivalence proof for every input.

Latest search update at05:03HKT: h6/223204 completed exit0 at04:42:38,8/32 solved,
32 calls,247598 output tokens,1022seconds/2H800/0.567778GPUh. The full audit and
native round-two comparison are archived; h1 remains accepted. The third paid
proposal completed seven GLM calls but stopped on the unrecognized candidate
metadata alias. Explicit recovery3 retains all candidate code and prior evidence,
adds only the agreeing alias, and resumes the original loop at iteration3 with
three remaining rounds. Controller1169570 reused all seven audited evaluations
and the exact third proposal, then started h7/223272 at04:53:28. Both actual worker
checks pass. Known terminal allocations total15.563056GPUh, excluding live h7.
Project API accounting is0.272512CNY/42 settled calls, zero unresolved calls;
44.727488CNY remains available for reservation within the45CNY project ceiling.
This is conservative request accounting, not an invoice or provider balance.

Native phase-two entrypoint: joint_training_phase2 requires a completed independent
joint search, the selected harness and the same phase-one native checkpoint.
search_completion reproduces per-round native validation, FST rejections,
ranking and stopping from every audited evaluation without rewriting the archive.
native_resume_observation checks all loaded FP32 actor values, scheduler and RNG,
and inspects pending StatefulDataLoader state without creating an iterator.
CPU fixtures exercise actual checkpoint loading, unchanged subsequent random
draws, native Hydra resolution and the correct next task batch. The complete hook
retains the shared recording/compaction and is idempotent in the upstream cwd.
Archived weight-only data is only an explicit loader fixture, not a joint handoff.
No formal phase-two plan or GPU resume exists yet. Phase-two full trajectory
and terminal auditing, actual joint training/search and heldout evaluation remain.

At05:12HKT, h7/223272 is fully audited:6/32,32 calls,250228 output tokens,
COMPLETED/exit0 at05:10:54,1046seconds/0.581111GPUh. h8/223273 started05:10:59;
both worker proofs pass. No round-three selection or additional proposer call
yet. Known terminal project allocations total16.144167GPUh, excluding live h8.
The final12-test native integration suite passes, with116 base tests passing and
81 native-dependency skips. Existing frozen source hashes remain unchanged.
The first integration run's negative fixture left a fifth proposal request and
hit that guard before its intended early-completion check; the fixture was fixed
and both failed/passing logs are retained. No real archive or criterion changed.

At05:53HKT, h8/223273 and h9/223332 are fully audited at7/32 each,32 calls each,
249983 and243851 output tokens respectively. Their completed allocations are
1027 and1047 seconds on2H800; known terminal project use is17.296389GPUh.
The native third-round selector retains h1. Proposal4 completed9 GLM calls,
0.084025CNY; conservative project accounting is0.356537CNY across51 settled
requests, no unresolved calls,44.643463CNY available under the45CNY ceiling.
This is not the provider bill or account balance. h10/223408 is running and both
worker weight/seed checks pass; the full five-round search remains incomplete.

The phase-two full batch auditor, restore-receipt reader and terminal finalizer
are implemented, and canonical export supports both joint phases. An actual old
B2 archive supplies64 CPU replay fixtures; synthetic GPU/Slurm identities are
explicit and are not joint results. Real small-VLM native exports exercise both
phases with unchanged and FP32-only changes. Final native integration:12 PASS
in64.651s; base117 PASS/84 dependency skips. Initial fixture omissions and their
failed logs remain archived. Five frozen training/search plans and three recovery
source bindings are unchanged. No actual joint GPU resume or heldout result yet.

At06:07HKT, h10/223408 has completed and passed full audit:8/32,32 calls,
249684 generated tokens,1013 seconds/2H800/0.562778GPUh. Known completed project
allocation is17.859167GPUh. h11/223466 started06:04:58; both worker proofs pass.
Round4 still awaits h11/h12 and no final five-round search result exists yet.
The unopened heldout protocol is frozen; four logical16-case shards support
predeclared4/2/1-GPU execution widths without changing ID assignment. No test run.
Read-only native semantics checks confirm equal-cost lower-score points remain
on its frontier, while acceptance still prioritizes score. Retry environment
values are defaults, and selected harness attributes keep their native precedence.

At06:25HKT, h11/223466 is complete and fully audited:6/32,32 calls,248721 tokens,
1024 seconds/2H800/0.568889GPUh. Known terminal project allocation is18.428056GPUh.
h12/223507 started06:22:30 and both worker proofs pass; round4 awaits its result.
API accounting remains0.356537CNY/51 settled calls, no unresolved requests.

Continuous weight-only seeds43/44 now have a completion finalizer and adapter to
the shared canonical exporter. The preserved seed42 exporter also has a separate
read-only terminal finalizer, retaining its schedule-preflight deviation. Eleven
native CPU checks pass, including six actual tiny-VLM native exports; final base
suite119 PASS/87 dependency skips. Historical trajectory/state and synthetic
launch identities are explicit fixtures, not new trials. Eight frozen execution
records and97 seed42-export code sources are unchanged. No new GPU jobs/test reads.
The final trial-identity matrix and executable64-case heldout runner remain pending.


At07:05HKT, h12/223507 and h13/223573 are complete and fully audited:6/32 and8/32,
32 calls each,238780/246575 generated tokens,1025/1026 seconds on2H800.
Known terminal project allocation is19.567500GPUh. Round4 retained native h1;
round5 h14/223615 is running with both worker proofs PASS, and h15 is pending.
Project API conservative accounting is0.474527CNY/62 settled calls, no unresolved
requests; this is not a provider invoice or account balance.

Final12-trial certification, four-shard64-case inference/audit, and descriptive
three-seed aggregation are implemented. Six native CPU tests pass in19.174s,
two summary tests in0.024s, and base124 PASS/90 dependency skips. The replay
fixture duplicates32 old MH cases into64 synthetic IDs; child GPU processes,
cohort identities and parent certification responses are stubbed. No heldout
rows/replies were read, no official final cohort or plan was created, and no
new model call occurred. The source report binds eight unchanged frozen
execution records and preserved seed42 export code. Missing trials remain
missing; known failed allocation subtotals are retained, with incomplete cost
coverage explicit. Actual joint runs, GPU heldout evaluation and F0 remain open.


At07:19HKT, h14/223615 is complete and fully audited:8/32,32 calls,236698 tokens,
1015 seconds/2H800/0.563889GPUh. Known terminal project allocation is20.131389GPUh.
The final h15/223617 is running with two actual worker proofs PASS. API accounting
remains0.474527CNY/62 settled requests. No completed five-round proof exists yet.

All six remaining first-stage plans are prepared and CPU checked: harness-only
seeds43/44 and independent WHALE/FST phase1 seeds43/44. The harness-only adapter
predeclares the existing metadata compatibility without changing the original
loop/evaluator. Two native tests run synthetic full-five-round and perfect-stop
fixtures; base124 PASS/92 dependency skips. No corresponding GPU jobs are submitted.
The single-use scheduling coordinator PID1330451 waits for completion certification,
then checks and submits the independent seed42 canonical export once. Its live
state is data/seed42-search-export-handoff-20260910/state.json; do not race its slot
or restart it based only on a stale state file. Original plan inputs remain frozen.

Observed free storage is about158GiB; the remaining16 native checkpoints and15
canonical exports project435.60GiB before temporary files and any verified reuse.
The next phase fits; full retention needs extra capacity or verified storage reuse.
A shared-storage path was requested while current work continues. No artifact
was deleted/deduplicated and no other project changed.
