# Chess bootstrap diagnosis and reproduction plan

Updated 2026-09-10 10:58 HKT. **HARNESS_ONLY_SEED42_SEARCH_COMPLETE; WEIGHT_ONLY_SEED42_EXPORT_COMPLETE_WITH_TRAINING_PREFLIGHT_DEVIATION; WHALE_SEED42_PHASE1_AND_EXPORT_COMPLETE_JOINT_ROUND1_H2_SELECTED_H4_H5_CERTIFIED_H6_RUNNING.**

The controlled common-theta0 search completed all five rounds and16 evaluations; the native selector retained h1 (8/32), with h0 at1/32 on the same MH optimization tasks. These are not heldout results. Weight-only222801 completed128 fresh trajectories and one optimizer step; canonical export223620 passed the full723-tensor native-to-BF16 comparison. The original expected task-order deviation remains part of its result.

The independent WHALE seed42 first batch223622 completed exit0 in2207seconds/1.226111GPUh. All64 rows passed native/token/mask/board auditing:6 accepted,45996 loss tokens,68 calls/517481 output tokens,zero errors and one real optimizer step. Saved sampler/scheduler and RNG state passed; its own canonical export223757 completed244seconds/0.067778GPUh and passed the full723-tensor exact-BF16 check. Its fresh joint h0 evaluation223824 completed1163seconds/0.646111GPUh, with actual two-worker coordinate/seed checks and full32-task auditing:1/32 solved,32 calls/260128 tokens, all calls at the8129-token limit without thinking end. Equal aggregate score to common-theta0 h0 hides two changed success labels; it does not mean identical predictions or unchanged weights. Joint h1 evaluation223895 also completed1/32, with32 calls/260128 tokens and all length stops, in1134seconds/0.630000GPUh. Joint h2/223896 completed5/32 in1099seconds/0.610556GPUh; h3/223898 completed3/32 in1129seconds/0.627222GPUh. The native first-round selector retained h2. Round-two h4/223905 completed6/32 in1125seconds/0.625000GPUh, with32 calls/258316 output tokens. Joint h5/223908 also completed6/32 in1098seconds/0.610000GPUh, with32 calls/256806 output tokens. Controller1413742 is evaluating h6 as job223909; round two has not selected a candidate. The independent searches do not share candidate code merely because both call a candidate h1. Do not substitute the independent weight-only checkpoint or harness-only selection. The remaining seeds/conditions have frozen plans but no completed trials. Native GPU phase-two restoration is still pending.

The shared visual dataset passed two native CPU input/batch tests. The shared binary visual reward passed two further tests including native dynamic loading, reward placement and original RSFT filtering with mixed/all-invalid scripted replies. Pixel tensors and response masks survive selection. Actor/Ray transport remain stubs; no actual visual actor forward/backward or end-to-end VETO run has occurred. The shared native crop/tool loop now passes one CPU test with six cases through actual agent YAML/Hydra, tool parsing/crop, final-answer reward and image/mask collation. Valid hidden image metadata cannot restore evidence in the no-image control. Executable visual candidate dispatch now passes two native CPU tests, including five callbacks, source identity, real crop changes and native tool/no-tool batch collation. Model replies remain scripted; the paired evaluation callable now passes native manager/worker/loop CPU dispatch, output identity joins and partial-failure handling. Actual serving startup, visual proposer, VETO selection and actor update are pending. All16 frozen execution records remain unchanged; the latest check covers155 resolved code/configuration files including the joint source map. Earlier153 counted literal main-map paths, corresponding to149 physical files. No heldout tasks have been loaded.

The historical bootstrap stages below explain prerequisites, not the current queue. Consult[PROJECT_STATUS](../PROJECT_STATUS.md) for the current matrix and[the archived chronology](history/chess_reproduction_plan_20260910_before_first_joint.md) for failures and successive snapshots.

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
