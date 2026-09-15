# Controlled Chess heldout evaluation

This extension fixes the final evaluation before reading test tasks or replies.
It keeps the train128/MH32/test64 partition and the four conditions and three
seeds in `controlled_pilot_protocol.md`. The accompanying JSON is a protocol
record, not a runnable plan or a claim that training has finished.

For each condition c and seed s, the primary measurement is

`A[c,s] = (1/64) * sum_i exact_solved(theta[c,s], h[c,s], x_test[i])`.

The final table reports each seed and the mean and sample standard deviation
across seeds42/43/44. A trajectory succeeds only if the original verifier and
independent board replay agree on the complete reference continuation. This
measurement is the final evaluation box after E3/E4 in the Method diagram; it
does not add a loss, select another harness, or establish a VETO contribution.

Before the first test task is loaded, seal the status of all12 condition/seed
trials: either a certified final model/harness pair or a preserved failed or
incomplete run with its reason and cost. A missing trial stays missing in the
comparison; do not impute a score or present a partial-seed mean as a full result.
Only certified completed trials may run heldout inference. Their identities are:

| Condition | Final weights | Final harness |
|---|---|---|
| weight_only | Canonical export of its continuous native global_step_2 | Original h0 |
| harness_only | Common untrained theta0 | Original selector after all5 rounds or native perfect-score stop |
| whale_fst | Canonical export of its independent joint global_step_2 | Same prompt-only harness used in its second training phase |
| whale | Canonical export of its independent joint global_step_2 | Same full harness used in its second training phase |

The seed42 weight-only schedule-preflight deviation remains explicit. Successful
or empty native training is valid if fully audited; require no positive weight
delta or preferred score. Joint conditions must retain their independent first
phase, completed MH search and native second-phase restoration lineage.

All64 test IDs are evaluated once per complete trial with the native reader's
seeded order. Assign four fixed logical shards using `sorted(test_ids)[j::4]`;
preserve native relative order within each shard. Each shard starts a fresh
engine/process, with the trial seed in Python/NumPy/Torch/vLLM and concurrency4.
Generation settings stay temperature1, top_p1, top_k20, thinking enabled,
8129 assistant tokens per puzzle and at most8129 per policy call. Preserve
native default turns9/cap18 and format/illegal retries1. Native retries count
toward usage; no additional best-of, answer repair or resampling is introduced.

Prefer4 H800s,32 CPUs,256GiB for all four shards concurrently. The same logical
shards may run in2 waves on2 H800s or4 waves on1 H800, with8 CPUs/64GiB per active
shard. Select the physical allocation before submission based on availability;
do not change logical assignment, seeds or decoding based on scores. Time caps
are40/80/160 minutes for4/2/1 GPUs, respectively. Estimated wall times are
17–20/34–40/68–80 minutes, about1.13–1.33GPUh per trial in each configuration;
these extrapolate MH timings, not measured test timings. Wider allocations can
queue longer. Four disjoint shard receipts and exact coverage are mandatory.

For each trial retain all actual replies, request accounting, token/mask and
board checks, actual worker weight/seed probes, model/harness hashes and Slurm
terminal status. Live worker probes sample8 coordinates; they do not constitute
full served-parameter inspection. No error, missing shard or unfinished request
can enter a completed score. Preserve interrupted attempts and their costs;
recovery requires the same input identity and cannot be motivated by accuracy.

Secondary measures are actual policy calls and generated tokens per test puzzle,
plus total training/search/evaluation allocation costs reported separately.
Show paired per-seed differences for WHALE/FST against both fixed-weight and
fixed-harness controls. With3 seeds and64 shared puzzles, report the raw results
and descriptive uncertainty; do not claim statistical significance or90% gain
attribution. Never send test tasks, answers or feedback to the GLM proposer, and
never revise selection/training on the basis of this heldout evaluation.
