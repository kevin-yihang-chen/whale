# Controlled Chess pilot, fixed before fresh trial sampling

This pilot prepares F0 evidence; it does not redefine paper-scale reproduction or
replace the visual research gates. Common theta0 is the fully audited untrained
BF16 checkpoint from222797. Use the existing disjoint train128/MH32/test64 split,
all seeds42/43/44, native h0, exact verifier and shared execution fixes.

| Condition | Weight budget | Search budget | Phase behavior |
|---|---|---|---|
| weight_only | 2 native batches,8 prompts x8 trajectories each | None | One continuous native trainer under h0 |
| harness_only | No update, common theta0 fixed | 5 iterations x3 candidate slots,32 MH cases | Original full harness search |
| whale_fst | 1 native batch, then search, then 1 native batch | 5 iterations x3 candidate slots,32 MH cases | Native phase resume, prompt subspace only |
| whale | 1 native batch, then search, then 1 native batch | 5 iterations x3 candidate slots,32 MH cases | Native phase resume, full harness search |

Each enabled search retains the released candidate count and five iterations;
native perfect-score stopping and invalid candidates are recorded. GLM replaces
the released proposer, under the user's existing reviewed-MH-data authorization
and global45CNY budget. Never supply training or test answers to the proposer.
No candidate is manually weakened or discarded based on a desired ranking.

The two native batch steps bound128 fresh training trajectories and1,040,512
assistant-output tokens per weight-training condition. They do not promise two
optimizer steps: the original success filter may yield zero or multiple updates.
Record every failed/empty phase and its costs. Preserve full trajectories and
native parameter checkpoints for audit. Do not retry sampling to obtain a better
score. Runtime failures may be recovered only with preserved input identity.

The published launcher saves model/extra, including RNG and scheduler, and its
trainer saves dataloader state; Adam moments are excluded. Native alternating
phase resume must retain that behavior and its distinction from continuous
weight-only training. Do not pool a prefix if it changes state continuation.
Any phase-reset weight-only sensitivity control must be separately named.

Only complete train/MH runs enter the eventual heldout comparison, with the
evaluation protocol frozen before opening the test examples. Report all three
seeds and mean/sample standard deviation, task success, actual calls/tokens and
allocation costs. Optimization-set wins are not heldout gains. Unfinished runs
remain unfinished, and this small pilot does not establish a VETO contribution.

Released scale differs: train16384, batch256, MH256,256 total native batches and
13 batches per weight phase. Pilot budgets, GLM substitution, precision control,
backend repairs and hardware must be disclosed in any eventual paper comparison.
