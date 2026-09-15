# Released Chess selection and retry semantics

The local pinned implementation is the authority for this reproduction. The
read-only check in `results/native-frontier-retry-semantics-20260910.json`
records both behaviors below without changing upstream or the running search.

`chess_puzzle_benchmark.compute_pareto` sorts by descending success and ascending
calls, then keeps a point when `turns <= best_turns`. Thus a lower-score point
with equal calls remains marked on the native frontier. Across the completed
first three rounds, the native list is h1,h6,h5,h8,h9,h4,h7,h2, whereas strict
nondominance would retain h1 and h6. The native `_best` independently sorts all
points by score then calls; its accepted h1 is consistent with the full scores.
Preserve this native behavior and proposer feedback for reproduction. Label the
recorded flags as native frontier membership, not proof of strict nondominance.
The interpretation check is not an alternative selector or another experiment.

`runner.retry_budgets` treats the environment values as defaults and gives a
selected harness's declared retry attributes precedence. For the real h9,
environment defaults (1,1) yield effective format/illegal budgets (1,2).
The heldout protocol's retry values specify those shared environment defaults;
the frozen final harness retains its native attributes. Likewise the shared
turn cap is18 while the original runner resolves the harness's effective turns.
Do not overwrite a selected harness at test time to force its retries to1.
Record actual effective budgets and all retry requests in the final audit.

These are implementation semantics, not new optimization mechanisms. The
read-only check made no new model/API calls and loaded no heldout tasks.
