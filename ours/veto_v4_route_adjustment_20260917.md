# VETO route adjustment after the V3 distinct-selection stop

## Decision

The complete seed42 V3 archive is a negative diagnostic. WHALE, VETO and the
matched point-estimate gate all selected the incoming h0, so no second-stage
training difference can be attributed to VETO. The V3 archive is closed; it
must not be extended with more seeds or continuations to search for a favorable
result.

## What the archive teaches

h1 and h2 show that generic prompt mutations can lower both H and paired C.
h2 is the sharpest example: it converted every task into an operand-subtraction
procedure, scored 0/43 on comparison questions, and produced many truncated or
unparseable numeric answers. h3 preserved task behavior and improved the C point
estimate slightly, but it scored one fewer H answer than h0 and its adjusted
safety intervals crossed zero.

The next research question is therefore narrower:

> Can task-type-aware proposal constraints produce a harness that improves the
> shared H objective without cross-task regressions and still passes a fresh
> paired audit?

## Bounded V4 pilot to preregister before execution

1. Keep the V3 weights, parser, verifier and H objective fixed.
2. Generate at most three candidates with explicit behavior contracts for
   binary comparison, value reading and numeric difference. A candidate may
   share visual extraction steps, but its final-answer rule must dispatch by
   host-provided task type.
3. Because V3 outcomes informed this design, do not reuse V3 C256 as confirmatory
   audit evidence. Freeze a source-disjoint optimization audit or a statistically
   valid reuse rule before proposing candidates. Continue to exclude V/T/R and
   ChartQA from design and selection.
4. Evaluate one seed at fixed weights. Proceed to any RSFT continuation only if
   a candidate both improves H over h0 and passes the frozen safety audit, and
   VETO selects differently from ordinary WHALE or the matched point gate.
5. Stop after the pilot if all decisions remain identical. Do not add seeds to
   compensate for a failed distinct-selection gate.

The proposed cap is one paid proposal, three candidate evaluations, at most
3 GPU hours and the existing project-wide 45 CNY API ceiling. This document is
a route proposal only; it does not authorize or start V4.

## Method and architecture mapping

The task-type contract would sit inside the candidate-harness box before the
existing E1-v3 audit. It does not add a training loss or alter native RSFT.
The paired evaluator remains E1-v3, the uncertainty-aware eligibility test
remains E2-v3, and WHALE ranking remains E3-v3. E4 is reached only after the
new distinct-selection gate passes.
