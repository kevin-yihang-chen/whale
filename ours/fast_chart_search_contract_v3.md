# VETO-v3 chart harness search contract

Improve ordinary H accuracy from the incoming h0 while addressing the supplied,
observable H-pair failure patterns. Produce the three allocated standalone
candidates h1, h2 and h3 in one round. Every slot counts even if missing,
invalid or unable to execute; do not create replacement slots or another round.

The slots test distinct mechanisms:

- h1: `visual_recheck` — change how the policy verifies the requested
  category, series, axis or operands before committing an answer.
- h2: `reasoning_decomposition` — change the bounded read/compare/compute
  procedure while retaining the same visual input and host scorer.
- h3: `response_control` — change bounded recovery, answer commitment or
  formatting behavior without changing the host parser or verifier.

Do not use three paraphrases of the same prompt mechanism. `pending_eval.json`
must contain, for every slot, `name`, `parent`, `mechanism`, `hypothesis`,
`change`, `predicted_failure_addressed`, and `expected_cost` with integer
`extra_policy_calls` and `extra_tool_calls`. The declared mechanism must match
the slot above.

Preserve literal SYSTEM_PROMPT and USER_PROMPT and the callback signatures
format_observation(question), prepare_tool(arguments,image_size),
format_feedback(text), parse_answer(text), nudge(text,assistant_turns). The host
owns final-answer parsing and scoring; the candidate parser cannot change them.
At most three policy calls and 1024 generated assistant tokens are allowed per
question. The common V3 configuration does not expose a crop tool because the
existing development screen did not support it; `extra_tool_calls` must be zero.

Only math, re and statistics utility imports are allowed. No files, networking,
private attributes, hidden metadata, mutable global state, shell or extra
dependencies. Callbacks run in fresh processes with OS filesystem/network
confinement. H and H-pair are proposal data. C is private selection data and
V/T/R are excluded from proposal generation.

Write only harnesses/h1/harness.py through harnesses/h3/harness.py,
pending_eval.json and logs/iteration_001/report.md. Existing evidence must
remain unchanged.
