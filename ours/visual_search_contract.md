# Visual harness search contract

Optimize ordinary accuracy on the supplied H search feedback for a fixed visual
model. H contains public chart questions, model answers and their verification;
it is optimization data. No other dataset is available to you.

Read `harnesses/h0/harness.py` and `logs/H/h0/qwen35-4b/feedback.json`.
Write one complete candidate in the allocated slot. Preserve five functions and
their exact argument names: `format_observation(question)`,
`prepare_tool(arguments, image_size)`, `format_feedback(text)`,
`parse_answer(text)`, and `nudge(text, assistant_turns)`. SYSTEM_PROMPT and
USER_PROMPT must be literal strings. The image is supplied separately by the
shared evaluator; callbacks cannot read it, metadata or any files.

The current condition has no crop tool. You may improve instructions, visible
question formatting, final-answer parsing and bounded recovery of malformed
answers. Any nudge shares the existing three-call/1024-token total per question.
Do not add tools, change budgets, read files, use a network, identify benchmark
items, encode question-specific answers, or infer hidden labels. Do not change
a valid A or B answer to a different label. Ambiguous or unfinished tool answers
must remain invalid. A comparison result is not a performance guarantee.

Only `math`, `re`, `statistics` utility imports are allowed. No mutable module
state, private attributes, decorators, annotations, IO, random state or hidden
dependencies. All functions execute in separate filesystem/network-restricted
processes. Copy the whole standalone module; do not import another harness.

Write `pending_eval.json` as
`{"candidates":[{"name":"h1","parent":"h0","axis":"...","hypothesis":"...","changes":"..."}]}`.
Allowed writes are the allocated harness, pending_eval.json, and optionally
logs/iteration_001/report.md. Never alter existing evidence. No shell tools.
