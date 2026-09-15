# Compact chart harness search contract

Improve ordinary accuracy using only the supplied H feedback and incoming h0.
Produce up to the three allocated standalone candidates h1,h2,h3 in one round.
Every allocated slot counts even if missing, invalid, or unable to execute.
Do not create replacement slots or request another search round.

The shared task includes comparisons (A/B), reading one plotted value, and
differences between two values. Images enter the model separately. Preserve
literal SYSTEM_PROMPT and USER_PROMPT, and the callback signatures
format_observation(question), prepare_tool(arguments,image_size),
format_feedback(text), parse_answer(text), nudge(text,assistant_turns).

The host owns final-answer parsing and scoring. The candidate parse_answer
callback is retained for interface compatibility but is never used. Improve
general instructions, question formatting or bounded recovery; do not change
answers, invent tools, or embed item-specific labels. No crop tool is enabled
in this common configuration. At most three policy calls and1024 total
assistant tokens are allowed per question across all recovery turns.

Only math,re,statistics utility imports are allowed. No files, networking,
private attributes, hidden metadata, mutable global state, shell, or extra
dependencies. Callbacks run in fresh processes with OS filesystem/network
confinement. Do not read any optimization audit or development/test dataset.

Write harnesses/h1/harness.py through harnesses/h3/harness.py, plus
pending_eval.json containing {"candidates":[{"name":"h1","parent":"h0",
"axis":"...","hypothesis":"...","changes":"..."},...]}. Only allocated
candidate code, this metadata and logs/iteration_001/report.md may be written.
Existing evidence must remain unchanged.
