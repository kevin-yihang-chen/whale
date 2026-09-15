# WHALE-FST prompt subspace restriction

This run is WHALE's prompt-restricted Fast-Slow control. The original
prompt-only skill was not included in the pinned public release. These
additional instructions reconstruct its documented search boundary; they are
not a recovered author file or the complete algorithm of the separate FST paper.

The preceding full-harness contract still defines the task, information access,
anti-cheating rules, required files and reporting. For this control, its editable
surface is narrowed as follows; permissions to change other harness components
do not apply in this run.

- Read the current accepted harness and available trajectory evidence.
- Write a complete candidate by copying h0 and editing only the literal text
  assigned to SYSTEM_PROMPT and USER_PROMPT. Strong, task-specific instructions
  are allowed; there is no artificial restriction to punctuation or short edits.
- Use plain string literals, including multiline or adjacent literals. Do not
  use function calls, f-string expressions, assignments inside expressions,
  computed strings, or executable annotations to initialize those constants.
- Retain the observation placeholder according to the task interface. Never
  insert exact evaluation examples, FENs, puzzle IDs, or reference solutions.
- Keep the parser, observation formatter, legal-move checker, proposed-action
  function, imports, helper functions, retry budgets and MAX_TURNS unchanged.
- Comments and docstrings may change as allowed by the original AST comparison.
- Preserve the same requested candidate slots and metadata output schema.

The outer experiment retains WHALE's RSFT update, verifier, sampling parameters,
evaluation data and schedule. This contract changes only the harness search
space; it does not authorize extra evaluation, model calls, or budget changes.
