"""Restore WHALE's Chess prompt-only control without changing upstream files.

The proposer instructions are reconstructed from the published full-harness
contract plus the paper/code restriction. They are not an unpublished original
FST skill. This context only configures the native search; it does not launch a
proposer, choose a provider, change a budget, or perform a weight update.
"""
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch


CONTRACT = Path(__file__).parent / 'prompts/chess_prompt_subspace.md'
PROMPTS = {'SYSTEM_PROMPT', 'USER_PROMPT'}


def validate_literal_prompts(source):
    """Keep the prompt subspace textual, before the native loader executes code.

The native AST comparison masks entire prompt expressions. Requiring literal
strings additionally prevents an expression from changing module/random state.
All natural-language strings remain available, including multiline strings and
Python's adjacent-literal concatenation.
    """
    found = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not PROMPTS.intersection(names):
                continue
            if len(node.targets) != 1 or len(names) != 1:
                raise ValueError('Prompt constants must have one independent target')
            name, value = names[0], node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in PROMPTS:
            annotation = node.annotation
            if not ((isinstance(annotation, ast.Name) and annotation.id == 'str') or
                    (isinstance(annotation, ast.Constant) and annotation.value == 'str')):
                raise ValueError('Prompt annotations must be inert str annotations')
            name, value = node.target.id, node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise ValueError('Prompt values must be literal strings without executable expressions')
        found.append(name)
    if sorted(found) != sorted(PROMPTS):
        raise ValueError('Exactly one SYSTEM_PROMPT and USER_PROMPT definition is required')


@contextmanager
def prompt_subspace(native):
    """Configure native run_evolve/propose_claude within a bounded outer runner.

Callers retain their existing evaluation, GLM budget/isolation and RSFT setup.
The unfinished staged run in mh_phase.py does not call this context. A future
FST phase must explicitly bind these sources in its own frozen plan.
    """
    original_skill = native.SKILL_DIR / 'SKILL.md'
    original_text = original_skill.read_text()
    restriction = CONTRACT.read_text()
    combined = original_text + '\n\n' + restriction
    original_validate = native.validate_candidate

    def validate_candidate(run_dir, name):
        if not native.PROMPT_ONLY:
            raise ValueError('Prompt subspace context was overridden by another mode')
        # Validate both files statically before original load_harness executes.
        for slot in ('h0', name):
            validate_literal_prompts((Path(run_dir) / 'harnesses' / slot / 'harness.py').read_text())
        native.validate_prompt_only(Path(run_dir), name)
        return original_validate(run_dir, name)

    with tempfile.TemporaryDirectory(prefix='whale-fst-contract-') as directory:
        skill = Path(directory) / 'SKILL.md'
        skill.write_text(combined)
        provenance = {
            'kind': 'reconstructed_whale_fst_prompt_control',
            'full_skill_sha256': hashlib.sha256(original_text.encode()).hexdigest(),
            'restriction_sha256': hashlib.sha256(restriction.encode()).hexdigest(),
            'combined_skill_sha256': hashlib.sha256(combined.encode()).hexdigest(),
            'native_ast_validator': True,
            'additional_literal_guard': True,
            'original_unpublished_skill_recovered': False,
        }
        with patch.object(native, 'PROMPT_ONLY', True), \
             patch.object(native, 'PROMPT_ONLY_SKILL_DIR', Path(directory)), \
             patch.object(native, 'validate_candidate', validate_candidate):
            yield provenance
