"""Executable visual harness contract shared by training and evaluation.

This defines the shared E3 search space, not the VETO acceptance rule. Only
visible question/response text, tool arguments and image dimensions enter
callbacks. AST validation is an execution contract, not an OS security sandbox.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
import inspect
import hashlib
import os
from pathlib import Path

from .prompt_subspace import validate_literal_prompts, PROMPTS
from .visual_task import file_sha256

VARIABLE = 'WHALE_VISUAL_HARNESS_PATH'
BASE = Path(__file__).parent / 'visual_harnesses/base_harness.py'
SIGNATURES = {'format_observation': ('question',), 'prepare_tool': ('arguments', 'image_size'),
    'format_feedback': ('text',), 'parse_answer': ('text',), 'nudge': ('text', 'assistant_turns')}


def configured_visual_harness(config):
    execution = config.get('visual_harness_execution', 'in_process')
    if execution == 'isolated':
        from .isolated_visual_harness import load_isolated_visual_harness
        harness = load_isolated_visual_harness(config.get('visual_harness_path'))
    elif execution == 'in_process':
        harness = load_visual_harness(config.get('visual_harness_path'))
    else:
        raise ValueError('Unknown visual callback execution mode')
    protocol = config.get('visual_answer_protocol')
    if protocol is not None:
        from .chart_answer_protocol import PROTOCOL, SharedAnswerHarness
        if protocol != PROTOCOL:
            raise ValueError('Unknown shared visual answer protocol')
        harness = SharedAnswerHarness(harness)
    return harness


def nonprompt_program(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in PROMPTS for t in node.targets):
            node.value = ast.Constant(value='PROMPT')
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in PROMPTS:
            node.value = ast.Constant(value='PROMPT')
    return ast.dump(tree, include_attributes=False)


@dataclass(frozen=True)
class VisualHarness:
    path: Path
    sha256: str
    source: str
    namespace: dict

    def unchanged(self):
        if file_sha256(self.path) != self.sha256:
            raise ValueError('Visual harness changed after construction')

    def invoke(self, name, **visible):
        self.unchanged()
        if name not in SIGNATURES or set(visible) != set(SIGNATURES[name]):
            raise ValueError('Unexpected visual callback inputs')
        return self.namespace[name](**deepcopy(visible))

    @property
    def system_prompt(self):
        return self.namespace['SYSTEM_PROMPT']


def load_visual_harness(path=None, *, prompt_reference=None):
    from autoharness_chess_puzzle.harness import _SafetyVisitor, SAFE_BUILTINS, _safe_import

    path = Path(path or os.environ.get(VARIABLE) or BASE).resolve()
    raw_source = path.read_bytes()
    source = raw_source.decode('utf-8')
    validate_literal_prompts(source)
    tree = ast.parse(source)
    class VisualContractVisitor(_SafetyVisitor):
        def visit_Import(self, node):
            if any(alias.name not in {'math', 're', 'statistics'} or (alias.asname or '').startswith('_') for alias in node.names):
                raise ValueError('Visual callbacks support only declared utility imports')
            super().visit_Import(node)
        def visit_ImportFrom(self, node):
            if node.module not in {'math', 're', 'statistics'} or node.level or any(
                    alias.name.startswith('_') or (alias.asname or '').startswith('_') or alias.name == '*' for alias in node.names):
                raise ValueError('Visual callbacks support only declared utility imports')
            super().visit_ImportFrom(node)
        def visit_Attribute(self, node):
            if node.attr.startswith('_') or isinstance(node.ctx, (ast.Store, ast.Del)):
                raise ValueError('Private or mutable attributes are outside the visual harness contract')
            super().visit_Attribute(node)
        def visit_Name(self, node):
            if node.id.startswith('_'):
                raise ValueError('Private names are outside the visual harness contract')
        def visit_Global(self, node):
            raise ValueError('Cross-trajectory global state is outside the visual harness contract')
        visit_Nonlocal = visit_Global
    VisualContractVisitor().visit(tree)
    def immutable(value):
        return isinstance(value, (str, int, float, bool, type(None))) or (
            isinstance(value, tuple) and all(immutable(item) for item in value))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if isinstance(node, ast.AnnAssign) and (not isinstance(node.target, ast.Name) or node.target.id not in PROMPTS):
                raise ValueError('Only prompt constants support annotations')
            value = ast.literal_eval(node.value)
            if not immutable(value):
                raise ValueError('Module constants must be immutable literals')
            continue
        if isinstance(node, ast.FunctionDef) and not node.decorator_list:
            for default in (*node.args.defaults, *[v for v in node.args.kw_defaults if v is not None]):
                if not immutable(ast.literal_eval(default)):
                    raise ValueError('Callback defaults must be immutable literals')
            if node.returns is not None or any(arg.annotation is not None for arg in
                    (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                     *([node.args.vararg] if node.args.vararg else []), *([node.args.kwarg] if node.args.kwarg else []))):
                raise ValueError('Visual callback declarations require inert unannotated signatures')
            continue
        raise ValueError('Visual harness module initialization must be declarative')
    if prompt_reference is not None:
        reference = Path(prompt_reference).read_text()
        validate_literal_prompts(reference)
        if nonprompt_program(source) != nonprompt_program(reference):
            raise ValueError('Prompt-only visual candidate changed executable code')
    namespace = {'__builtins__': {**{k: v for k, v in SAFE_BUILTINS.items() if k != 'print'}, '__import__': _safe_import}}
    exec(compile(tree, str(path), 'exec'), namespace)
    for name, parameters in SIGNATURES.items():
        callback = namespace.get(name)
        if not callable(callback):
            raise ValueError(f'Missing visual callback: {name}')
        inspect.signature(callback).bind(**{parameter: None for parameter in parameters})
    harness = VisualHarness(path, hashlib.sha256(raw_source).hexdigest(), source, namespace)
    harness.unchanged()
    return harness
