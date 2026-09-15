"""Shared chart answer semantics for E1 evaluation and native E4 rewards.

Candidates never receive verifier metadata and cannot replace this parser.
Numeric truth uses chart-axis units and a five-percent relative tolerance;
zero has an explicit absolute tolerance. These are derived-task semantics,
not a claim to reproduce the official ChartQA evaluator.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import re

from .evidence import fingerprint
from .visual_task import file_sha256

PROTOCOL = 'chart_answer_v1'
DATA_SOURCE = 'visual_chart_shared_v1'
RELATIVE_TOLERANCE = Decimal('0.05')
ZERO_TOLERANCE = Decimal('0.000000001')
NUMBER = re.compile(r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+|\d*\.\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?')


def number(value):
    if not isinstance(value, str) or not NUMBER.fullmatch(value.strip()):
        return None
    try:
        result = Decimal(value.strip().replace(',', ''))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def decimal_text(value):
    value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError('Nonfinite chart value')
    return '0' if value == 0 else format(value.normalize(), 'f')


def encode_truth(kind, value):
    if kind not in {'binary', 'numeric'}:
        raise ValueError('Unknown derived chart answer kind')
    value = str(value) if kind == 'binary' else decimal_text(value)
    if kind == 'binary' and value not in {'A', 'B'}:
        raise ValueError('Binary truth must be A or B')
    return json.dumps({'protocol': PROTOCOL, 'kind': kind, 'value': value}, sort_keys=True, separators=(',', ':'))


def decode_truth(truth):
    record = json.loads(truth)
    if set(record) != {'protocol', 'kind', 'value'} or record['protocol'] != PROTOCOL:
        raise ValueError('Unknown chart answer protocol')
    if encode_truth(record['kind'], record['value']) != truth:
        raise ValueError('Truth is not canonical')
    return record


def parse_answer(text):
    """Extract one final answer, without seeing its type or ground truth."""
    if not isinstance(text, str):
        raise TypeError('Expected decoded answer text')
    if any(marker in text.lower() for marker in ('<tool_call', '</tool_call', '<function=')):
        return text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ''
    last = lines[-1]
    explicit = re.fullmatch(r'(?:final\s+answer|answer)\s*:\s*(.+)', last, re.IGNORECASE)
    if explicit:
        last = explicit.group(1).strip()
    boxed = re.fullmatch(r'\\boxed\{([^{}]+)\}', last)
    if boxed:
        last = boxed.group(1).strip()
    if re.fullmatch(r'[AB][.!]?', last, re.IGNORECASE):
        return last[0].upper()
    value = number(last)
    if value is not None:
        return decimal_text(value)
    # Open-vocabulary ChartQA answers use the same extraction boundary.
    return last if explicit or len(lines) == 1 else text.strip()


def tolerance(value):
    return abs(value) * RELATIVE_TOLERANCE if value else ZERO_TOLERANCE


def disjoint_numeric_answers(first, second):
    a, b = Decimal(str(first)), Decimal(str(second))
    return abs(a - b) > tolerance(a) + tolerance(b)


def verify(prediction, truth):
    record = decode_truth(truth)  # Invalid labels are data errors, never misses.
    if record['kind'] == 'binary':
        return prediction == record['value']
    actual, expected = number(prediction), Decimal(record['value'])
    return actual is not None and abs(actual - expected) <= tolerance(expected)


def identity():
    from pathlib import Path
    return fingerprint({'protocol': PROTOCOL, 'source_sha256': file_sha256(Path(__file__)),
                        'relative_tolerance': str(RELATIVE_TOLERANCE), 'zero_tolerance': str(ZERO_TOLERANCE)})


@dataclass(frozen=True)
class SharedAnswerHarness:
    """Keep executable callbacks isolated; answer commitment is host-owned."""
    harness: object

    def __getattr__(self, name):
        return getattr(self.harness, name)

    def invoke(self, name, **visible):
        self.harness.unchanged()
        if name == 'parse_answer':
            if set(visible) != {'text'}:
                raise ValueError('Unexpected answer parser input')
            return parse_answer(visible['text'])
        return self.harness.invoke(name, **visible)
