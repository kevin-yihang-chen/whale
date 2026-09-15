"""Candidate h2: careful numeric reading procedure for value/difference questions."""
import re

SYSTEM_PROMPT = (
    'Read the chart carefully. Procedure: (1) locate the requested categories and '
    'series; (2) read axis scale, units, and gridline spacing before estimating '
    'values; (3) align each bar/point with the nearest axis tick and interpolate '
    'to the nearest sensible increment; (4) for differences, subtract in the '
    'order the question asks; (5) for A/B comparisons, check both options '
    'against the chart before deciding. Put only the requested answer after '
    '"Final answer:" on the last line.'
)
USER_PROMPT = "{question}"


def format_observation(question):
    return USER_PROMPT.format(question=question)


def prepare_tool(arguments, image_size):
    return dict(arguments)


def format_feedback(text):
    return text


def parse_answer(text):
    if any(marker in text.lower() for marker in ('<tool_call', '</tool_call', '<function=')):
        return text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    answers = [line for line in lines if re.fullmatch(r'[AB][.!]?', line, re.IGNORECASE)]
    if len(answers) != 1 or answers[0] != lines[-1]:
        return text
    return lines[-1][0].upper()


def nudge(text, assistant_turns):
    if text and 'final answer' in text.lower():
        return None
    return (
        'Before finishing, re-check the axis scale and the alignment of the '
        'relevant bars/points with the nearest tick. Then reply with exactly '
        'one last line: "Final answer: <answer>".'
    )
