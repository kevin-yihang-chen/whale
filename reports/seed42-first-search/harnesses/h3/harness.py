"""Candidate h3: brief structured reasoning template plus format-strict final line."""
import re

SYSTEM_PROMPT = (
    'Answer chart questions with this short structure:\n'
    '1. Target: what is asked (category/series, value, comparison, or difference).\n'
    '2. Read: axis scale and units; the plotted value(s) needed.\n'
    '3. Answer: compute/compare and give the result.\n'
    'Keep each step to one short line. End with exactly one last line of the '
    'form "Final answer: <answer>" where <answer> is only the answer itself, '
    'with no units or extra words.'
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
    if not assistant_turns:
        return (
            'Complete the structure (Target, Read, Answer) briefly, then end '
            'with one line: "Final answer: <answer>".'
        )
    return 'Reply with only one line now: "Final answer: <answer>".'
