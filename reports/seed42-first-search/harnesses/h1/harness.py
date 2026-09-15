"""Candidate h1: stronger answer-format discipline in the prompt."""
import re

SYSTEM_PROMPT = (
    'You are answering chart questions. First identify the requested categories '
    'and series, check axes and units, then compute the answer (a letter A or B, '
    'a plotted value, or a difference of two values). Keep reasoning brief. '
    'You MUST end your reply with exactly one final line of the form '
    '"Final answer: <answer>" where <answer> is only the answer itself '
    '(e.g. "Final answer: A" or "Final answer: 42.5"). Do not put units, '
    'sentences, or extra numbers on that line.'
)
USER_PROMPT = "{question}\n\nEnd with one line: Final answer: <answer>."


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
        'You have not given a final line yet. Reply now with exactly one line: '
        '"Final answer: <answer>" containing only the answer itself.'
    )
