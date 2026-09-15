"""Shared answer-format calibration; no image, label, or private-data access."""
import re

SYSTEM_PROMPT = "Answer the user's question using the supplied evidence. Return only the requested answer."
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
    return None
