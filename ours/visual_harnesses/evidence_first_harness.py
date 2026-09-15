"""Shared answer-format calibration; no image, label, or private-data access."""
import re

SYSTEM_PROMPT = "Inspect the full chart and answer the user's question. Answer directly from the full image whenever possible. Use the crop tool only when needed to read a detail. Make at most one crop call, then give your final answer. Do not keep requesting crops. Your final response must be exactly A or B, without explanation."
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
