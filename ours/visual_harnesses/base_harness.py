"""Shared initial visual harness; not an optimized or validated strong baseline."""

SYSTEM_PROMPT = "Answer the user's question using the supplied evidence. Return only the requested answer."
USER_PROMPT = "{question}"


def format_observation(question):
    return USER_PROMPT.format(question=question)


def prepare_tool(arguments, image_size):
    return dict(arguments)


def format_feedback(text):
    return text


def parse_answer(text):
    return text


def nudge(text, assistant_turns):
    return None
