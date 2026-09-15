"""Shared visual input adapter feeding Method E1; not a new optimization block."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from PIL import Image

SYSTEM_PROMPT = "Answer the user's question using the supplied evidence. Return only the requested answer."


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_answer_verifier(prediction: str, truth: str) -> bool:
    """Frozen engineering verifier: one A/B label, optional terminal punctuation."""
    match = re.fullmatch(r"\s*([AB])\s*[.!]?\s*", prediction, flags=re.IGNORECASE)
    return match is not None and match.group(1).upper() == truth


@dataclass(frozen=True)
class VisualTaskAdapter:
    """Provide only image pixels and question to the target processor.

    The caller retains pair IDs, answers, source metadata and file paths outside
    the messages. This adapter does not provide a candidate-execution sandbox.
    """

    min_pixels: int = 64 * 28 * 28
    max_pixels: int = 512 * 28 * 28

    def prepare(self, processor, *, question: str, image: Image.Image | None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The model requires a nonempty question")
        content = []
        if image is not None:
            content.append({"type": "image", "image": image.convert("RGB")})
        content.append({"type": "text", "text": question})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [prompt], "return_tensors": "pt", "padding": True}
        if image is not None:
            kwargs["images"] = [image.convert("RGB")]
        return processor(**kwargs), prompt
