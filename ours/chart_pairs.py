"""Engineering inputs for E1: rerender an answer-changing bar-height intervention.

This generated fixture suite is not PlotQA, ChartQA or evidence of model quality.
The manifest contains verifier-only labels; a future VLM adapter must send only
the image bytes and question, with no paths, pair IDs or manifest to the model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

from .evidence import VisualPair, fingerprint

BAR_COLOR = (48, 101, 177)
QUESTION = "Which bar is taller, A or B? Answer with A or B only."


def render(values: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", (384, 320), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    draw.line((45, 30, 45, 275, 345, 275), fill=(35, 35, 35), width=2)
    for left, height, name in zip((100, 230), values, ("A", "B")):
        draw.rectangle((left, 274 - height, left + 50, 273), fill=BAR_COLOR)
        draw.text((left + 16, 284), name, font=font, fill=(35, 35, 35))
    return image


def pixel_oracle(image: Image.Image) -> str:
    """Independently count rendered bar-color pixels; no access to scene values."""
    rgb = image.convert("RGB")
    counts = [sum(rgb.getpixel((x, y)) == BAR_COLOR for x in range(left, left + 51)
                  for y in range(30, 274)) for left in (100, 230)]
    if counts[0] == counts[1]:
        raise ValueError("Image is tied or lacks the engineering fixture's bars")
    return "A" if counts[0] > counts[1] else "B"


def generate(output: Path, *, count: int = 8, seed: int = 17) -> dict:
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("Engineering suite supports 1-64 pairs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    rng = random.Random(seed)
    # Sample unique unordered height pairs to avoid duplicate source interventions.
    scenes = rng.sample([(a, b) for a in range(50, 221, 10) for b in range(a + 20, 241, 10)], count)
    manifest, files = [], {}
    for index, values in enumerate(scenes):
        if rng.randrange(2):
            values = values[::-1]
        truths = ("A", "B") if values[0] > values[1] else ("B", "A")
        digests = []
        for side, heights in enumerate((values, values[::-1])):
            name = fingerprint(["engineering-image", seed, index, side]) + ".png"
            path = output / "images" / name
            render(heights).save(path)
            with Image.open(path) as image:
                if pixel_oracle(image) != truths[side]:
                    raise ValueError("Rendered evidence disagrees with the scene oracle")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            digests.append(sha)
            files[sha] = "images/" + name
        pair = VisualPair(fingerprint(["pair", seed, index]), fingerprint(["source", seed, index]),
                          QUESTION, tuple(digests), truths)
        manifest.append(asdict(pair))
    result = {"schema": 1, "role": "engineering", "kind": "rendered_input_fixture",
              "seed": seed, "pillow": pillow_version, "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "pairs": manifest, "audit_data_sha256": fingerprint(manifest), "image_files": files,
              "pixel_oracle_checks": 2 * count, "model_calls": 0,
              "limitation": "No VLM readability or benchmark performance has been evaluated."}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = generate(args.output, count=args.count, seed=args.seed)
    print(json.dumps({"role": report["role"], "pairs": len(report["pairs"]),
                      "pixel_oracle_checks": report["pixel_oracle_checks"], "model_calls": 0,
                      "manifest_sha256": report["audit_data_sha256"]}))
