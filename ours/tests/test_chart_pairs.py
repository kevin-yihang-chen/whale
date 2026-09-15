import hashlib
import json

import pytest
from PIL import Image

from ours.chart_pairs import generate, pixel_oracle
from ours.evidence import fingerprint


def test_saved_images_independently_support_opposite_answers(tmp_path):
    output = tmp_path / "suite"
    report = generate(output, count=3)
    assert report["role"] == "engineering"
    assert report["audit_data_sha256"] == fingerprint(report["pairs"])
    for pair in report["pairs"]:
        observed = []
        for sha in pair["images_sha256"]:
            path = output / report["image_files"][sha]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
            with Image.open(path) as image:
                assert not image.info  # No answer-bearing image metadata.
                observed.append(pixel_oracle(image))
        assert tuple(observed) == tuple(pair["answers"])
        assert observed[0] != observed[1]
    loaded = json.loads((output / "manifest.json").read_text())
    assert loaded["audit_data_sha256"] == fingerprint(loaded["pairs"])
    with pytest.raises(FileExistsError):
        generate(output, count=3)


def test_generation_reproducible_within_pinned_renderer(tmp_path):
    first = generate(tmp_path / "first", count=2)
    second = generate(tmp_path / "second", count=2)
    assert first == second
