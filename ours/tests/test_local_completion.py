"""Checkpoint-integrity tests for the shared baseline inference client."""

import json

import pytest

from ours.local_completion import LLMClient, LLMConfig, checkpoint_manifest


def checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (tmp_path / "model.safetensors").write_bytes(b"integrity-test-fixture-not-model-weights")
    (tmp_path / "tokenizer.json").write_text('{"version":1}')
    return checkpoint_manifest(tmp_path)


@pytest.mark.parametrize("filename,changed", [
    ("model.safetensors", b"changed-weight-content"),
    ("tokenizer.json", b'{"version":2}'),
])
def test_stale_phase_identity_rejected_before_loading_model(tmp_path, filename, changed):
    original = checkpoint(tmp_path)
    (tmp_path / filename).write_bytes(changed)
    with pytest.raises(ValueError, match="phase identity"):
        LLMClient(LLMConfig(model=str(tmp_path), expected_weights_sha256=original["weights_sha256"]))


def test_partial_checkpoint_rejected(tmp_path):
    checkpoint(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"layer0": "model.safetensors", "layer1": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="shard coverage"):
        checkpoint_manifest(tmp_path)


def test_remote_provider_never_silently_replaced_by_local_model():
    with pytest.raises(ValueError, match="provider=local-transformers"):
        LLMConfig(model="some-model", provider="openai-compatible")
