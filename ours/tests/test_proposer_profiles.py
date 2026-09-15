import json

import pytest

from ours.proposer_profiles import GLM_FLASH, ProposerProfile


def test_profile_is_process_local_and_does_not_expose_credentials():
    env = {"ZAI_API_KEY": "unit-test-not-a-real-secret", "ANTHROPIC_API_KEY": "old-test-provider"}
    result = GLM_FLASH.claude_environment(env)
    assert env["ANTHROPIC_API_KEY"] == "old-test-provider"
    assert "ANTHROPIC_API_KEY" not in result
    assert result["ANTHROPIC_AUTH_TOKEN"] == env["ZAI_API_KEY"]
    assert result["PROPOSER_MODEL"] == "glm-5.3-flash"
    assert env["ZAI_API_KEY"] not in json.dumps(GLM_FLASH.public_description(env))


def test_absent_key_does_not_fall_back_to_another_paid_provider():
    with pytest.raises(ValueError, match="credential is absent"):
        GLM_FLASH.claude_environment({"ANTHROPIC_API_KEY": "not-a-real-key"})


def test_local_openai_api_is_not_misrepresented_as_claude_protocol():
    local = ProposerProfile("served-model", "http://127.0.0.1:8000/v1", None, "openai_chat")
    with pytest.raises(ValueError, match="tool-agent adapter"):
        local.claude_environment({})


@pytest.mark.parametrize("url", ["https://user:password@example.com", "https://example.com?api_key=secret",
                                 "http://remote.example.com/v1", "file:///tmp/server"])
def test_profile_rejects_credential_bearing_or_invalid_endpoint(url):
    with pytest.raises(ValueError):
        ProposerProfile("model", url, "TEST_KEY", "anthropic")
