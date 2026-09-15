"""Provider configuration for the shared proposer, outside Method E1-E4.

GLM can retain WHALE's Claude Code tool interface using Z.ai's documented
Anthropic-compatible endpoint. No HTTP calls, credential persistence, installation
or changes to ~/.claude are performed by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from typing import Mapping
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ProposerProfile:
    model: str
    base_url: str
    key_env: str | None
    transport: str

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Endpoint must not contain credentials, query strings or fragments")
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("Invalid model endpoint")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Plain HTTP profile is limited to a loopback model server")
        if self.transport not in {"anthropic", "openai_chat"} or not self.model:
            raise ValueError("Invalid model or transport")

    def claude_environment(self, inherited: Mapping[str, str]) -> dict[str, str]:
        """Build a process-local environment for the existing WHALE CLI wrapper.

        Pass model explicitly as --proposer-model; merely changing model alias
        environment variables does not override upstream's explicit model flag.
        """
        if self.transport != "anthropic":
            raise ValueError("OpenAI-chat local server needs a tool-agent adapter; it is not a Claude endpoint")
        if self.key_env is None or not inherited.get(self.key_env):
            raise ValueError("Required proposer credential is absent from the runtime environment")
        env = dict(inherited)
        env.pop("ANTHROPIC_API_KEY", None)
        env["ANTHROPIC_BASE_URL"] = self.base_url
        env["ANTHROPIC_AUTH_TOKEN"] = inherited[self.key_env]
        env["PROPOSER_MODEL"] = self.model
        for size in ("OPUS", "SONNET", "HAIKU"):
            env[f"ANTHROPIC_DEFAULT_{size}_MODEL"] = self.model
        return env

    def public_description(self, inherited: Mapping[str, str]) -> dict:
        return {"model": self.model, "base_url": self.base_url, "transport": self.transport,
                "credential_variable": self.key_env,
                "credential_present": bool(self.key_env and inherited.get(self.key_env)),
                "status": "CONFIGURATION_ONLY_NOT_CONNECTED"}


GLM_FLASH = ProposerProfile("glm-5.3-flash", "https://api.z.ai/api/anthropic", "ZAI_API_KEY", "anthropic")
GLM_FLASH_CN = ProposerProfile("glm-5.3-flash", "https://open.bigmodel.cn/api/anthropic", "ZHIPU_API_KEY", "anthropic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["glm_flash", "glm_flash_cn", "local"], default="glm_flash")
    parser.add_argument("--model", help="Required served model ID for the local OpenAI-chat profile")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    if args.profile == "local" and not args.model:
        parser.error("--model is required for the local profile")
    profiles = {"glm_flash": GLM_FLASH, "glm_flash_cn": GLM_FLASH_CN}
    profile = profiles[args.profile] if args.profile in profiles else ProposerProfile(args.model, args.base_url, None, "openai_chat")
    print(json.dumps(profile.public_description(os.environ), indent=2))
