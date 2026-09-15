"""Local target inference for the released Chess runner's completion contract.

This is a new, declared compatibility implementation, not recovered TextArena
source or a replacement for WHALE's trainer. It contributes no Method equation.
Every request is bound to the loaded checkpoint and saved with actual token use.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math
import os
from pathlib import Path
import threading
import time

from .evidence import fingerprint
from .visual_task import file_sha256


@dataclass
class LLMConfig:
    model: str
    provider: str = "local-transformers"
    max_tokens: int = 8129
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 20
    batch_concurrency: int = 1
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": True})
    seed: int = 42
    trace_path: str | None = None
    expected_weights_sha256: str | None = None

    def __post_init__(self):
        if self.provider != "local-transformers":
            raise ValueError("This compatibility client requires provider=local-transformers")
        if self.batch_concurrency != 1:
            raise ValueError("This serial local client requires batch_concurrency=1")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self):
        if self.role not in {"system", "user", "assistant"} or not isinstance(self.content, str):
            raise ValueError("Expected a text system/user/assistant message")


@dataclass(frozen=True)
class CompletionResponse:
    content: str
    usage: dict[str, int]


def checkpoint_manifest(path: Path) -> dict:
    """Hash all loaded weight and tokenizer/config assets, not the directory name."""
    path = path.resolve()
    weights = sorted(path.glob("*.safetensors"))
    if not weights or not (path / "config.json").is_file():
        raise ValueError("An existing local HF safetensors checkpoint is required")
    weight_hashes = {p.name: file_sha256(p) for p in weights}
    assets = {p.name: file_sha256(p) for p in sorted(path.iterdir())
              if p.is_file() and p.suffix in {".json", ".jinja", ".txt"}}
    if (path / "model.safetensors.index.json").is_file():
        index = json.loads((path / "model.safetensors.index.json").read_text())
        if set(index["weight_map"].values()) != set(weight_hashes):
            raise ValueError("Checkpoint shard coverage disagrees with the HF index")
    identity = {"weights": weight_hashes, "assets": assets}
    return {"path": str(path), **identity, "weights_sha256": fingerprint(identity)}


class LLMClient:
    """Implement only the interface actually consumed by the pinned Chess runner.

    Load a fresh model for each client, reject an unexpected checkpoint identity,
    and expose a raw request ledger. No remote endpoint or fallback is used.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.manifest = checkpoint_manifest(Path(config.model))
        if config.expected_weights_sha256 and config.expected_weights_sha256 != self.manifest["weights_sha256"]:
            raise ValueError("Loaded checkpoint content differs from the required phase identity")
        raw_config = json.loads((Path(self.manifest["path"]) / "config.json").read_text())
        if raw_config.get("model_type") != "qwen3_5":
            raise ValueError("This baseline adapter currently supports Qwen3.5 only")
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            os.environ.pop(key, None)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("Allocate a CUDA GPU before constructing the local target")
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        self.processor = AutoProcessor.from_pretrained(self.manifest["path"], local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.manifest["path"], dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True).to("cuda").eval()
        self.lock = threading.Lock()
        self.calls = 0
        self.ledger = None
        if config.trace_path:
            trace = Path(config.trace_path)
            trace.parent.mkdir(parents=True, exist_ok=True)
            self.ledger = trace.open("x")

    def complete_response(self, messages: list[ChatMessage], *, max_tokens: int | None = None) -> CompletionResponse:
        limit = self.config.max_tokens if max_tokens is None else max_tokens
        if type(limit) is not int or limit <= 0:
            raise ValueError("The per-call token limit must be a positive integer")
        if not messages or any(not isinstance(m, ChatMessage) for m in messages):
            raise ValueError("Expected a nonempty list of ChatMessage instances")
        torch = self.torch
        with self.lock:
            payload = [asdict(message) for message in messages]
            prompt = self.processor.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True, **self.config.chat_template_kwargs)
            inputs = self.processor(text=[prompt], return_tensors="pt").to("cuda")
            input_length = int(inputs["input_ids"].shape[-1])
            sampling = {"do_sample": self.config.temperature > 0, "max_new_tokens": limit,
                        "use_cache": True, "pad_token_id": self.processor.tokenizer.pad_token_id}
            if sampling["do_sample"]:
                sampling.update(temperature=self.config.temperature, top_p=self.config.top_p,
                                top_k=self.config.top_k)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(**inputs, **sampling)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            tokens = output[0, input_length:]
            content = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            usage = {"prompt_tokens": input_length, "completion_tokens": int(tokens.numel()),
                     "total_tokens": input_length + int(tokens.numel())}
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            finish = "eos" if tokens.numel() and int(tokens[-1]) in eos else "length"
            self.calls += 1
            if self.ledger:
                record = {"call": self.calls, "messages": payload, "raw_answer": content,
                          "usage": usage, "finish": finish, "seconds": seconds,
                          "weights_sha256": self.manifest["weights_sha256"],
                          "input_ids_sha256": fingerprint(inputs["input_ids"].tolist()),
                          "output_ids": tokens.tolist(), "sampling": sampling,
                          "chat_template_kwargs": self.config.chat_template_kwargs,
                          "client_source_sha256": file_sha256(Path(__file__))}
                self.ledger.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.ledger.flush()
            return CompletionResponse(content, usage)

    def close(self):
        if self.ledger:
            self.ledger.close()
            self.ledger = None
