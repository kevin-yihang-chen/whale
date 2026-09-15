"""Check response accounting and local endpoint boundaries without model calls."""
import json
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from ours.local_completion import ChatMessage
from ours.vllm_runtime import RecordedVLLMClient, VLLMConfiguration


class VLLMRuntimeTests(unittest.TestCase):
    def test_endpoint_rejects_remote_and_ambiguous_origins(self):
        for endpoint in ("https://127.0.0.1", "http://example.com", "http://127.0.0.1/x",
                         "http://user@127.0.0.1", "http://127.0.0.1?x=1"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                VLLMConfiguration(model="fixture", base_url=endpoint)

    def test_raw_reasoning_is_not_rewritten(self):
        self._response_test(2, "<think>analysis</think><move>a1a2</move>", False)

    def test_inconsistent_usage_is_preserved_then_rejected(self):
        self._response_test(3, "fixture", True)

    def test_requests_overlap_but_ledger_has_unique_call_numbers(self):
        barrier = threading.Barrier(2)
        def respond(*args, **kwargs):
            barrier.wait(timeout=2)
            return {"model": "fixture", "choices": [{"message": {"content": "fixture"}, "token_ids": [2]}],
                    "prompt_token_ids": [1], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            client = RecordedVLLMClient(VLLMConfiguration(
                model="fixture", base_url="http://127.0.0.1:1234", batch_concurrency=2, trace_path=str(path)))
            try:
                with patch("ours.vllm_runtime.json_request", side_effect=respond), ThreadPoolExecutor(2) as pool:
                    futures = [pool.submit(client.complete_response, [ChatMessage("user", "fixture")]) for _ in range(2)]
                    self.assertEqual([f.result().content for f in futures], ["fixture", "fixture"])
            finally:
                client.close()
            self.assertEqual([json.loads(line)["call"] for line in path.read_text().splitlines()], [1, 2])

    def _response_test(self, reported_tokens, content, rejects):
        body = {"model": "fixture", "choices": [{"message": {"content": content},
                                                "token_ids": [11, 12]}],
                "prompt_token_ids": [1],
                "usage": {"prompt_tokens": 1, "completion_tokens": reported_tokens}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            client = RecordedVLLMClient(VLLMConfiguration(
                model="fixture", base_url="http://127.0.0.1:1234", trace_path=str(path)))
            try:
                with patch("ours.vllm_runtime.json_request", return_value=body) as post:
                    if rejects:
                        with self.assertRaisesRegex(ValueError, "Output token evidence"):
                            client.complete_response([ChatMessage("user", "fixture")], max_tokens=5)
                    else:
                        result = client.complete_response([ChatMessage("user", "fixture")], max_tokens=5)
                        self.assertEqual(result.content, content)
                        self.assertEqual(result.usage["completion_tokens"], 2)
                    self.assertEqual(post.call_count, 1)
                    self.assertEqual(post.call_args.args[1]["max_tokens"], 5)
            finally:
                client.close()
            self.assertEqual(json.loads(path.read_text())["raw_response"], body)
