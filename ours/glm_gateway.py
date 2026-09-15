"""Loopback relay for the shared proposer with a persistent CNY budget journal.

The relay preserves Anthropic request/response bodies. Only GLM-5.3-Flash on
the official domestic endpoint is allowed. Credentials remain in the relay.
Accounting uses conservative rates, not a provider invoice. Incomplete calls
keep their full reservations, including after a process restart.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import urllib.error
import urllib.request
import uuid

from .probe_glm import NoRedirect

ENDPOINT = "https://open.bigmodel.cn/api/anthropic/v1/messages"
MODEL = "glm-5.3-flash"
LIMIT = Decimal("45")
INPUT_RATE = Decimal("1")
OUTPUT_RATE = Decimal("4")
CONTEXT = 1048576


def cost(input_tokens, output_tokens):
    for value in (input_tokens, output_tokens):
        if type(value) is not int or value < 0:
            raise ValueError("Usage counts must be nonnegative integers")
    return (INPUT_RATE * input_tokens + OUTPUT_RATE * output_tokens) / 1000000


class BudgetJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with self.path.open("a+") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0)
            rows = [json.loads(line) for line in file]
            yield file, rows

    @staticmethod
    def append(file, row):
        row["time_utc"] = datetime.now(timezone.utc).isoformat()
        file.write(json.dumps(row) + "\n")
        file.flush()
        import os
        os.fsync(file.fileno())

    @staticmethod
    def balances(rows):
        outstanding = {}
        charged = Decimal(0)
        for row in rows:
            if row["event"] == "reserve":
                if row["id"] in outstanding:
                    raise ValueError("Duplicate reservation")
                outstanding[row["id"]] = Decimal(row["cny"])
            elif row["event"] == "settle":
                if row["id"] not in outstanding:
                    raise ValueError("Settlement without reservation")
                outstanding.pop(row["id"])
                charged += Decimal(row["cny"])
            elif row["event"] == "initial_probe":
                charged += cost(row["input_tokens"], row["output_tokens"])
            else:
                raise ValueError("Unknown budget event")
        return charged, outstanding

    def initialize_probe(self, probe_path):
        with self.locked() as (file, rows):
            if rows:
                return
            probe = json.loads(Path(probe_path).read_text())
            if probe["status"] != "CONNECTED" or probe["response"]["model"] != MODEL:
                raise ValueError("The existing direct probe must be reconciled")
            usage = probe["response"]["usage"]
            cost(usage["input_tokens"], usage["output_tokens"])
            self.append(file, {"event": "initial_probe", "input_tokens": usage["input_tokens"],
                               "output_tokens": usage["output_tokens"]})

    def reserve(self, max_tokens):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 131072:
            raise ValueError("Output limit must be within the published model limit")
        amount = cost(CONTEXT, max_tokens)
        with self.locked() as (file, rows):
            charged, pending = self.balances(rows)
            if charged + sum(pending.values()) + amount > LIMIT:
                raise ValueError("Project CNY budget exhausted or held by unresolved calls")
            ident = uuid.uuid4().hex
            self.append(file, {"event": "reserve", "id": ident, "cny": str(amount), "max_tokens": max_tokens})
        return ident

    def settle(self, ident, usage):
        input_tokens = sum(usage.get(name, 0) for name in
                           ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        amount = cost(input_tokens, usage["output_tokens"])
        if any(usage.get("server_tool_use", {}).values()):
            raise ValueError("Unexpected server tools require billing reconciliation")
        with self.locked() as (file, rows):
            _, pending = self.balances(rows)
            if ident not in pending or amount > pending[ident]:
                raise ValueError("Usage exceeds its reservation or reservation is missing")
            self.append(file, {"event": "settle", "id": ident, "cny": str(amount), "usage": usage})

    def summary(self):
        with self.locked() as (_, rows):
            charged, pending = self.balances(rows)
        held = sum(pending.values(), Decimal(0))
        return {"budget_cny": str(LIMIT), "conservative_usage_cny": str(charged),
                "held_cny": str(held), "available_for_reservation_cny": str(LIMIT-charged-held),
                "unresolved_calls": len(pending), "settled_relay_calls": sum(r["event"] == "settle" for r in rows),
                "rates_cny_per_million": {"input_including_cache": str(INPUT_RATE), "output": str(OUTPUT_RATE)},
                "actual_provider_bill_cny": None,
                "scope": "This project's requests only; conservative accounting, not account balance or invoice"}


def merge_usage(current, update):
    for key, value in update.items():
        if key in {"input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"}:
            if type(value) is not int or value < 0:
                raise ValueError("Invalid usage")
            current[key] = max(current.get(key, 0), value)
        elif key == "server_tool_use":
            current[key] = value


def completed_usage(data, streamed):
    """Extract cumulative usage only from a complete provider response."""
    usage = {}
    completed = False
    if streamed:
        for line in data.splitlines():
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line[5:])
            if event.get("type") == "message_start":
                if event["message"].get("model") != MODEL:
                    raise ValueError("Provider returned a different model")
                merge_usage(usage, event["message"].get("usage", {}))
            merge_usage(usage, event.get("usage", {}))
            completed |= event.get("type") == "message_stop"
    else:
        message = json.loads(data)
        if message.get("model") != MODEL:
            raise ValueError("Provider returned a different model")
        merge_usage(usage, message.get("usage", {}))
        completed = message.get("type") == "message"
    if not completed or "input_tokens" not in usage or "output_tokens" not in usage:
        raise ValueError("Incomplete provider response; reservation retained")
    return usage


@contextmanager
def relay(journal, credential_path, *, max_requests=12, capture_directory=None):
    credential = Path(credential_path)
    if credential.stat().st_mode & 0o077:
        raise ValueError("Credential file must be private")
    api_key = json.loads(credential.read_text())["ZHIPU_API_KEY"]
    local_token = secrets.token_hex(32)
    requests_lock = threading.Lock()
    requests_sent = 0

    def diagnostic(fields):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), **fields}
        with (journal.path.parent / "transport.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            nonlocal requests_sent
            diagnostic({"event": "request", "path": self.path,
                        "content_length": self.headers.get("Content-Length"),
                        "content_encoding": self.headers.get("Content-Encoding"),
                        "transfer_encoding": self.headers.get("Transfer-Encoding")})
            if self.headers.get("Authorization") != "Bearer " + local_token and self.headers.get("x-api-key") != local_token:
                self.send_error(401)
                return
            if self.path.split("?")[0] != "/v1/messages":
                self.send_error(404)
                return
            started_response = False
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                raw = self.rfile.read(length)
                request = json.loads(raw)
                if request.get("model") != MODEL:
                    raise ValueError("Only the authorized model is enabled")
                if any(str(tool.get("type", "")).startswith(("web_search", "web_fetch")) for tool in request.get("tools", [])):
                    raise ValueError("Paid server tools are not enabled")
                with requests_lock:
                    if requests_sent >= max_requests:
                        raise ValueError("This session reached its request cap")
                    ident = journal.reserve(request["max_tokens"])
                    requests_sent += 1
                outbound = urllib.request.Request(ENDPOINT, data=raw, method="POST",
                    headers={"x-api-key": api_key, "content-type": "application/json",
                             "anthropic-version": "2023-06-01"})
                with urllib.request.build_opener(NoRedirect).open(outbound, timeout=120) as response:
                    # The native client stalls on EOF-delimited SSE in this environment.
                    # Preserve the exact body, buffer it, then provide explicit HTTP framing.
                    data = response.read(32 * 1024 * 1024 + 1)
                    if len(data) > 32 * 1024 * 1024:
                        raise ValueError("Provider response exceeds the transport limit")
                    if capture_directory is not None and request.get("stream"):
                        (Path(capture_directory) / f"{ident}.sse").write_bytes(data)
                    usage = completed_usage(data, request.get("stream", False))
                    journal.settle(ident, usage)
                    self.send_response(response.status)
                    self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Connection", "close")
                    self.close_connection = True
                    self.end_headers()
                    started_response = True
                    self.wfile.write(data)
                    self.wfile.flush()
                    diagnostic({"event": "response_complete", "usage_complete": True})
            except Exception as exc:
                # No upstream error bodies/headers or exception strings are logged.
                # Ambiguous outcomes retain their reservation and cannot overspend it.
                diagnostic({"event": "failure", "error_type": type(exc).__name__,
                            "local_reason": str(exc) if isinstance(exc, ValueError) else None,
                            "http_status": exc.code if isinstance(exc, urllib.error.HTTPError) else None})
                if not started_response:
                    self.send_error(502, "Request rejected or provider unavailable; check budget journal")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield {"base_url": f"http://127.0.0.1:{server.server_port}", "token": local_token}
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
