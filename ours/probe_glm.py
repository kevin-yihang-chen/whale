"""Single bounded official GLM request; shared infrastructure outside E1-E4.

Never log headers or credential values. No retries, tools, fallback model or
redirects. A pre-request cost reservation remains held on an uncertain failure.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .proposer_profiles import GLM_FLASH_CN


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Credential-bearing request redirects are disabled")


def main():
    credential = Path("/userhome/cs3/yihangc/.config/whale-delta/credentials.json")
    if credential.stat().st_mode & 0o077:
        raise ValueError("Credential file must be private")
    key = os.environ.get("ZHIPU_API_KEY") or json.loads(credential.read_text())["ZHIPU_API_KEY"]
    output = Path("results/glm-cn-connectivity-20260909.json")
    reservation = Path("results/glm-cn-connectivity-reservation-20260909.json")
    if output.exists() or reservation.exists():
        raise ValueError("Probe is one-shot; existing attempts must be reconciled first")
    body = {"model": GLM_FLASH_CN.model, "max_tokens": 256,
            "messages": [{"role": "user", "content": "Reply with exactly OK."}]}
    plan = {"purpose": "API connectivity only", "budget_authorized_cny": 45,
            "reserved_cny": 1.01, "reservation_basis": "Conservative 1 CNY/M input for full 1M context plus 4 CNY/M output at max256; not an official tariff or actual bill",
            "endpoint": GLM_FLASH_CN.base_url + "/v1/messages", "request": body,
            "created_at_utc": datetime.now(timezone.utc).isoformat()}
    reservation.write_text(json.dumps(plan, indent=2) + "\n")
    request = urllib.request.Request(plan["endpoint"], data=json.dumps(body).encode(),
                                    headers={"x-api-key": key, "Content-Type": "application/json",
                                             "anthropic-version": "2023-06-01"}, method="POST")
    started = time.perf_counter()
    result = {"model_requested": GLM_FLASH_CN.model, "endpoint": plan["endpoint"], "request": body}
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=90) as response:
            payload = json.loads(response.read())
            result.update(http_status=response.status, response=payload,
                          status="CONNECTED" if payload.get("type") == "message" else "UNEXPECTED_RESPONSE")
    except urllib.error.HTTPError as exc:
        result.update(status="HTTP_ERROR", http_status=exc.code,
                      response=exc.read(8192).decode(errors="replace").replace(key, "[REDACTED]"))
    except Exception as exc:
        result.update(status="TRANSPORT_ERROR", error_type=type(exc).__name__)
    result.update(seconds=time.perf_counter() - started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  actual_bill_cny=None, note="Provider usage is recorded; no billing amount is assumed from CLI USD estimates")
    encoded = json.dumps(result, indent=2, ensure_ascii=False).replace(key, "[REDACTED]") + "\n"
    output.write_text(encoded)
    safe = {k: result[k] for k in ("status", "seconds", "model_requested")}
    if isinstance(result.get("response"), dict):
        safe["model_returned"] = result["response"].get("model")
        safe["usage"] = result["response"].get("usage")
    print(json.dumps(safe))
    if result["status"] != "CONNECTED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
