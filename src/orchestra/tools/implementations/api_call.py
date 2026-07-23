"""Generic outbound HTTP call, gated by an explicit host allowlist.

No host is allowed by default — set ORCHESTRA_API_ALLOWLIST (comma-separated hostnames) to
permit specific hosts. This exists so an agent can't be tricked into making arbitrary outbound
requests (SSRF) just because the api_call tool is registered for its domain.
"""

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse


def _check_allowlist(url: str) -> None:
    allowlist = {h.strip() for h in os.environ.get("ORCHESTRA_API_ALLOWLIST", "").split(",") if h.strip()}
    host = urlparse(url).hostname or ""
    if host not in allowlist:
        raise PermissionError(
            f"Host {host!r} is not in ORCHESTRA_API_ALLOWLIST. "
            f"Allowed hosts: {sorted(allowlist) or '(none configured)'}"
        )


def api_call(
    method: str,
    url: str,
    headers: dict | None = None,
    json_body: dict | None = None,
    timeout_seconds: int = 10,
) -> dict:
    _check_allowlist(url)

    data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    request = urllib.request.Request(url, data=data, method=method.upper())
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    if json_body is not None:
        request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return {
                "status_code": response.status,
                "body": response.read().decode("utf-8", errors="replace"),
            }
    except urllib.error.HTTPError as exc:
        return {"status_code": exc.code, "body": exc.read().decode("utf-8", errors="replace")}
