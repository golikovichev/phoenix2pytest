"""Minimal FastAPI web layer for phoenix2pytest.

Exposes a paste-and-generate form so a user can drop a Phoenix trace + its
extracted failure details into a browser and get back a runnable pytest
file. Intended as the demo surface for the hackathon submission; the heavy
lifting (trace parsing, failure extraction, Gemini synthesis) lives in
`phoenix2pytest.synthesiser`.

Endpoints:
- GET  /          : inline HTML form
- POST /generate  : accepts trace + failure details JSON, returns code

The Gemini client is injected through FastAPI's dependency-override hook so
tests can swap in a fake without monkeypatching the module.

Hardening notes:
- /generate is gated by an optional shared secret: when P2P_API_TOKEN is set,
  callers must send a matching X-API-Token header (see require_api_token).
  With the token unset the endpoint is open, which is intended for the public
  demo; cost is bounded by the request-size cap, max-instances, and a billing
  budget. Add a rate-limit middleware before any heavier exposure.
- Request body capped at MAX_BODY_BYTES below; oversized posts get 413.
"""

from __future__ import annotations

import hmac
import html
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .synthesiser import (
    DEFAULT_MODEL,
    FailureDetails,
    GeminiClient,
    TraceData,
    synthesise,
)

# Cap on the raw POST body, defends against accidental megabyte pastes that
# would hang the worker and burn Gemini tokens. 256 KB is comfortably above
# realistic trace payloads while still bounding cost.
MAX_BODY_BYTES = 256 * 1024

logger = logging.getLogger(__name__)

# Optional shared-secret gate for /generate. When P2P_API_TOKEN is set in the
# environment, callers must send a matching X-API-Token header; otherwise the
# endpoint is open (local/dev default). This keeps a publicly reachable
# deployment from letting anyone spend the project's Gemini quota.
_API_TOKEN_ENV = "P2P_API_TOKEN"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Wire the production Gemini client once, when the server boots."""
    _wire_default_client()
    yield


app = FastAPI(
    title="phoenix2pytest",
    description="Turn Phoenix LLM failure traces into pytest regression tests.",
    version="0.0.1",
    lifespan=_lifespan,
)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject oversized POST bodies before they reach the route handler."""
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_BODY_BYTES:
                    return JSONResponse(
                        {"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes"},
                        status_code=413,
                    )
            except ValueError:
                # Malformed Content-Length header: let downstream handle the
                # request normally. Body size limit still applies via ASGI.
                pass
    return await call_next(request)


# Module-level slot for the Gemini client. Production wiring sets this from
# env at startup; tests override via `app.dependency_overrides[get_client]`.
_gemini_client: GeminiClient | None = None


def get_client() -> GeminiClient:
    """FastAPI dependency: return the active Gemini client.

    Raises an HTTPException if no client is configured so the error reaches
    the user via JSON rather than a 500 traceback.
    """
    if _gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini client is not configured. Set up google-genai credentials "
                "and call phoenix2pytest.web.configure_client(...) before serving."
            ),
        )
    return _gemini_client


def configure_client(client: GeminiClient) -> None:
    """Install a GeminiClient instance for production use."""
    global _gemini_client
    _gemini_client = client


def _wire_default_client() -> None:
    """Wire a real Gemini client at startup when the host has not injected one.

    Without this, a deployed instance left ``_gemini_client`` at ``None`` and
    every /generate call returned a 503. Tests inject their own client (via
    configure_client or dependency_overrides), so this skips when one is already
    set. Building the client must never crash boot: on failure (e.g. missing
    credentials) the slot stays ``None`` and /generate degrades to a clear 503.
    """
    global _gemini_client
    if _gemini_client is not None:
        return
    try:
        from .synthesiser import build_default_client

        configure_client(build_default_client())
    except Exception as exc:  # boot must survive a genai/creds failure
        logger.warning("default Gemini client wiring failed; /generate will 503: %s", exc)


# A working example pre-filled into the form so a first-time visitor (e.g. a
# judge) can click Generate with no edits. Kept as the single source of truth:
# rendered into the form and asserted valid by tests.
EXAMPLE_TRACE_JSON = (
    '{"user_prompt": "What is the capital of France?", '
    '"llm_output": "The capital of France is Berlin."}'
)
EXAMPLE_DETAILS_JSON = (
    '{"failure_mode": "hallucination", "evidence": "answered Berlin", '
    '"expected_behavior": "should answer Paris", '
    '"assertion_strategy": "substring_excluded", '
    '"key_strings_to_exclude": ["Berlin"], "key_patterns_required": ["Paris"]}'
)

# ruff: noqa: E501 (HTML payload kept verbatim for browser rendering)
# Shared inline style for the form and result pages. Inline + dependency-free so
# the demo loads instantly and works under a strict CSP with no external assets.
_STYLE = """
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1c1e21; --muted:#6b7280; --accent:#e8590c; --border:#e2e5ea; --code-bg:#0d1117; --code-ink:#e6edf3; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--ink); margin: 0; padding: 2.5rem 1rem; line-height: 1.5; }
  .card { max-width: 820px; margin: 0 auto; background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.75rem 2rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
  h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
  .tagline { color: var(--muted); margin: 0 0 1rem; }
  label { display: block; margin: 1.1rem 0 .35rem; font-weight: 600; }
  textarea { width: 100%; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem; padding: .6rem .7rem; border: 1px solid var(--border); border-radius: 8px; resize: vertical; background: #fcfcfd; }
  textarea:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(232,89,12,.15); }
  button { margin-top: 1.25rem; padding: .65rem 1.3rem; font-size: 1rem; font-weight: 600; color: #fff; background: var(--accent); border: 0; border-radius: 8px; cursor: pointer; }
  button:hover { background: #cf4f08; }
  .hint { color: var(--muted); font-size: .85rem; margin-top: .3rem; }
  pre { background: var(--code-bg); color: var(--code-ink); padding: 1rem 1.1rem; border-radius: 10px; overflow-x: auto; font-size: .85rem; line-height: 1.45; }
  a { color: var(--accent); }
  footer { max-width: 820px; margin: 1rem auto 0; color: var(--muted); font-size: .85rem; }
"""

_FORM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>phoenix2pytest: trace to pytest</title>
  <style>__STYLE__</style>
</head>
<body>
  <main class="card">
    <h1>phoenix2pytest</h1>
    <p class="tagline">Turn a production LLM failure trace into a runnable pytest regression test.</p>
    <p class="hint">An example is pre-filled below: just click Generate. Generation calls Gemini and usually takes a few seconds.</p>

    <form method="post" action="/generate">
      <label for="trace_json">Trace JSON</label>
      <textarea id="trace_json" name="trace_json" rows="5" required>__TRACE_EXAMPLE__</textarea>
      <div class="hint">Fields: user_prompt (required), llm_output, span_id.</div>

      <label for="details_json">Failure details JSON</label>
      <textarea id="details_json" name="details_json" rows="8" required>__DETAILS_EXAMPLE__</textarea>
      <div class="hint">Fields: failure_mode (required), evidence, expected_behavior, assertion_strategy, key_strings_to_exclude, key_patterns_required.</div>

      <button type="submit">Generate pytest file</button>
    </form>
  </main>
  <footer>Source: <a href="https://github.com/golikovichev/phoenix2pytest">github.com/golikovichev/phoenix2pytest</a></footer>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def form_page() -> str:
    """Render the paste-and-submit HTML form, pre-filled with a runnable example."""
    return (
        _FORM_HTML.replace("__STYLE__", _STYLE)
        .replace("__TRACE_EXAMPLE__", html.escape(EXAMPLE_TRACE_JSON))
        .replace("__DETAILS_EXAMPLE__", html.escape(EXAMPLE_DETAILS_JSON))
    )


def _parse_json_field(field_name: str, raw: str) -> dict:
    """Decode a JSON string from a form field with a clear 400 on failure."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is not valid JSON: {exc.msg} at column {exc.colno}",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a JSON object, got {type(parsed).__name__}",
        )
    return parsed


def require_api_token(request: Request) -> None:
    """Gate /generate behind a shared secret when P2P_API_TOKEN is configured.

    No token configured -> open endpoint (local/dev). Token configured -> the
    request must carry a matching X-API-Token header, compared in constant time.
    This stops an exposed public URL from spending the project's Gemini quota.
    """
    expected = os.environ.get(_API_TOKEN_ENV)
    if not expected:
        return
    provided = request.headers.get("X-API-Token", "")
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Token.")


@app.post("/generate", response_model=None)
def generate(
    request: Request,
    trace_json: Annotated[str, Form()],
    details_json: Annotated[str, Form()],
    client: Annotated[GeminiClient, Depends(get_client)],
    _: Annotated[None, Depends(require_api_token)] = None,
) -> JSONResponse | HTMLResponse:
    """Synthesise a pytest file from posted trace + failure JSON.

    Returns JSON when the caller asks for it via the Accept header, otherwise
    renders the result inline as HTML so the demo form is self-contained.
    """
    trace_payload = _parse_json_field("trace_json", trace_json)
    details_payload = _parse_json_field("details_json", details_json)

    if not trace_payload.get("user_prompt"):
        raise HTTPException(status_code=400, detail="trace_json.user_prompt is required")
    if not details_payload.get("failure_mode"):
        raise HTTPException(status_code=400, detail="details_json.failure_mode is required")

    trace = TraceData(
        user_prompt=str(trace_payload["user_prompt"]),
        llm_output=str(trace_payload.get("llm_output") or ""),
        span_id=str(trace_payload.get("span_id") or ""),
    )
    details = FailureDetails.from_dict(details_payload)

    code = synthesise(trace, details, client, model=DEFAULT_MODEL)

    if _client_wants_json(request.headers.get("accept") or ""):
        return JSONResponse(
            {
                "failure_mode": details.failure_mode,
                "model": DEFAULT_MODEL,
                "code": code,
            }
        )

    # Both `code` and `failure_mode` come from user-controlled JSON; escape
    # everything before interpolating into the response template.
    safe_code = html.escape(code)
    safe_mode = html.escape(details.failure_mode)
    body = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>phoenix2pytest: generated test</title>"
        "<style>" + _STYLE + "</style></head><body>"
        '<main class="card">'
        f"<h1>Generated pytest</h1>"
        f'<p class="tagline">Failure mode: {safe_mode}</p>'
        f"<pre>{safe_code}</pre>"
        '<p><a href="/">Generate another</a></p>'
        "</main></body></html>"
    )
    return HTMLResponse(body)


def _client_wants_json(accept_header: str) -> bool:
    """Return True when the caller explicitly prefers JSON over HTML.

    The previous implementation treated any occurrence of `application/json`
    as a JSON request, which broke for browsers sending
    `text/html, application/xhtml+xml, application/json;q=0.1` because HTML
    is in the list. We treat `*/*` (curl default) and HTML-bearing Accept
    headers as preferring HTML so the demo form keeps working. API clients
    that explicitly want JSON should send `Accept: application/json`.
    """
    lowered = accept_header.lower()
    if "application/json" not in lowered:
        return False
    # If text/html is also in the list (the typical browser shape) HTML wins
    # so the demo form keeps rendering. API clients should narrow their Accept
    # header to just application/json.
    return "text/html" not in lowered
