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
- No auth on /generate. Each call hits the Gemini quota of whoever wired
  the client via configure_client(). Acceptable for a private hackathon
  demo, NOT acceptable for a public URL. Front this with a shared-secret
  header or a rate-limit middleware before sharing the deployment.
- Request body capped at MAX_BODY_BYTES below; oversized posts get 413.
"""

from __future__ import annotations

import html
import json
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

app = FastAPI(
    title="phoenix2pytest",
    description="Turn Phoenix LLM failure traces into pytest regression tests.",
    version="0.0.1",
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


# ruff: noqa: E501 (HTML payload kept verbatim for browser rendering)
_FORM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>phoenix2pytest: trace to pytest</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 880px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.5rem; }
    textarea { width: 100%; font-family: ui-monospace, monospace; font-size: 0.85rem; }
    label { display: block; margin: 1rem 0 0.25rem; font-weight: 600; }
    button { margin-top: 1rem; padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
    pre { background: #f4f4f4; padding: 1rem; overflow-x: auto; }
    .hint { color: #555; font-size: 0.85rem; margin-top: 0.25rem; }
  </style>
</head>
<body>
  <h1>phoenix2pytest</h1>
  <p>Paste a Phoenix LLM trace and its extracted failure details to get a runnable pytest file.</p>

  <form method="post" action="/generate">
    <label for="trace_json">Trace JSON</label>
    <textarea id="trace_json" name="trace_json" rows="6" required
      placeholder='{"user_prompt": "...", "llm_output": "...", "span_id": "..."}'></textarea>
    <div class="hint">Fields: user_prompt (required), llm_output, span_id.</div>

    <label for="details_json">Failure details JSON</label>
    <textarea id="details_json" name="details_json" rows="8" required
      placeholder='{"failure_mode": "hallucination", "evidence": "...", "expected_behavior": "...", "assertion_strategy": "substring_excluded", "key_strings_to_exclude": ["..."]}'></textarea>
    <div class="hint">Fields: failure_mode (required), evidence, expected_behavior, assertion_strategy, key_strings_to_exclude, key_patterns_required.</div>

    <button type="submit">Generate pytest file</button>
  </form>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def form_page() -> str:
    """Render the paste-and-submit HTML form."""
    return _FORM_HTML


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


@app.post("/generate", response_model=None)
def generate(
    request: Request,
    trace_json: Annotated[str, Form()],
    details_json: Annotated[str, Form()],
    client: Annotated[GeminiClient, Depends(get_client)],
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
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>phoenix2pytest: generated test</title></head><body>"
        f"<h1>Generated pytest for failure mode: {safe_mode}</h1>"
        f"<pre>{safe_code}</pre>"
        '<p><a href="/">Generate another</a></p>'
        "</body></html>"
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
