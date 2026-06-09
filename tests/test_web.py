"""Tests for the FastAPI web layer.

These exercise the form rendering, JSON parsing, validation, and the JSON
vs HTML response negotiation. Synthesis itself is covered separately in
`test_synthesiser.py`; here we inject a stub client so the route is
deterministic and offline.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from phoenix2pytest import web

DEFAULT_STUB_REPLY = "import pytest\n\ndef test_stub():\n    assert True\n"


class _StubGemini:
    """Implements the GeminiClient protocol; records every call it sees."""

    def __init__(self, reply: str = DEFAULT_STUB_REPLY):
        self.reply = reply
        self.calls: list[dict] = []

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        self.calls.append({"model": model, "system": system, "user": user})
        return self.reply


@pytest.fixture(autouse=True)
def isolate_web_state(monkeypatch):
    """Snapshot and restore web module state around every test.

    Prevents one test mutating `_gemini_client` or `dependency_overrides`
    from leaking into the next test.
    """
    saved_client = web._gemini_client
    saved_overrides = dict(web.app.dependency_overrides)
    try:
        yield
    finally:
        web._gemini_client = saved_client
        web.app.dependency_overrides.clear()
        web.app.dependency_overrides.update(saved_overrides)


@pytest.fixture
def client_with_stub() -> tuple[TestClient, _StubGemini]:
    """Yield a TestClient with a stub Gemini client wired through DI override."""
    stub = _StubGemini()
    web.app.dependency_overrides[web.get_client] = lambda: stub
    yield TestClient(web.app), stub


def test_form_page_returns_html_form() -> None:
    client = TestClient(web.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Trace JSON" in response.text
    assert "Failure details JSON" in response.text


def test_generate_returns_html_with_generated_code(client_with_stub) -> None:
    client, stub = client_with_stub

    trace_json = json.dumps({"user_prompt": "What is 2+2?", "llm_output": "5"})
    details_json = json.dumps(
        {
            "failure_mode": "wrong_answer",
            "evidence": "Replied 5 instead of 4",
            "expected_behavior": "Reply with 4",
            "assertion_strategy": "answer_must_be_exact",
            "key_strings_to_exclude": ["5"],
        }
    )

    response = client.post(
        "/generate",
        data={"trace_json": trace_json, "details_json": details_json},
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "wrong_answer" in response.text
    assert "test_stub" in response.text  # stub reply rendered inside <pre>
    assert len(stub.calls) == 1
    assert "USER PROMPT:" in stub.calls[0]["user"]
    assert "What is 2+2?" in stub.calls[0]["user"]


def test_generate_returns_json_when_accept_header_requests_json(client_with_stub) -> None:
    client, stub = client_with_stub
    trace_json = json.dumps({"user_prompt": "Tell me a fact about cats"})
    details_json = json.dumps(
        {"failure_mode": "hallucination", "assertion_strategy": "substring_excluded"}
    )

    response = client.post(
        "/generate",
        data={"trace_json": trace_json, "details_json": details_json},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["failure_mode"] == "hallucination"
    assert payload["code"] == stub.reply
    assert payload["model"] == "gemini-2.5-pro"


def test_generate_returns_html_for_browser_accept_header(client_with_stub) -> None:
    """Browser Accept headers list both HTML and JSON; HTML must win."""
    client, _ = client_with_stub
    trace_json = json.dumps({"user_prompt": "x"})
    details_json = json.dumps({"failure_mode": "x"})
    response = client.post(
        "/generate",
        data={"trace_json": trace_json, "details_json": details_json},
        headers={"Accept": "text/html,application/xhtml+xml,application/json;q=0.1,*/*;q=0.8"},
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_generate_escapes_failure_mode_in_html_response(client_with_stub) -> None:
    """User-controlled failure_mode must not enable reflected XSS."""
    client, _ = client_with_stub
    trace_json = json.dumps({"user_prompt": "x"})
    details_json = json.dumps({"failure_mode": "<script>alert(1)</script>"})
    response = client.post(
        "/generate",
        data={"trace_json": trace_json, "details_json": details_json},
    )
    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_generate_rejects_oversized_body() -> None:
    """Request body above MAX_BODY_BYTES must be rejected before parsing."""
    stub = _StubGemini()
    web.app.dependency_overrides[web.get_client] = lambda: stub
    client = TestClient(web.app)
    huge = "x" * (web.MAX_BODY_BYTES + 100)
    response = client.post(
        "/generate",
        data={
            "trace_json": '{"user_prompt": "' + huge + '"}',
            "details_json": '{"failure_mode": "x"}',
        },
    )
    assert response.status_code == 413
    assert stub.calls == []


def test_generate_rejects_invalid_json(client_with_stub) -> None:
    client, _ = client_with_stub
    response = client.post(
        "/generate",
        data={"trace_json": "{not json", "details_json": '{"failure_mode": "x"}'},
    )
    assert response.status_code == 400
    assert "trace_json is not valid JSON" in response.json()["detail"]


def test_generate_rejects_missing_user_prompt(client_with_stub) -> None:
    client, _ = client_with_stub
    trace_json = json.dumps({"llm_output": "..."})
    details_json = json.dumps({"failure_mode": "x"})
    response = client.post(
        "/generate",
        data={"trace_json": trace_json, "details_json": details_json},
    )
    assert response.status_code == 400
    assert "user_prompt is required" in response.json()["detail"]


def test_generate_rejects_missing_failure_mode(client_with_stub) -> None:
    client, _ = client_with_stub
    trace_json = json.dumps({"user_prompt": "hi"})
    details_json = json.dumps({"evidence": "..."})
    response = client.post(
        "/generate",
        data={"trace_json": trace_json, "details_json": details_json},
    )
    assert response.status_code == 400
    assert "failure_mode is required" in response.json()["detail"]


def test_generate_rejects_non_object_json(client_with_stub) -> None:
    client, _ = client_with_stub
    response = client.post(
        "/generate",
        data={"trace_json": "[1, 2, 3]", "details_json": '{"failure_mode": "x"}'},
    )
    assert response.status_code == 400
    assert "must be a JSON object" in response.json()["detail"]


def test_get_client_raises_when_no_client_configured() -> None:
    """If configure_client was never called and no DI override is in place,
    the dependency should surface a clear 503 rather than blowing up at import.
    The autouse isolate_web_state fixture restores prior state after this test.
    """
    web._gemini_client = None
    client = TestClient(web.app)
    response = client.post(
        "/generate",
        data={
            "trace_json": '{"user_prompt": "hi"}',
            "details_json": '{"failure_mode": "x"}',
        },
    )
    assert response.status_code == 503
    assert "Gemini client is not configured" in response.json()["detail"]


def test_configure_client_installs_client_for_runtime() -> None:
    """configure_client() should make get_client() return the installed instance."""
    web._gemini_client = None
    stub = _StubGemini()
    web.configure_client(stub)
    assert web.get_client() is stub


def test_pages_share_single_batch_nav() -> None:
    """Both forms render the shared brand + Single/Batch nav for cohesion."""
    client = TestClient(web.app)
    single = client.get("/").text
    batch = client.get("/batch").text
    for page in (single, batch):
        assert 'class="nav"' in page
        assert 'href="/"' in page
        assert 'href="/batch"' in page
    # The active tab differs per page.
    assert 'href="/" class="active"' in single
    assert 'href="/batch" class="active"' in batch


def test_health_endpoint_returns_ok() -> None:
    """A lightweight /health endpoint lets uptime probes confirm the service."""
    client = TestClient(web.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_batch_form_page_returns_html_with_example_array() -> None:
    """GET /batch renders a form pre-filled with a runnable example array."""
    client = TestClient(web.app)
    response = client.get("/batch")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "items_json" in response.text
    # The pre-filled example is an html-escaped JSON array of trace+details
    # pairs, so the field names appear inside &quot;...&quot; rather than raw.
    assert "trace" in response.text
    assert "details" in response.text
    assert "&quot;" in response.text


def _batch_items(*pairs: tuple[str, str]) -> str:
    """Build the items_json payload from (prompt, failure_mode) pairs."""
    return json.dumps(
        [
            {
                "trace": {"user_prompt": prompt, "llm_output": "bad output"},
                "details": {"failure_mode": mode, "assertion_strategy": "substring_excluded"},
            }
            for prompt, mode in pairs
        ]
    )


def test_generate_batch_groups_by_failure_mode_in_html(client_with_stub) -> None:
    """Three traces across two failure modes produce two grouped sections."""
    client, stub = client_with_stub
    items_json = _batch_items(
        ("What is the capital of France?", "hallucination"),
        ("List the planets", "hallucination"),
        ("Summarise this", "format_break"),
    )
    response = client.post("/generate-batch", data={"items_json": items_json})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "hallucination" in response.text
    assert "format_break" in response.text
    # synthesise_many groups by failure mode: one Gemini call per distinct mode.
    assert len(stub.calls) == 2
    # The summary reports generated files vs input traces (the value of batch:
    # 3 traces folded into 2 files), not a tautology.
    assert "2 file(s) from 3 trace(s)" in response.text


def test_generate_batch_returns_json_mapping_when_requested(client_with_stub) -> None:
    """JSON callers get a {failure_mode_slug: code} mapping."""
    client, stub = client_with_stub
    items_json = _batch_items(
        ("Tell me a fact", "hallucination"),
        ("Give me JSON", "format_break"),
    )
    response = client.post(
        "/generate-batch",
        data={"items_json": items_json},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert set(payload["files"].keys()) == {"hallucination", "format_break"}
    assert payload["files"]["hallucination"] == stub.reply
    assert payload["model"] == "gemini-2.5-pro"


def test_generate_batch_rejects_non_array(client_with_stub) -> None:
    client, _ = client_with_stub
    response = client.post(
        "/generate-batch",
        data={"items_json": '{"trace": {}, "details": {}}'},
    )
    assert response.status_code == 400
    assert "must be a JSON array" in response.json()["detail"]


def test_generate_batch_rejects_empty_array(client_with_stub) -> None:
    client, _ = client_with_stub
    response = client.post("/generate-batch", data={"items_json": "[]"})
    assert response.status_code == 400
    assert "at least one" in response.json()["detail"]


def test_generate_batch_rejects_invalid_json(client_with_stub) -> None:
    client, _ = client_with_stub
    response = client.post("/generate-batch", data={"items_json": "[{not json"})
    assert response.status_code == 400
    assert "items_json is not valid JSON" in response.json()["detail"]


def test_generate_batch_rejects_non_object_item(client_with_stub) -> None:
    client, _ = client_with_stub
    response = client.post("/generate-batch", data={"items_json": "[1, 2, 3]"})
    assert response.status_code == 400
    assert "each item must be a JSON object" in response.json()["detail"]


def test_generate_batch_rejects_item_missing_user_prompt(client_with_stub) -> None:
    client, _ = client_with_stub
    items_json = json.dumps(
        [{"trace": {"llm_output": "x"}, "details": {"failure_mode": "hallucination"}}]
    )
    response = client.post("/generate-batch", data={"items_json": items_json})
    assert response.status_code == 400
    assert "user_prompt is required" in response.json()["detail"]


def test_generate_batch_rejects_item_missing_failure_mode(client_with_stub) -> None:
    client, _ = client_with_stub
    items_json = json.dumps([{"trace": {"user_prompt": "hi"}, "details": {"evidence": "x"}}])
    response = client.post("/generate-batch", data={"items_json": items_json})
    assert response.status_code == 400
    assert "failure_mode is required" in response.json()["detail"]


def test_generate_batch_escapes_failure_mode_in_html(client_with_stub) -> None:
    """User-controlled failure_mode must not enable reflected XSS in batch view."""
    client, _ = client_with_stub
    items_json = json.dumps(
        [
            {
                "trace": {"user_prompt": "x"},
                "details": {"failure_mode": "<script>alert(1)</script>"},
            }
        ]
    )
    response = client.post("/generate-batch", data={"items_json": items_json})
    assert response.status_code == 200
    # The failure mode becomes a sanitised slug (only [a-z0-9_]) and is also
    # html-escaped, so no raw script tag can reach the rendered page.
    assert "<script>alert(1)</script>" not in response.text
    assert "<script" not in response.text
