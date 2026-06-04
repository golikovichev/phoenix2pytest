"""Tests for the production Gemini client wiring.

The web layer injects a GeminiClient via configure_client(); historically nothing
wired a real client at startup, so a deployed instance answered /generate with a
503 ("client not configured"). These tests pin the production adapter and the
startup wiring that fixes that.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import phoenix2pytest.web as web
from phoenix2pytest.synthesiser import VertexGeminiClient, build_default_client


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})

        class _Resp:
            text = "GENERATED CODE"

        return _Resp()


class _FakeGenai:
    def __init__(self) -> None:
        self.models = _FakeModels()


def test_adapter_maps_generate_text_to_genai_call():
    """generate_text(model, system, user) -> genai generate_content with the
    user message as contents and the system text as system_instruction."""
    fake = _FakeGenai()
    client = VertexGeminiClient(genai_client=fake)

    out = client.generate_text(model="gemini-2.5-pro", system="SYS", user="USR")

    assert out == "GENERATED CODE"
    assert len(fake.models.calls) == 1
    call = fake.models.calls[0]
    assert call["model"] == "gemini-2.5-pro"
    assert call["contents"] == "USR"
    # system text travels as the config's system_instruction
    assert getattr(call["config"], "system_instruction", None) == "SYS"


def test_adapter_satisfies_protocol_without_network():
    """build_default_client() returns a usable GeminiClient and constructing it
    must not require network or credentials (lazy genai init)."""
    client = build_default_client()
    assert callable(getattr(client, "generate_text", None))


def test_startup_wires_a_client_when_none_configured(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(web, "_gemini_client", None, raising=False)
    monkeypatch.setattr("phoenix2pytest.synthesiser.build_default_client", lambda: sentinel)

    web._wire_default_client()

    assert web.get_client() is sentinel


def test_startup_does_not_clobber_existing_client(monkeypatch):
    existing = object()
    monkeypatch.setattr(web, "_gemini_client", existing, raising=False)
    monkeypatch.setattr(
        "phoenix2pytest.synthesiser.build_default_client",
        lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    web._wire_default_client()

    assert web.get_client() is existing


def test_startup_swallows_wiring_failure(monkeypatch):
    """A genai/credential failure at startup must not crash boot; the client
    stays None and /generate degrades to a clear 503 instead."""
    monkeypatch.setattr(web, "_gemini_client", None, raising=False)

    def _boom():
        raise RuntimeError("no creds")

    monkeypatch.setattr("phoenix2pytest.synthesiser.build_default_client", _boom)

    web._wire_default_client()  # must not raise

    assert web._gemini_client is None


def test_ensure_client_passes_vertex_kwargs(monkeypatch):
    """The lazy genai client is built for Vertex with explicit project/location
    kwargs (no reliance on mutating process env)."""
    import google.genai as genai_mod

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(genai_mod, "Client", _FakeClient)

    client = VertexGeminiClient(project="proj-x", location="loc-y")
    client._ensure_client()

    assert captured.get("vertexai") is True
    assert captured.get("project") == "proj-x"
    assert captured.get("location") == "loc-y"


def _req(headers: dict):
    return SimpleNamespace(headers=headers)


def test_api_token_gate_open_when_unset(monkeypatch):
    monkeypatch.delenv("P2P_API_TOKEN", raising=False)
    # no token configured -> open endpoint, no raise even without a header
    web.require_api_token(_req({}))


def test_api_token_gate_rejects_missing_or_wrong(monkeypatch):
    monkeypatch.setenv("P2P_API_TOKEN", "s3cret")
    with pytest.raises(HTTPException) as missing:
        web.require_api_token(_req({}))
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as wrong:
        web.require_api_token(_req({"X-API-Token": "nope"}))
    assert wrong.value.status_code == 401


def test_api_token_gate_accepts_match(monkeypatch):
    monkeypatch.setenv("P2P_API_TOKEN", "s3cret")
    # correct token -> no raise
    web.require_api_token(_req({"X-API-Token": "s3cret"}))
