"""Tests for multi-trace handling: synthesise_many + write_test_files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from phoenix2pytest.synthesiser import (
    DEFAULT_MODEL,
    FailureDetails,
    TraceData,
    build_user_message_for_group,
    synthesise_many,
    write_test_files,
)


class _RecordingClient:
    """Stub GeminiClient that records calls and returns a scripted reply."""

    def __init__(self, reply: str = "def test_stub(): pass\n") -> None:
        self.reply = reply
        self.calls: list[dict[str, str]] = []

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        self.calls.append({"model": model, "system": system, "user": user})
        return self.reply


def _trace(prompt: str, span_id: str = "") -> TraceData:
    return TraceData(user_prompt=prompt, llm_output="bad output", span_id=span_id)


def _details(mode: str = "hallucination") -> FailureDetails:
    return FailureDetails(
        failure_mode=mode,
        evidence="bot fabricated a number",
        expected_behavior="bot says it does not know",
        assertion_strategy="substring_excluded",
        key_strings_to_exclude=["fabricated"],
        key_patterns_required=[],
    )


def test_synthesise_many_returns_empty_dict_for_empty_input():
    client = _RecordingClient()
    assert synthesise_many([], client) == {}
    assert client.calls == []


def test_synthesise_many_single_item_one_call():
    client = _RecordingClient(reply="def test_single(): pass\n")
    result = synthesise_many([(_trace("p1"), _details())], client)
    assert list(result.keys()) == ["hallucination"]
    assert result["hallucination"] == "def test_single(): pass\n"
    assert len(client.calls) == 1


def test_synthesise_many_distinct_failures_one_call_each():
    client = _RecordingClient()
    items = [
        (_trace("p1"), _details("hallucination")),
        (_trace("p2"), _details("refusal_when_safe")),
        (_trace("p3"), _details("format_violation")),
    ]
    result = synthesise_many(items, client)
    assert list(result.keys()) == [
        "hallucination",
        "refusal_when_safe",
        "format_violation",
    ]
    assert len(client.calls) == 3


def test_synthesise_many_groups_duplicates_into_single_call():
    client = _RecordingClient()
    items = [
        (_trace("p1"), _details("hallucination")),
        (_trace("p2"), _details("hallucination")),
        (_trace("p3"), _details("hallucination")),
    ]
    result = synthesise_many(items, client)
    assert list(result.keys()) == ["hallucination"]
    # One grouped call, not three individual ones
    assert len(client.calls) == 1
    # Group prompt mentions all three user prompts
    user_msg = client.calls[0]["user"]
    assert "p1" in user_msg
    assert "p2" in user_msg
    assert "p3" in user_msg
    assert "parametrize" in user_msg


def test_synthesise_many_mixed_distinct_and_grouped():
    client = _RecordingClient()
    items = [
        (_trace("p1"), _details("hallucination")),
        (_trace("p2"), _details("format_violation")),
        (_trace("p3"), _details("hallucination")),  # group with p1
        (_trace("p4"), _details("refusal_when_safe")),
    ]
    result = synthesise_many(items, client)
    assert list(result.keys()) == [
        "hallucination",
        "format_violation",
        "refusal_when_safe",
    ]
    # 3 distinct slugs = 3 calls
    assert len(client.calls) == 3
    # The hallucination call is the grouped one
    halluc_call = next(c for c in client.calls if "parametrize" in c["user"])
    assert "p1" in halluc_call["user"] and "p3" in halluc_call["user"]


def test_synthesise_many_passes_model_through():
    client = _RecordingClient()
    synthesise_many([(_trace("p"), _details())], client, model="gemini-2.5-pro-preview")
    assert client.calls[0]["model"] == "gemini-2.5-pro-preview"


def test_synthesise_many_default_model_is_used():
    client = _RecordingClient()
    synthesise_many([(_trace("p"), _details())], client)
    assert client.calls[0]["model"] == DEFAULT_MODEL


def test_synthesise_many_uses_slug_for_messy_failure_mode():
    client = _RecordingClient()
    items = [(_trace("p"), _details("Hallucination! With $pecial Chars"))]
    result = synthesise_many(items, client)
    assert list(result.keys()) == ["hallucination__with__pecial_chars"]


def test_synthesise_many_handles_empty_failure_mode_as_unknown():
    client = _RecordingClient()
    result = synthesise_many([(_trace("p"), _details(""))], client)
    assert list(result.keys()) == ["unknown"]


def test_build_user_message_for_group_includes_all_prompts():
    traces = [_trace("first"), _trace("second"), _trace("third")]
    msg = build_user_message_for_group(traces, _details())
    assert "first" in msg
    assert "second" in msg
    assert "third" in msg
    assert "parametrize" in msg
    assert "single pytest module" in msg


def test_build_user_message_for_group_preserves_non_ascii():
    """Russian / unicode prompts must not be JSON-escaped to ASCII."""
    traces = [_trace("привет мир"), _trace("こんにちは")]
    msg = build_user_message_for_group(traces, _details())
    assert "привет мир" in msg
    assert "こんにちは" in msg


def test_write_test_files_writes_one_file_per_slug(tmp_path: Path):
    codes = {
        "hallucination": "def test_a(): pass\n",
        "format_violation": "def test_b(): pass\n",
    }
    paths = write_test_files(codes, tmp_path)
    assert len(paths) == 2
    assert (tmp_path / "test_hallucination.py").exists()
    assert (tmp_path / "test_format_violation.py").exists()
    assert (tmp_path / "test_hallucination.py").read_text(encoding="utf-8") == (
        "def test_a(): pass\n"
    )


def test_write_test_files_creates_target_dir(tmp_path: Path):
    target = tmp_path / "nested" / "dir"
    paths = write_test_files({"slug": "code\n"}, target)
    assert target.is_dir()
    assert paths[0] == target / "test_slug.py"


def test_write_test_files_empty_dict_creates_dir_only(tmp_path: Path):
    target = tmp_path / "empty"
    paths = write_test_files({}, target)
    assert paths == []
    assert target.is_dir()


def test_write_test_files_overwrites_existing(tmp_path: Path):
    existing = tmp_path / "test_slug.py"
    existing.write_text("OLD\n", encoding="utf-8")
    write_test_files({"slug": "NEW\n"}, tmp_path)
    assert existing.read_text(encoding="utf-8") == "NEW\n"


@pytest.fixture
def stub_client_factory():
    """Return a factory so tests pick their own reply payload."""

    def _make(reply: str = "def test_stub(): pass\n") -> _RecordingClient:
        return _RecordingClient(reply)

    return _make


def test_synthesise_many_strips_markdown_fences_per_group(
    stub_client_factory: Any,
):
    """Each group output passes through strip_markdown_fences."""
    client = stub_client_factory(reply="```python\ndef test_x(): pass\n```")
    result = synthesise_many([(_trace("p"), _details())], client)
    assert "```" not in result["hallucination"]
    assert "def test_x(): pass" in result["hallucination"]
