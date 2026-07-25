"""Tests for the trace-to-FailureDetails extractor.

``extract_trace_data`` is pure and maps a raw Phoenix span into the fields the
pipeline needs. ``extract_failure_details`` asks a Gemini client for the
evidence JSON; the client is stubbed via the GeminiClient protocol so these
tests run offline and deterministically.
"""

from __future__ import annotations

import json

import pytest

from phoenix2pytest.extractor import (
    ExtractionError,
    extract_failure_details,
    extract_trace_data,
)
from phoenix2pytest.synthesiser import FailureDetails


class _StubGemini:
    """Returns a canned reply, records the messages it was given."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.system: str | None = None
        self.user: str | None = None

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        self.system = system
        self.user = user
        return self._reply


def test_extract_trace_data_maps_span_fields() -> None:
    span = {
        "context": {"span_id": "span-1"},
        "name": "llm_call",
        "attributes": {
            "input.value": "What time is it in Tokyo?",
            "output.value": "It is 3pm.",
            "llm.model_name": "gemini-2.5-flash",
            "phoenix2pytest.failure_mode": "stale_data",
        },
    }
    data = extract_trace_data(span)
    assert data["span_id"] == "span-1"
    assert data["user_prompt"] == "What time is it in Tokyo?"
    assert data["llm_output"] == "It is 3pm."
    assert data["failure_mode_label"] == "stale_data"


def test_extract_trace_data_defaults_missing_fields() -> None:
    data = extract_trace_data({"attributes": {}})
    assert data["user_prompt"] == ""
    assert data["failure_mode_label"] == ""


def test_extract_failure_details_parses_json_into_failuredetails() -> None:
    reply = json.dumps(
        {
            "evidence": "claimed a live time",
            "expected_behavior": "should refuse real-time data",
            "assertion_strategy": "refusal_marker_required",
            "key_strings_to_exclude": ["3pm"],
            "key_patterns_required": ["cannot"],
        }
    )
    client = _StubGemini(reply)
    trace_data = {
        "user_prompt": "What time is it?",
        "llm_output": "It is 3pm.",
        "failure_mode_label": "stale_data",
    }
    details = extract_failure_details(trace_data, client)
    assert isinstance(details, FailureDetails)
    assert details.failure_mode == "stale_data"
    assert details.evidence == "claimed a live time"
    assert details.key_strings_to_exclude == ["3pm"]
    # The trace content must reach the model.
    assert "3pm" in client.user


def test_extract_failure_details_unknown_label_when_missing() -> None:
    client = _StubGemini(json.dumps({"evidence": "x"}))
    details = extract_failure_details(
        {"user_prompt": "p", "llm_output": "o", "failure_mode_label": ""}, client
    )
    assert details.failure_mode == "unknown"


def test_extract_failure_details_rejects_non_json_reply() -> None:
    client = _StubGemini("Sorry, I cannot help with that.")
    with pytest.raises(ExtractionError):
        extract_failure_details(
            {"user_prompt": "p", "llm_output": "o", "failure_mode_label": "m"}, client
        )
