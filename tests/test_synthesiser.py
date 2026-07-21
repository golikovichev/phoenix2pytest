"""Tests for the synthesiser agent.

The Gemini client is stubbed via the GeminiClient protocol so these tests
run offline. A real-API integration test belongs in the hackathon e2e
script, not here.
"""

from __future__ import annotations

import json

import pytest

from phoenix2pytest.synthesiser import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    FailureDetails,
    GeminiClient,
    SynthesisError,
    TraceData,
    build_user_message,
    strip_markdown_fences,
    synthesise,
    synthesise_many,
    write_test_file,
)


class _StubGemini:
    """Records inputs, returns a canned reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, str]] = []

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        self.calls.append({"model": model, "system": system, "user": user})
        return self.reply


@pytest.fixture
def trace() -> TraceData:
    return TraceData(
        user_prompt="What time does the Madrid stock market open today?",
        llm_output="The Madrid stock market opens at 09:00 today.",
        span_id="span-abc-123",
    )


@pytest.fixture
def details() -> FailureDetails:
    return FailureDetails(
        failure_mode="stale_data",
        evidence="The Madrid stock market opens at 09:00 today.",
        expected_behavior="Refuse with a marker that real-time data is not available.",
        assertion_strategy="refusal_marker_required",
        key_strings_to_exclude=["09:00"],
        key_patterns_required=["cannot access real-time"],
    )


def test_failure_details_from_dict_round_trip():
    payload = {
        "failure_mode": "hallucination",
        "evidence": "fabricated fact",
        "expected_behavior": "say I do not know",
        "assertion_strategy": "substring_excluded",
        "key_strings_to_exclude": ["foo", "bar"],
        "key_patterns_required": [],
    }
    details = FailureDetails.from_dict(payload)
    assert details.failure_mode == "hallucination"
    assert details.key_strings_to_exclude == ["foo", "bar"]
    assert details.key_patterns_required == []


def test_failure_details_defaults_lists_to_empty():
    details = FailureDetails(failure_mode="off_topic_drift")
    assert details.key_strings_to_exclude == []
    assert details.key_patterns_required == []


def test_build_user_message_inlines_all_fields(trace, details):
    msg = build_user_message(trace, details)
    assert "USER PROMPT:" in msg
    assert "Madrid stock market" in msg
    assert "FAILURE MODE: stale_data" in msg
    assert "ASSERTION STRATEGY: refusal_marker_required" in msg
    assert json.dumps(details.key_strings_to_exclude) in msg


def test_strip_markdown_fences_handles_python_fenced_block():
    raw = "```python\nimport pytest\ndef test_x():\n    assert True\n```"
    assert strip_markdown_fences(raw).startswith("import pytest")
    assert "```" not in strip_markdown_fences(raw)


def test_strip_markdown_fences_handles_bare_fences():
    raw = "```\nimport pytest\n```"
    assert strip_markdown_fences(raw).startswith("import pytest")


def test_strip_markdown_fences_passes_clean_code_through():
    raw = "import pytest\n\ndef test_x():\n    assert True\n"
    cleaned = strip_markdown_fences(raw)
    assert cleaned.startswith("import pytest")
    assert cleaned.endswith("\n")


def test_synthesise_passes_system_prompt_and_chosen_model(trace, details):
    canned = "import pytest\n\ndef test_no_stale_data_madrid():\n    assert True\n"
    stub = _StubGemini(reply=canned)
    code = synthesise(trace, details, stub, model="gemini-2.5-pro")
    assert "test_no_stale_data_madrid" in code
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["model"] == "gemini-2.5-pro"
    assert call["system"] == SYSTEM_PROMPT
    assert "Madrid stock market" in call["user"]


def test_synthesise_defaults_to_pro_model(trace, details):
    stub = _StubGemini(reply="def test_x(): assert True\n")
    synthesise(trace, details, stub)
    assert stub.calls[0]["model"] == DEFAULT_MODEL == "gemini-2.5-pro"


def test_synthesise_strips_markdown_fences_from_model_output(trace, details):
    fenced = "```python\nimport pytest\n\ndef test_x():\n    pass\n```"
    stub = _StubGemini(reply=fenced)
    code = synthesise(trace, details, stub)
    assert "```" not in code
    assert code.startswith("import pytest")


def test_synthesise_raises_on_invalid_python(trace, details):
    # The model misbehaved and returned prose / broken syntax instead of code.
    # We must fail loudly, not write a broken .py file the user would run.
    stub = _StubGemini(reply="Sure! Here is your test: def (oops not valid")
    with pytest.raises(SynthesisError):
        synthesise(trace, details, stub)


def test_synthesise_accepts_valid_python(trace, details):
    stub = _StubGemini(reply="import pytest\n\ndef test_x():\n    assert True\n")
    code = synthesise(trace, details, stub)
    assert "test_x" in code


@pytest.mark.parametrize("reply", ["", "   \n  ", "```python\n```"])
def test_synthesise_raises_on_empty_reply(trace, details, reply):
    # An empty / whitespace / bare-fence reply parses as a valid but empty
    # module. Without a guard it lands on disk as a test-less .py that pytest
    # silently collects nothing from - the opposite of a regression test.
    stub = _StubGemini(reply=reply)
    with pytest.raises(SynthesisError):
        synthesise(trace, details, stub)


def test_synthesise_raises_on_testless_code(trace, details):
    # Valid Python, but no test function (imports only). pytest would run
    # nothing, so this is a broken artifact and must fail loudly.
    stub = _StubGemini(reply="import os\nimport pytest\n")
    with pytest.raises(SynthesisError):
        synthesise(trace, details, stub)


def test_synthesise_many_raises_on_invalid_python(trace, details):
    stub = _StubGemini(reply="def broken(:\n    pass\n")
    with pytest.raises(SynthesisError):
        synthesise_many([(trace, details)], stub)


def test_write_test_file_sanitises_failure_mode_in_filename(tmp_path):
    code = "import pytest\n\ndef test_x():\n    pass\n"
    target = write_test_file("hallucination/v2", code, tmp_path)
    assert target.name == "test_hallucination_v2.py"
    assert target.read_text(encoding="utf-8") == code


def test_write_test_file_falls_back_when_failure_mode_empty(tmp_path):
    code = "import pytest\n"
    target = write_test_file("", code, tmp_path)
    assert target.name == "test_unknown.py"


def test_write_test_file_creates_target_dir_if_missing(tmp_path):
    nested = tmp_path / "deep" / "nested"
    code = "import pytest\n"
    target = write_test_file("format_break", code, nested)
    assert target.exists()
    assert target.parent == nested


def test_gemini_client_is_runtime_checkable_protocol():
    # Smoke check that the protocol is importable and accepts a duck-typed stub.
    stub: GeminiClient = _StubGemini(reply="")
    assert hasattr(stub, "generate_text")
