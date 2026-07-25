"""Tests for the offline traces-file loader.

The loader reads the same item shape the batch web endpoint accepts
(a JSON list of ``{"trace": {...}, "details": {...}}`` objects) and turns it
into the ``(TraceData, FailureDetails)`` pairs the synthesiser consumes, so the
CLI can run without a live Phoenix server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phoenix2pytest.loader import TraceLoadError, load_traces, parse_item
from phoenix2pytest.synthesiser import FailureDetails, TraceData


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "traces.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_item_builds_trace_and_details() -> None:
    trace, details = parse_item(
        {
            "trace": {"user_prompt": "hi", "llm_output": "oops", "span_id": "s1"},
            "details": {"failure_mode": "hallucination", "evidence": "made up a date"},
        }
    )
    assert isinstance(trace, TraceData)
    assert isinstance(details, FailureDetails)
    assert trace.user_prompt == "hi"
    assert trace.llm_output == "oops"
    assert trace.span_id == "s1"
    assert details.failure_mode == "hallucination"
    assert details.evidence == "made up a date"


def test_load_traces_reads_a_list(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {"trace": {"user_prompt": "a"}, "details": {"failure_mode": "m1"}},
            {"trace": {"user_prompt": "b"}, "details": {"failure_mode": "m2"}},
        ],
    )
    pairs = load_traces(path)
    assert len(pairs) == 2
    assert [t.user_prompt for t, _ in pairs] == ["a", "b"]
    assert [d.failure_mode for _, d in pairs] == ["m1", "m2"]


def test_load_traces_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TraceLoadError):
        load_traces(tmp_path / "nope.json")


def test_load_traces_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(TraceLoadError):
        load_traces(path)


def test_load_traces_root_must_be_a_list(tmp_path: Path) -> None:
    path = _write(tmp_path, {"trace": {"user_prompt": "a"}, "details": {"failure_mode": "m"}})
    with pytest.raises(TraceLoadError) as exc:
        load_traces(path)
    assert "list" in str(exc.value)


def test_item_missing_user_prompt_raises() -> None:
    with pytest.raises(TraceLoadError) as exc:
        parse_item({"trace": {}, "details": {"failure_mode": "m"}})
    assert "user_prompt is required" in str(exc.value)


def test_item_missing_failure_mode_raises() -> None:
    with pytest.raises(TraceLoadError) as exc:
        parse_item({"trace": {"user_prompt": "a"}, "details": {}})
    assert "failure_mode is required" in str(exc.value)


def test_item_not_an_object_raises() -> None:
    with pytest.raises(TraceLoadError) as exc:
        parse_item(["not", "a", "dict"])
    assert "each item must be a JSON object" in str(exc.value)
