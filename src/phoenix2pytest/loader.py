"""Load flagged traces from a JSON file for offline synthesis.

The file format matches the batch web endpoint: a JSON list of items, each an
object with a ``trace`` and a ``details`` object. Sharing the item shape means a
payload captured for the web UI runs unchanged through the CLI, and both paths
validate it the same way. The web endpoint wraps :func:`parse_item` and
translates :class:`TraceLoadError` into an HTTP 400, so the validation messages
live here once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phoenix2pytest.synthesiser import FailureDetails, TraceData


class TraceLoadError(ValueError):
    """A traces file or one of its items is missing or malformed."""


def parse_item(item: Any) -> tuple[TraceData, FailureDetails]:
    """Validate one item and build its ``TraceData`` + ``FailureDetails``.

    Raises :class:`TraceLoadError` with a clear message when the item is not an
    object or a required field is missing. The wording is shared with the web
    batch endpoint, so its error contract stays identical.
    """
    if not isinstance(item, dict):
        raise TraceLoadError("each item must be a JSON object with 'trace' and 'details'")
    trace_payload = item.get("trace") or {}
    details_payload = item.get("details") or {}
    if not isinstance(trace_payload, dict) or not isinstance(details_payload, dict):
        raise TraceLoadError("each item needs object 'trace' and 'details' fields")
    if not trace_payload.get("user_prompt"):
        raise TraceLoadError("trace.user_prompt is required for every item")
    if not details_payload.get("failure_mode"):
        raise TraceLoadError("details.failure_mode is required for every item")
    trace = TraceData(
        user_prompt=str(trace_payload["user_prompt"]),
        llm_output=str(trace_payload.get("llm_output") or ""),
        span_id=str(trace_payload.get("span_id") or ""),
    )
    return trace, FailureDetails.from_dict(details_payload)


def load_traces(path: str | Path) -> list[tuple[TraceData, FailureDetails]]:
    """Read a JSON traces file and return one ``(trace, details)`` pair per item.

    Raises :class:`TraceLoadError` when the file is missing, is not valid JSON,
    does not hold a list at the top level, or when any item fails validation.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TraceLoadError(f"cannot read traces file {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TraceLoadError(
            f"traces file {path} is not valid JSON: {exc.msg} at line {exc.lineno}"
        ) from exc
    if not isinstance(payload, list):
        raise TraceLoadError(
            f"traces file {path} must hold a JSON list of items, got {type(payload).__name__}"
        )
    return [parse_item(item) for item in payload]
