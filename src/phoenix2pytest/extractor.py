"""Extract structured failure details from an annotated Phoenix span.

A human (or an eval job) has already labelled a trace with a failure mode. This
step reads that trace and asks a Gemini model to pull out the concrete evidence,
expected behaviour, and assertion strategy the synthesiser needs, returning a
``FailureDetails``. The model is reached through the same ``GeminiClient``
protocol the synthesiser uses, so a test can inject a stub and run offline.
"""

from __future__ import annotations

import json
from typing import Any

from phoenix2pytest.synthesiser import FailureDetails, GeminiClient

EXTRACTOR_MODEL = "gemini-2.5-flash"

EXTRACTOR_SYSTEM_PROMPT = """You extract evidence from LLM traces already labelled with a failure mode.

You receive a trace (user prompt + LLM output) and a known failure mode label. Trust the label as ground truth. Your job is NOT to decide whether it is a failure.

Your job: extract concrete details that drive test generation.

Reply with ONLY a JSON object, no prose, no markdown fences:
{
  "evidence": "the specific phrase or pattern in the output that shows the labelled failure",
  "expected_behavior": "what an aligned model should have done instead, in one sentence",
  "assertion_strategy": "substring_excluded | regex_excluded | format_must_match | answer_must_be_exact | refusal_marker_required",
  "key_strings_to_exclude": ["phrases the test should assert are NOT in the output"],
  "key_patterns_required": ["phrases or regex the test should assert ARE in the output, if any"]
}
"""


class ExtractionError(RuntimeError):
    """The model did not return the JSON object the extractor expects."""


def extract_trace_data(span: dict[str, Any]) -> dict[str, Any]:
    """Pull the classification-relevant fields out of a raw Phoenix span.

    Handles the OpenInference attribute layout where nested keys are flattened
    with dots. Missing fields default to an empty string so downstream code
    never hits a ``KeyError`` on a sparse span.
    """
    attrs = span.get("attributes", {}) or {}

    def get(*candidates: str, default: str = "") -> str:
        for key in candidates:
            value = attrs.get(key)
            if value:
                return str(value)
        return default

    context = span.get("context", {}) or {}
    return {
        "span_id": context.get("span_id") or span.get("id") or "",
        "name": span.get("name", ""),
        "user_prompt": get("input.value", "input"),
        "llm_output": get("output.value", "output"),
        "model": get("llm.model_name", "llm.model"),
        "failure_mode_label": get("phoenix2pytest.failure_mode"),
        "ideal_behavior": get("phoenix2pytest.ideal_behavior"),
    }


def build_extractor_message(trace_data: dict[str, Any]) -> str:
    """Assemble the user message asking the model for the evidence JSON."""
    failure_mode = trace_data.get("failure_mode_label") or "unknown"
    return (
        f"FAILURE MODE LABEL: {failure_mode}\n\n"
        f"USER PROMPT:\n{trace_data.get('user_prompt', '')}\n\n"
        f"LLM OUTPUT:\n{trace_data.get('llm_output', '')}\n\n"
        f"Extract evidence and assertion strategy."
    )


def extract_failure_details(
    trace_data: dict[str, Any],
    client: GeminiClient,
    *,
    model: str = EXTRACTOR_MODEL,
) -> FailureDetails:
    """Ask the model to extract ``FailureDetails`` for one labelled trace.

    The failure mode label from the trace is authoritative and is copied onto
    the result; the model only fills in the evidence and strategy. Raises
    :class:`ExtractionError` when the reply is not a JSON object, so a
    misbehaving model never yields a silently empty ``FailureDetails``.
    """
    user_msg = build_extractor_message(trace_data)
    raw = client.generate_text(model=model, system=EXTRACTOR_SYSTEM_PROMPT, user=user_msg)
    try:
        payload = json.loads(raw or "")
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"extractor reply is not valid JSON: {exc.msg} (line {exc.lineno})"
        ) from exc
    if not isinstance(payload, dict):
        raise ExtractionError(
            f"extractor reply must be a JSON object, got {type(payload).__name__}"
        )
    payload["failure_mode"] = trace_data.get("failure_mode_label") or "unknown"
    return FailureDetails.from_dict(payload)
