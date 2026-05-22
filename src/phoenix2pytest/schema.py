"""Pydantic models for trace scenarios and extractor output.

The vertical-slice pipeline (Phoenix span -> Gemini extractor -> Gemini
synthesiser -> generated pytest file) currently passes plain dicts between
steps. That works for one trace but loses every contract guarantee the moment
the dataset grows: required fields are not enforced, typos in assertion
strategy names are silent, and JSON shape drift between Gemini calls is hard
to detect.

This module locks the wire shapes down with pydantic v2 models so each step
gets a typed payload:

* :class:`TraceScenario` - the trace data extracted from a Phoenix span,
  consumed by the failure-mode extractor.
* :class:`ExtractorResponse` - the structured JSON returned by the Gemini
  extractor, consumed by the test synthesiser.

The :data:`FailureMode` and :data:`AssertionStrategy` literals are the
authoritative vocabulary. ``FailureMode`` mirrors
``scripts.ingest_demo_dataset.VALID_FAILURE_MODES`` to keep the demo dataset
and the live pipeline aligned.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FailureMode = Literal[
    "hallucination",
    "format_break",
    "off_topic_drift",
    "stale_real_time_data",
    "wrong_reasoning",
    "refusal_bug",
]
"""Closed set of failure modes the pipeline recognises.

Must stay in sync with ``scripts.ingest_demo_dataset.VALID_FAILURE_MODES``.
"""

AssertionStrategy = Literal[
    "substring_excluded",
    "regex_excluded",
    "format_must_match",
    "answer_must_be_exact",
    "refusal_marker_required",
]
"""Strategies the synthesiser knows how to turn into pytest assertions."""


def _stripped_non_empty(value: str) -> str:
    """Reject blank or whitespace-only strings; return the stripped form."""
    if not isinstance(value, str):
        raise TypeError("expected a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("must not be blank")
    return cleaned


class TraceScenario(BaseModel):
    """One failed trace, ready for the extractor step.

    A trace scenario is the bridge between Phoenix (span ingestion) and Gemini
    (failure-evidence extraction). The two free-text fields, ``user_prompt``
    and ``llm_output``, are the minimum the extractor needs to do its job.
    Everything else is best-effort metadata copied from span attributes.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    user_prompt: str = Field(..., description="The prompt sent to the model.")
    llm_output: str = Field(..., description="The model's response, verbatim.")
    failure_mode: FailureMode = Field(
        ..., description="The labelled failure category for this trace."
    )
    ideal_behavior: str | None = Field(
        default=None,
        description="What an aligned model should have done, if recorded.",
    )
    model: str | None = Field(default=None, description="Model name (e.g. gemini-2.5-flash).")
    span_id: str | None = Field(
        default=None, description="Phoenix span ID, when sourced from Phoenix."
    )
    dataset_id: str | None = Field(
        default=None, description="Dataset row ID, when sourced from a curated set."
    )
    tokens_total: int = Field(
        default=0, ge=0, description="Total tokens reported by the span, if any."
    )

    @field_validator("user_prompt", "llm_output", mode="before")
    @classmethod
    def _require_non_blank(cls, value: object) -> str:
        return _stripped_non_empty(value if isinstance(value, str) else str(value))

    @field_validator("ideal_behavior", "model", "span_id", "dataset_id", mode="before")
    @classmethod
    def _normalise_optional_blank(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value


class ExtractorResponse(BaseModel):
    """Structured output of the Gemini extractor prompt.

    The extractor receives a :class:`TraceScenario` and returns the concrete
    evidence plus an assertion plan that the synthesiser turns into a pytest
    file. Two list fields default to empty lists so consumers can iterate
    without guarding for ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    failure_mode: FailureMode = Field(
        ..., description="Echoed from the input scenario to keep payloads self-contained."
    )
    evidence: str = Field(
        ..., description="Specific phrase or pattern in the output that proves the failure."
    )
    expected_behavior: str = Field(
        ..., description="What an aligned model should have done, in one sentence."
    )
    assertion_strategy: AssertionStrategy = Field(
        ..., description="Concrete strategy the synthesiser will encode."
    )
    key_strings_to_exclude: list[str] = Field(
        default_factory=list,
        description="Substrings the regression test asserts are NOT in the output.",
    )
    key_patterns_required: list[str] = Field(
        default_factory=list,
        description="Patterns or substrings the regression test asserts ARE in the output.",
    )

    @field_validator("evidence", "expected_behavior", mode="before")
    @classmethod
    def _require_non_blank(cls, value: object) -> str:
        return _stripped_non_empty(value if isinstance(value, str) else str(value))

    @field_validator("key_strings_to_exclude", "key_patterns_required", mode="before")
    @classmethod
    def _drop_blank_list_items(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item.strip()]
        return value


__all__ = [
    "AssertionStrategy",
    "ExtractorResponse",
    "FailureMode",
    "TraceScenario",
]
