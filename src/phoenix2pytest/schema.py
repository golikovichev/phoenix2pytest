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
authoritative vocabulary. Downstream runtime collections derive from these
Literals via :func:`typing.get_args` (see :data:`FAILURE_MODE_VALUES` and
:data:`ASSERTION_STRATEGY_VALUES`), so the demo dataset and the live
pipeline cannot drift apart by accident.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

FailureMode = Literal[
    "hallucination",
    "format_break",
    "off_topic_drift",
    "stale_real_time_data",
    "wrong_reasoning",
    "refusal_bug",
]
"""Closed set of failure modes the pipeline recognises."""

FAILURE_MODE_VALUES: tuple[str, ...] = get_args(FailureMode)
"""Tuple of failure mode strings derived from the :data:`FailureMode` Literal.

This is the single source of truth re-exported for downstream code that needs
a runtime collection (e.g. ``scripts.ingest_demo_dataset.VALID_FAILURE_MODES``
sets itself from this tuple). Any change to the Literal propagates here
automatically, so the dataset cannot drift from the schema by accident.
"""

AssertionStrategy = Literal[
    "substring_excluded",
    "regex_excluded",
    "format_must_match",
    "answer_must_be_exact",
    "refusal_marker_required",
]
"""Strategies the synthesiser knows how to turn into pytest assertions."""

ASSERTION_STRATEGY_VALUES: tuple[str, ...] = get_args(AssertionStrategy)
"""Tuple of assertion-strategy strings derived from :data:`AssertionStrategy`."""


def _stripped_non_empty(value: object) -> object:
    """Reject blank or whitespace-only strings; return the stripped form.

    Non-string inputs are passed through unchanged so pydantic's own type
    validation can reject them with a clear ``string_type`` error. The
    previous version coerced via ``str(value)`` which silently turned
    non-strings into their string form, masking the original type.
    """
    if not isinstance(value, str):
        return value
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
    def _require_non_blank(cls, value: object) -> object:
        return _stripped_non_empty(value)

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
    def _require_non_blank(cls, value: object) -> object:
        return _stripped_non_empty(value)

    @field_validator("key_strings_to_exclude", "key_patterns_required", mode="before")
    @classmethod
    def _drop_blank_list_items(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item.strip()]
        return value


__all__ = [
    "ASSERTION_STRATEGY_VALUES",
    "FAILURE_MODE_VALUES",
    "AssertionStrategy",
    "ExtractorResponse",
    "FailureMode",
    "TraceScenario",
]
