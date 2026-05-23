"""Tests for the trace scenario and extractor response schemas.

Covers:
* Construction with minimum required fields.
* Optional metadata pass-through and blank-string normalisation.
* Rejection of blank required fields, unknown failure modes, unknown
  assertion strategies, negative token counts, and extra unknown fields.
* List field defaults and blank-item filtering.
* JSON round-trip via ``model_dump`` / ``model_validate``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from phoenix2pytest.schema import (
    ASSERTION_STRATEGY_VALUES,
    FAILURE_MODE_VALUES,
    AssertionStrategy,
    ExtractorResponse,
    FailureMode,
    TraceScenario,
)
from scripts.ingest_demo_dataset import VALID_FAILURE_MODES

# ---------------------------------------------------------------------------
# Literal vocabulary stays in sync with the demo dataset
# ---------------------------------------------------------------------------


def test_failure_mode_matches_demo_dataset_vocabulary() -> None:
    """The schema's FailureMode must mirror VALID_FAILURE_MODES exactly.

    With the schema as the canonical source (ingest_demo_dataset.py derives its
    set from FAILURE_MODE_VALUES), this is a belt-and-braces invariant rather
    than the sole gate, but it still catches accidental local overrides.
    """
    schema_modes = set(FailureMode.__args__)  # type: ignore[attr-defined]
    assert schema_modes == VALID_FAILURE_MODES


def test_failure_mode_values_tuple_matches_literal() -> None:
    """The derived FAILURE_MODE_VALUES tuple must match the Literal exactly."""
    assert set(FAILURE_MODE_VALUES) == set(FailureMode.__args__)  # type: ignore[attr-defined]
    assert isinstance(FAILURE_MODE_VALUES, tuple)


def test_assertion_strategy_values_tuple_matches_literal() -> None:
    """The derived ASSERTION_STRATEGY_VALUES tuple must match the Literal exactly."""
    assert set(ASSERTION_STRATEGY_VALUES) == set(AssertionStrategy.__args__)  # type: ignore[attr-defined]
    assert isinstance(ASSERTION_STRATEGY_VALUES, tuple)


def test_assertion_strategy_exposes_five_known_strategies() -> None:
    """AssertionStrategy is the closed set the synthesiser knows how to encode."""
    expected = {
        "substring_excluded",
        "regex_excluded",
        "format_must_match",
        "answer_must_be_exact",
        "refusal_marker_required",
    }
    assert set(AssertionStrategy.__args__) == expected  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TraceScenario
# ---------------------------------------------------------------------------


def _minimal_scenario_kwargs() -> dict:
    return {
        "user_prompt": "What is 2+2?",
        "llm_output": "5",
        "failure_mode": "wrong_reasoning",
    }


def test_trace_scenario_accepts_minimum_fields() -> None:
    scenario = TraceScenario(**_minimal_scenario_kwargs())
    assert scenario.user_prompt == "What is 2+2?"
    assert scenario.llm_output == "5"
    assert scenario.failure_mode == "wrong_reasoning"
    assert scenario.ideal_behavior is None
    assert scenario.model is None
    assert scenario.span_id is None
    assert scenario.dataset_id is None
    assert scenario.tokens_total == 0


def test_trace_scenario_preserves_optional_metadata() -> None:
    scenario = TraceScenario(
        **_minimal_scenario_kwargs(),
        ideal_behavior="Answer 4.",
        model="gemini-2.5-flash",
        span_id="abc-123",
        dataset_id="halluc_001",
        tokens_total=128,
    )
    assert scenario.ideal_behavior == "Answer 4."
    assert scenario.model == "gemini-2.5-flash"
    assert scenario.span_id == "abc-123"
    assert scenario.dataset_id == "halluc_001"
    assert scenario.tokens_total == 128


def test_trace_scenario_strips_required_strings() -> None:
    scenario = TraceScenario(
        user_prompt="  hi  ",
        llm_output="\nresponse\n",
        failure_mode="hallucination",
    )
    assert scenario.user_prompt == "hi"
    assert scenario.llm_output == "response"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_trace_scenario_rejects_blank_user_prompt(blank: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        TraceScenario(
            user_prompt=blank,
            llm_output="something",
            failure_mode="hallucination",
        )


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_trace_scenario_rejects_blank_llm_output(blank: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        TraceScenario(
            user_prompt="q",
            llm_output=blank,
            failure_mode="hallucination",
        )


def test_trace_scenario_rejects_unknown_failure_mode() -> None:
    with pytest.raises(ValidationError):
        TraceScenario(
            user_prompt="q",
            llm_output="a",
            failure_mode="not_a_real_mode",  # type: ignore[arg-type]
        )


def test_trace_scenario_rejects_negative_token_count() -> None:
    with pytest.raises(ValidationError):
        TraceScenario(**_minimal_scenario_kwargs(), tokens_total=-1)


def test_trace_scenario_accepts_zero_tokens_explicitly() -> None:
    """``tokens_total=0`` is the deliberate default boundary, not an accident.

    Without this test, a future contributor could change ``ge=0`` to ``gt=0``
    without anything failing - the minimum-fields test only asserts the
    implicit default, not that zero is an allowed explicit value.
    """
    scenario = TraceScenario(**_minimal_scenario_kwargs(), tokens_total=0)
    assert scenario.tokens_total == 0


def test_trace_scenario_coerces_numeric_string_token_count() -> None:
    """Pydantic v2 default behaviour coerces numeric strings to int.

    Pins the chosen behaviour: when Gemini returns ``"tokens_total": "128"``
    (e.g. via a JSON string field), pydantic accepts and coerces. If we ever
    want strict mode, this test fails first and forces a decision.
    """
    scenario = TraceScenario(**_minimal_scenario_kwargs(), tokens_total="128")  # type: ignore[arg-type]
    assert scenario.tokens_total == 128


def test_trace_scenario_rejects_non_numeric_token_count() -> None:
    """Non-numeric strings still fail validation - coercion is not unconditional."""
    with pytest.raises(ValidationError):
        TraceScenario(**_minimal_scenario_kwargs(), tokens_total="lots")  # type: ignore[arg-type]


def test_trace_scenario_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TraceScenario(**_minimal_scenario_kwargs(), unknown="x")


def test_trace_scenario_blank_optional_strings_become_none() -> None:
    scenario = TraceScenario(
        **_minimal_scenario_kwargs(),
        ideal_behavior="   ",
        model="",
        span_id=None,
    )
    assert scenario.ideal_behavior is None
    assert scenario.model is None
    assert scenario.span_id is None


def test_trace_scenario_round_trips_json() -> None:
    original = TraceScenario(
        **_minimal_scenario_kwargs(),
        ideal_behavior="Answer 4.",
        model="gemini-2.5-flash",
        tokens_total=42,
    )
    payload = original.model_dump()
    restored = TraceScenario.model_validate(payload)
    assert restored == original


def test_trace_scenario_round_trips_json_string() -> None:
    """End-to-end JSON string round-trip catches encoder edge cases that the
    dict-level round-trip misses (None serialisation, custom encoders).
    """
    original = TraceScenario(
        **_minimal_scenario_kwargs(),
        ideal_behavior=None,
        model="gemini-2.5-flash",
        tokens_total=42,
    )
    payload = original.model_dump_json()
    assert isinstance(payload, str)
    restored = TraceScenario.model_validate_json(payload)
    assert restored == original


# ---------------------------------------------------------------------------
# ExtractorResponse
# ---------------------------------------------------------------------------


def _minimal_response_kwargs() -> dict:
    return {
        "failure_mode": "hallucination",
        "evidence": "claims numpy.linalg.frobenius_decompose exists",
        "expected_behavior": "State that no such function exists in numpy.",
        "assertion_strategy": "substring_excluded",
    }


def test_extractor_response_accepts_minimum_fields() -> None:
    response = ExtractorResponse(**_minimal_response_kwargs())
    assert response.failure_mode == "hallucination"
    assert response.assertion_strategy == "substring_excluded"
    assert response.key_strings_to_exclude == []
    assert response.key_patterns_required == []


def test_extractor_response_preserves_lists() -> None:
    response = ExtractorResponse(
        **_minimal_response_kwargs(),
        key_strings_to_exclude=["frobenius_decompose", "numpy.linalg.frobenius"],
        key_patterns_required=["no such function"],
    )
    assert response.key_strings_to_exclude == [
        "frobenius_decompose",
        "numpy.linalg.frobenius",
    ]
    assert response.key_patterns_required == ["no such function"]


@pytest.mark.parametrize("field", ["key_strings_to_exclude", "key_patterns_required"])
def test_extractor_response_filters_blank_list_items(field: str) -> None:
    kwargs = _minimal_response_kwargs()
    kwargs[field] = ["real", "", "  ", "also_real"]
    response = ExtractorResponse(**kwargs)
    assert getattr(response, field) == ["real", "also_real"]


def test_extractor_response_treats_none_lists_as_empty() -> None:
    response = ExtractorResponse(
        **_minimal_response_kwargs(),
        key_strings_to_exclude=None,  # type: ignore[arg-type]
        key_patterns_required=None,  # type: ignore[arg-type]
    )
    assert response.key_strings_to_exclude == []
    assert response.key_patterns_required == []


@pytest.mark.parametrize("blank", ["", "   "])
def test_extractor_response_rejects_blank_evidence(blank: str) -> None:
    kwargs = _minimal_response_kwargs()
    kwargs["evidence"] = blank
    with pytest.raises(ValidationError, match="must not be blank"):
        ExtractorResponse(**kwargs)


@pytest.mark.parametrize("blank", ["", "\n"])
def test_extractor_response_rejects_blank_expected_behavior(blank: str) -> None:
    kwargs = _minimal_response_kwargs()
    kwargs["expected_behavior"] = blank
    with pytest.raises(ValidationError, match="must not be blank"):
        ExtractorResponse(**kwargs)


def test_extractor_response_rejects_unknown_assertion_strategy() -> None:
    kwargs = _minimal_response_kwargs()
    kwargs["assertion_strategy"] = "llm_as_judge"
    with pytest.raises(ValidationError):
        ExtractorResponse(**kwargs)


def test_extractor_response_rejects_unknown_failure_mode() -> None:
    kwargs = _minimal_response_kwargs()
    kwargs["failure_mode"] = "made_up_mode"
    with pytest.raises(ValidationError):
        ExtractorResponse(**kwargs)


def test_extractor_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ExtractorResponse(**_minimal_response_kwargs(), confidence=0.9)


def test_extractor_response_round_trips_json() -> None:
    original = ExtractorResponse(
        **_minimal_response_kwargs(),
        key_strings_to_exclude=["a", "b"],
        key_patterns_required=["c"],
    )
    payload = original.model_dump()
    restored = ExtractorResponse.model_validate(payload)
    assert restored == original
