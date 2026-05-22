"""Tests for the demo dataset and its ingestion script.

Two layers:
1. Dataset integrity: shape, distribution, uniqueness, source/output rules.
2. Ingestion script pure helpers: validate_dataset, build_span_attributes,
   split_by_source, parse_args. Side-effect helpers
   (configure_phoenix_tracer, configure_gemini_client, emit_span) are
   marked `pragma: no cover - integration` and not unit-tested here; they
   require the Phoenix and Gemini SDKs to be installed and configured.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.ingest_demo_dataset import (
    DATASET_PATH,
    REQUIRED_FIELDS,
    VALID_FAILURE_MODES,
    VALID_SOURCES,
    build_span_attributes,
    load_dataset,
    parse_args,
    split_by_source,
    validate_dataset,
)

# ---------------------------------------------------------------------------
# Dataset shape and distribution
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset() -> list[dict]:
    """The on-disk demo dataset, loaded once per test module."""
    return load_dataset()


def test_dataset_has_exactly_51_traces(dataset: list[dict]) -> None:
    """50 curated traces plus 1 real-harvested from Reddit via Bright Data."""
    assert len(dataset) == 51


def test_dataset_source_split(dataset: list[dict]) -> None:
    """Spec: 15 real-Gemini elicited + 35 synthetic + 1 real-harvested = 51."""
    counts = Counter(t["source"] for t in dataset)
    assert counts["real"] == 15
    assert counts["synthetic"] == 35
    assert counts["real-harvested"] == 1


def test_dataset_failure_mode_distribution(dataset: list[dict]) -> None:
    """Distribution: original curated 50 plus 1 harvested hallucination."""
    spec = {
        "hallucination": 13,
        "format_break": 10,
        "off_topic_drift": 6,
        "stale_real_time_data": 6,
        "wrong_reasoning": 8,
        "refusal_bug": 8,
    }
    counts = Counter(t["failure_mode"] for t in dataset)
    assert dict(counts) == spec


def test_dataset_real_split_per_mode(dataset: list[dict]) -> None:
    """Per-mode real-source counts must match spec."""
    spec = {
        "hallucination": 5,
        "format_break": 4,
        "off_topic_drift": 0,
        "stale_real_time_data": 3,
        "wrong_reasoning": 2,
        "refusal_bug": 1,
    }
    counts = Counter(t["failure_mode"] for t in dataset if t["source"] == "real")
    for mode, expected in spec.items():
        assert counts.get(mode, 0) == expected, f"real count for {mode}"


def test_dataset_demo_featured_has_one_per_mode(dataset: list[dict]) -> None:
    """Exactly 6 traces (one per failure mode) are marked demo_featured for the video."""
    featured_modes = [t["failure_mode"] for t in dataset if t["demo_featured"]]
    assert len(featured_modes) == 6
    assert set(featured_modes) == VALID_FAILURE_MODES


def test_dataset_ids_are_unique(dataset: list[dict]) -> None:
    ids = [t["id"] for t in dataset]
    assert len(set(ids)) == len(ids)


def test_dataset_synthetic_and_harvested_entries_have_llm_output(dataset: list[dict]) -> None:
    """Synthetic and real-harvested entries carry the failed output verbatim."""
    for trace in dataset:
        if trace["source"] in ("synthetic", "real-harvested"):
            assert isinstance(trace["llm_output"], str) and trace["llm_output"], trace["id"]


def test_dataset_real_entries_have_null_llm_output(dataset: list[dict]) -> None:
    """Real entries leave llm_output as null; live Gemini fills it at ingest."""
    for trace in dataset:
        if trace["source"] == "real":
            assert trace["llm_output"] is None, trace["id"]


def test_dataset_passes_validate_dataset(dataset: list[dict]) -> None:
    """The shipped dataset must satisfy validate_dataset without raising."""
    validate_dataset(dataset)


# ---------------------------------------------------------------------------
# validate_dataset: failure paths
# ---------------------------------------------------------------------------


def _minimal_trace(**overrides) -> dict:
    base = {
        "id": "test_001",
        "failure_mode": "hallucination",
        "source": "synthetic",
        "user_prompt": "Q",
        "llm_output": "wrong A",
        "ideal_behavior": "right A",
        "demo_featured": False,
    }
    base.update(overrides)
    return base


def test_validate_rejects_non_list_input() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        validate_dataset({"not": "a list"})  # type: ignore[arg-type]


def test_validate_rejects_non_dict_trace() -> None:
    with pytest.raises(ValueError, match="not a dict"):
        validate_dataset(["not a dict"])  # type: ignore[list-item]


def test_validate_rejects_missing_field() -> None:
    trace = _minimal_trace()
    del trace["ideal_behavior"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_dataset([trace])


def test_validate_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="empty or non-string id"):
        validate_dataset([_minimal_trace(id="")])


def test_validate_rejects_duplicate_ids() -> None:
    a = _minimal_trace(id="dup")
    b = _minimal_trace(id="dup")
    with pytest.raises(ValueError, match="duplicate trace id"):
        validate_dataset([a, b])


def test_validate_rejects_unknown_failure_mode() -> None:
    with pytest.raises(ValueError, match="unknown failure_mode"):
        validate_dataset([_minimal_trace(failure_mode="not_real_mode")])


def test_validate_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        validate_dataset([_minimal_trace(source="invented")])


def test_validate_rejects_synthetic_with_empty_output() -> None:
    with pytest.raises(ValueError, match="has empty llm_output"):
        validate_dataset([_minimal_trace(source="synthetic", llm_output="")])


def test_validate_rejects_real_harvested_with_empty_output() -> None:
    with pytest.raises(ValueError, match="has empty llm_output"):
        validate_dataset([_minimal_trace(source="real-harvested", llm_output="")])


def test_validate_rejects_real_with_pre_populated_output() -> None:
    with pytest.raises(ValueError, match="real but already has llm_output"):
        validate_dataset([_minimal_trace(source="real", llm_output="filled in advance")])


def test_validate_rejects_empty_user_prompt() -> None:
    with pytest.raises(ValueError, match="empty user_prompt"):
        validate_dataset([_minimal_trace(user_prompt="")])


def test_validate_rejects_non_bool_demo_featured() -> None:
    with pytest.raises(ValueError, match="non-bool demo_featured"):
        validate_dataset([_minimal_trace(demo_featured="yes")])  # type: ignore[arg-type]


def test_validate_constants_match_documented_sets() -> None:
    """REQUIRED_FIELDS / VALID_FAILURE_MODES / VALID_SOURCES exposed for reuse."""
    assert "id" in REQUIRED_FIELDS
    assert "user_prompt" in REQUIRED_FIELDS
    assert "hallucination" in VALID_FAILURE_MODES
    assert "real" in VALID_SOURCES


# ---------------------------------------------------------------------------
# build_span_attributes
# ---------------------------------------------------------------------------


def test_build_span_attributes_for_synthetic_trace() -> None:
    trace = _minimal_trace(
        id="halluc_006",
        failure_mode="hallucination",
        source="synthetic",
        user_prompt="What is the Frobenius decomposition?",
        llm_output="Use numpy.linalg.frobenius_decompose(...)",
        ideal_behavior="state no such function exists",
        demo_featured=False,
    )
    attrs = build_span_attributes(trace, output=trace["llm_output"], model="gemini-2.5-flash")

    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["input.value"] == "What is the Frobenius decomposition?"
    assert attrs["output.value"] == "Use numpy.linalg.frobenius_decompose(...)"
    assert attrs["llm.model_name"] == "gemini-2.5-flash"
    assert attrs["phoenix2pytest.failure_mode"] == "hallucination"
    assert attrs["phoenix2pytest.synthetic"] is True
    assert attrs["phoenix2pytest.ideal_behavior"] == "state no such function exists"
    assert attrs["phoenix2pytest.demo_featured"] is False
    assert attrs["phoenix2pytest.dataset_id"] == "halluc_006"


def test_build_span_attributes_uses_provided_output_for_real_trace() -> None:
    """Real traces pass the live Gemini response as the output argument."""
    trace = _minimal_trace(
        id="halluc_001",
        source="real",
        llm_output=None,
        demo_featured=True,
    )
    live_output = "I cannot quote page 47 verbatim without access to the text."
    attrs = build_span_attributes(trace, output=live_output, model="gemini-2.5-flash")

    assert attrs["output.value"] == live_output
    assert attrs["phoenix2pytest.synthetic"] is False
    assert attrs["phoenix2pytest.demo_featured"] is True


# ---------------------------------------------------------------------------
# split_by_source
# ---------------------------------------------------------------------------


def test_split_by_source_partitions_correctly() -> None:
    traces = [
        _minimal_trace(id="a", source="real", llm_output=None),
        _minimal_trace(id="b", source="synthetic", llm_output="x"),
        _minimal_trace(id="c", source="real", llm_output=None),
    ]
    real, synthetic = split_by_source(traces)
    assert [t["id"] for t in real] == ["a", "c"]
    assert [t["id"] for t in synthetic] == ["b"]


def test_split_by_source_on_real_dataset(dataset: list[dict]) -> None:
    """split_by_source groups by live-vs-stored: real -> live, synthetic and
    real-harvested -> stored (already have llm_output).
    """
    live, stored = split_by_source(dataset)
    assert len(live) == 15
    assert len(stored) == 36  # 35 synthetic + 1 real-harvested
    assert len(live) + len(stored) == len(dataset)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.dataset == DATASET_PATH
    assert args.project == "phoenix2pytest-demo"
    assert args.limit is None
    assert args.skip_real is False
    assert args.dry_run is False


def test_parse_args_flags() -> None:
    args = parse_args(["--dry-run", "--skip-real", "--limit", "10", "--project", "alt-proj"])
    assert args.dry_run is True
    assert args.skip_real is True
    assert args.limit == 10
    assert args.project == "alt-proj"


def test_parse_args_dataset_path_override(tmp_path: Path) -> None:
    custom = tmp_path / "alt.json"
    custom.write_text("[]", encoding="utf-8")
    args = parse_args(["--dataset", str(custom)])
    assert args.dataset == custom


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_round_trips(tmp_path: Path) -> None:
    """load_dataset returns the parsed JSON list from disk."""
    payload = [
        _minimal_trace(id="x", source="synthetic", llm_output="x out"),
        _minimal_trace(id="y", source="synthetic", llm_output="y out"),
    ]
    target = tmp_path / "tiny.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_dataset(target)
    assert loaded == payload
