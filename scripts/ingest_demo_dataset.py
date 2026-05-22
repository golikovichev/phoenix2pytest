"""Ingest the 50-trace demo dataset into Phoenix.

Reads tests/data/demo_dataset.json and emits one OTEL span per trace into
the `phoenix2pytest-demo` Phoenix project. Real entries call Gemini live
to capture the actual output; synthetic entries use the hand-curated
llm_output from the JSON.

The script is idempotent in the sense that a second run produces a fresh
batch of spans in the same project (Phoenix groups by trace id, not by
user input). To start from a clean dataset, delete the project in the
Phoenix UI and re-run.

Environment variables (loaded via python-dotenv if a .env file is
present):

- `PHOENIX_BASE_URL`: base URL of the Phoenix instance
- `PHOENIX_API_KEY`: API key for the Phoenix workspace
- `GEMINI_API_KEY`: optional, only required if the dataset contains real
  entries that need live Gemini calls
- `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`: optional Vertex AI
  routing for the Gemini client

Usage::

    python scripts/ingest_demo_dataset.py
    python scripts/ingest_demo_dataset.py --dry-run   # validate dataset, skip ingestion
    python scripts/ingest_demo_dataset.py --limit 5   # ingest first N entries

Cost: live Gemini calls for 15 real entries average about $0.01 total
on gemini-2.5-flash. Synthetic entries cost nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "tests" / "data" / "demo_dataset.json"
DEFAULT_PROJECT_NAME = "phoenix2pytest-demo"

REQUIRED_FIELDS = {
    "id",
    "failure_mode",
    "source",
    "user_prompt",
    "llm_output",
    "ideal_behavior",
    "demo_featured",
}

VALID_FAILURE_MODES = {
    "hallucination",
    "format_break",
    "off_topic_drift",
    "stale_real_time_data",
    "wrong_reasoning",
    "refusal_bug",
}

VALID_SOURCES = {"real", "synthetic", "real-harvested"}


# ---------------------------------------------------------------------------
# Pure helpers: dataset loading, validation, span attribute construction
# ---------------------------------------------------------------------------


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load the demo dataset from JSON on disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset(traces: list[dict[str, Any]]) -> None:
    """Raise ValueError if the dataset is malformed.

    Checks: every trace has the required fields, failure_mode and source
    are recognised, and synthetic / real-harvested entries have a
    non-null llm_output while real entries have llm_output=None.

    Source semantics:
        real            - prompt only; llm_output is regenerated live
                          against a real model at ingestion time.
        synthetic       - prompt and a curated bad output, both invented
                          by the dataset author for demonstration.
        real-harvested  - prompt and the actual bad output captured from
                          a real LLM session that someone reported online.
                          Distinct from synthetic so the source attribution
                          stays honest in the dataset.
    """
    if not isinstance(traces, list):
        raise ValueError(f"dataset must be a list, got {type(traces).__name__}")

    seen_ids: set[str] = set()
    for index, trace in enumerate(traces):
        if not isinstance(trace, dict):
            raise ValueError(f"trace at index {index} is not a dict")
        missing = REQUIRED_FIELDS - set(trace.keys())
        if missing:
            raise ValueError(f"trace {trace.get('id', index)!r} missing fields: {sorted(missing)}")

        trace_id = trace["id"]
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError(f"trace at index {index} has empty or non-string id")
        if trace_id in seen_ids:
            raise ValueError(f"duplicate trace id {trace_id!r}")
        seen_ids.add(trace_id)

        if trace["failure_mode"] not in VALID_FAILURE_MODES:
            raise ValueError(
                f"trace {trace_id!r} has unknown failure_mode {trace['failure_mode']!r}; "
                f"expected one of {sorted(VALID_FAILURE_MODES)}"
            )
        if trace["source"] not in VALID_SOURCES:
            raise ValueError(
                f"trace {trace_id!r} has unknown source {trace['source']!r}; "
                f"expected one of {sorted(VALID_SOURCES)}"
            )
        if trace["source"] in ("synthetic", "real-harvested") and not trace["llm_output"]:
            raise ValueError(
                f"trace {trace_id!r} source={trace['source']!r} but has empty llm_output; "
                "synthetic and real-harvested entries must include the failed output verbatim"
            )
        if trace["source"] == "real" and trace["llm_output"] is not None:
            raise ValueError(
                f"trace {trace_id!r} is real but already has llm_output; "
                "real entries should have llm_output=null and be filled at ingestion"
            )
        if not isinstance(trace["user_prompt"], str) or not trace["user_prompt"]:
            raise ValueError(f"trace {trace_id!r} has empty user_prompt")
        if not isinstance(trace["demo_featured"], bool):
            raise ValueError(
                f"trace {trace_id!r} has non-bool demo_featured: {trace['demo_featured']!r}"
            )


def build_span_attributes(trace: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    """Build the attribute dict for the OTEL span representing one trace.

    Uses OpenInference semantic conventions for the input/output fields
    so Phoenix renders the span correctly, plus custom
    `phoenix2pytest.*` attrs that the extractor reads later.
    """
    return {
        "openinference.span.kind": "LLM",
        "input.value": trace["user_prompt"],
        "output.value": output,
        "llm.model_name": model,
        "phoenix2pytest.failure_mode": trace["failure_mode"],
        "phoenix2pytest.synthetic": trace["source"] == "synthetic",
        "phoenix2pytest.ideal_behavior": trace["ideal_behavior"],
        "phoenix2pytest.demo_featured": trace["demo_featured"],
        "phoenix2pytest.dataset_id": trace["id"],
    }


def split_by_source(traces: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Return (live_traces, stored_traces).

    live_traces are 'real' source entries whose llm_output must be
    generated by calling Gemini live at ingestion.

    stored_traces are 'synthetic' and 'real-harvested' entries that
    already carry the failed llm_output verbatim and only need
    emission to Phoenix.
    """
    live = [t for t in traces if t["source"] == "real"]
    stored = [t for t in traces if t["source"] in ("synthetic", "real-harvested")]
    return live, stored


# ---------------------------------------------------------------------------
# Side-effect helpers: Phoenix tracer + Gemini client setup
# ---------------------------------------------------------------------------


def configure_phoenix_tracer(project_name: str):  # pragma: no cover - integration
    """Wire up the Phoenix OTEL exporter and return a tracer.

    Imported lazily so that unit tests can exercise pure helpers without
    requiring opentelemetry / phoenix-otel installed.
    """
    from opentelemetry import trace as otel_trace
    from phoenix.otel import register

    base_url = os.environ.get("PHOENIX_BASE_URL")
    api_key = os.environ.get("PHOENIX_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "PHOENIX_BASE_URL and PHOENIX_API_KEY must be set to ingest into Phoenix"
        )
    register(
        project_name=project_name,
        endpoint=f"{base_url}/v1/traces",
        protocol="http/protobuf",
        batch=True,
    )
    return otel_trace.get_tracer("phoenix2pytest.ingest_demo_dataset")


def configure_gemini_client(model: str = "gemini-2.5-flash"):  # pragma: no cover - integration
    """Build a Gemini client for live calls.

    Supports two auth paths:
    - Direct Gemini API: GEMINI_API_KEY or GOOGLE_API_KEY set.
    - Vertex AI: GOOGLE_GENAI_USE_VERTEXAI=True plus GOOGLE_CLOUD_PROJECT
      and GOOGLE_CLOUD_LOCATION set (Application Default Credentials
      handle auth, no API key needed).

    Raises if neither path is configured.
    """
    from google import genai
    from google.genai.types import HttpOptions

    has_api_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"true", "1", "yes"}
    has_vertex = use_vertex and bool(
        os.environ.get("GOOGLE_CLOUD_PROJECT") and os.environ.get("GOOGLE_CLOUD_LOCATION")
    )

    if not has_api_key and not has_vertex:
        raise RuntimeError(
            "Live Gemini calls require either GEMINI_API_KEY (or GOOGLE_API_KEY) "
            "for the direct API path, OR GOOGLE_GENAI_USE_VERTEXAI=True plus "
            "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION for Vertex AI. "
            "Use --skip-real to bypass and emit '[skipped]' for real entries."
        )
    return genai.Client(http_options=HttpOptions(api_version="v1")), model


def call_gemini(client, model: str, prompt: str) -> str:  # pragma: no cover - integration
    """Send the prompt to Gemini and return the text response."""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


# ---------------------------------------------------------------------------
# Ingestion driver
# ---------------------------------------------------------------------------


def emit_span(  # pragma: no cover - integration
    tracer,
    trace: dict[str, Any],
    output: str,
    model: str,
) -> None:
    """Emit one OTEL span representing the trace."""
    span_name = f"llm.{trace['failure_mode']}.{trace['id']}"
    attributes = build_span_attributes(trace, output, model)
    with tracer.start_as_current_span(span_name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)


def run_ingestion(  # pragma: no cover - integration
    *,
    dataset_path: Path,
    project_name: str,
    limit: int | None,
    skip_real: bool,
    dry_run: bool,
) -> int:
    """End-to-end driver. Returns process exit code."""
    traces = load_dataset(dataset_path)
    validate_dataset(traces)
    if limit is not None:
        traces = traces[:limit]
    real, synthetic = split_by_source(traces)
    print(f"dataset: {len(traces)} traces ({len(real)} real, {len(synthetic)} synthetic)")

    if dry_run:
        print("dry-run: dataset validated, no ingestion performed")
        return 0

    tracer = configure_phoenix_tracer(project_name)

    gemini_client = None
    gemini_model = "gemini-2.5-flash"
    needs_gemini = any(t["source"] == "real" for t in traces) and not skip_real
    if needs_gemini:
        gemini_client, gemini_model = configure_gemini_client(gemini_model)
        print(f"gemini client ready: model {gemini_model}")
    elif skip_real:
        print("skip-real: real entries will be ingested with llm_output='[skipped]'")

    for trace in traces:
        if trace["source"] == "real":
            if skip_real or gemini_client is None:
                output = "[skipped]"
            else:
                output = call_gemini(gemini_client, gemini_model, trace["user_prompt"])
        else:
            output = trace["llm_output"]
        emit_span(tracer, trace, output, gemini_model)
        print(f"  emitted {trace['id']} ({trace['failure_mode']}, {trace['source']})")

    print(f"done: {len(traces)} spans emitted to Phoenix project {project_name!r}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI argument parser, kept pure so tests can drive it."""
    parser = argparse.ArgumentParser(
        description="Ingest the demo dataset into Phoenix",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to demo_dataset.json (default: tests/data/demo_dataset.json)",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT_NAME,
        help=f"Phoenix project name (default: {DEFAULT_PROJECT_NAME})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N traces (default: all)",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Do not call Gemini for real entries; emit '[skipped]' as output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and exit without ingesting",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - integration
    """CLI entry point."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional; env may be set by other means

    args = parse_args(argv)
    return run_ingestion(
        dataset_path=args.dataset,
        project_name=args.project,
        limit=args.limit,
        skip_real=args.skip_real,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":  # pragma: no cover - script entry
    sys.exit(main())
