"""Generate sample failed LLM traces and ingest them into Phoenix Cloud via OTEL.

Produces 5 sample failed traces by deliberately prompting Gemini with edge
cases, then sends them through the OTEL exporter to Phoenix Cloud. Used to
validate the end-to-end ingestion pipeline before scaling to larger batches.

Usage:
    python scripts/trace_generator.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register

load_dotenv()


def configure_phoenix_tracing() -> trace.Tracer:
    """Set up Phoenix OTEL exporter and return a tracer."""
    base_url = os.environ.get("PHOENIX_BASE_URL")
    api_key = os.environ.get("PHOENIX_API_KEY")

    if not base_url or not api_key:
        sys.stderr.write("ERROR: PHOENIX_BASE_URL and PHOENIX_API_KEY must be set\n")
        sys.exit(1)

    # Phoenix Cloud OTLP HTTP traces endpoint lives at <base>/v1/traces
    register(
        project_name="phoenix2pytest-dryrun",
        endpoint=f"{base_url}/v1/traces",
        protocol="http/protobuf",
        batch=True,
    )
    return trace.get_tracer("phoenix2pytest.trace_generator")


def configure_gemini() -> genai.Client:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    return genai.Client(http_options=HttpOptions(api_version="v1"))


FAILURE_SCENARIOS = [
    {
        "mode": "hallucination",
        "user_prompt": (
            "Tell me three specific lines of dialogue from page 47 of "
            "'The Great Gatsby' by F. Scott Fitzgerald. Quote them exactly."
        ),
        "ideal_behavior": "refuse or admit the model cannot quote a specific page",
    },
    {
        "mode": "format_break",
        "user_prompt": (
            "Return ONLY a JSON object with fields 'name' and 'age', "
            "nothing else, no prose. The user said: 'hi how are you'"
        ),
        "ideal_behavior": "valid JSON {} or refuse - must not return prose",
    },
    {
        "mode": "off_topic_drift",
        "user_prompt": "Answer in exactly one word. What is the capital of France?",
        "ideal_behavior": "single word: Paris",
    },
    {
        "mode": "stale_real_time_data",
        "user_prompt": "What is today's closing stock price for AAPL?",
        "ideal_behavior": "refuse - model has no real-time data access",
    },
    {
        "mode": "wrong_reasoning",
        "user_prompt": (
            "Train A leaves NYC at 3pm going 60mph. Train B leaves LA at 4pm "
            "going 80mph. When do they meet?"
        ),
        "ideal_behavior": "ask for the distance between cities",
    },
]


def run_scenario(tracer: trace.Tracer, client: genai.Client, scenario: dict) -> None:
    span_name = f"llm.{scenario['mode']}"

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, "gemini-2.5-flash")
        span.set_attribute(SpanAttributes.INPUT_VALUE, scenario["user_prompt"])
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "text/plain")
        span.set_attribute("phoenix2pytest.failure_mode", scenario["mode"])
        span.set_attribute("phoenix2pytest.ideal_behavior", scenario["ideal_behavior"])

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=scenario["user_prompt"],
            )
            output = response.text or ""
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, output)
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")

            if response.usage_metadata:
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                    response.usage_metadata.prompt_token_count,
                )
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                    response.usage_metadata.candidates_token_count or 0,
                )
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                    response.usage_metadata.total_token_count,
                )

            print(f"\n[{scenario['mode']}]")
            print(f"  prompt:  {scenario['user_prompt'][:80]}...")
            print(f"  ideal:   {scenario['ideal_behavior']}")
            print(f"  actual:  {output[:200]}")

        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            print(f"\n[{scenario['mode']}] ERROR: {exc}", file=sys.stderr)


def main() -> int:
    print("=" * 60)
    print("Phoenix2Pytest: trace generator dry-run")
    print("=" * 60)

    tracer = configure_phoenix_tracing()
    client = configure_gemini()

    for scenario in FAILURE_SCENARIOS:
        run_scenario(tracer, client, scenario)

    # Force flush so spans are exported before script exits.
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=10000)

    print("\n" + "=" * 60)
    print("Done. Check Phoenix UI:")
    print("  https://app.phoenix.arize.com/s/golikomikhail")
    print("  Project: phoenix2pytest-dryrun")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
