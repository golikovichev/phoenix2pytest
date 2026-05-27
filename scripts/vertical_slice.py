"""End-to-end vertical slice: Phoenix trace into pytest regression test.

Pipeline:
    1. Fetch failed LLM trace from Phoenix project `phoenix2pytest-dryrun`
    2. Gemini classifies failure mode (structured JSON)
    3. Gemini synthesises pytest regression test from classified trace
    4. Write generated test to generated_tests/test_<mode>.py
    5. Run pytest on the generated file
    6. Print red/green verdict

Demonstrates the full architecture on a single trace before scaling.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from phoenix.client import Client

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = REPO_ROOT / "generated_tests"
GENERATED_DIR.mkdir(exist_ok=True)


# ---------- Step 1: Phoenix span fetch ----------


def fetch_spans(project: str, limit: int = 10) -> list[dict]:
    """Pull spans from Phoenix project."""
    client = Client(
        base_url=os.environ["PHOENIX_BASE_URL"],
        api_key=os.environ["PHOENIX_API_KEY"],
    )
    return client.spans.get_spans(project_identifier=project, limit=limit)


def extract_trace_data(span: dict) -> dict:
    """Pull the bits of a span we need for classification + synthesis."""
    attrs = span.get("attributes", {}) or {}

    # OpenInference flattens nested keys with dots in some serializations.
    def get(*candidates: str, default: str = "") -> str:
        for key in candidates:
            value = attrs.get(key)
            if value:
                return str(value)
        return default

    return {
        "span_id": span.get("context", {}).get("span_id") or span.get("id"),
        "name": span.get("name"),
        "user_prompt": get("input.value", "input"),
        "llm_output": get("output.value", "output"),
        "model": get("llm.model_name", "llm.model"),
        "failure_mode_label": get("phoenix2pytest.failure_mode"),
        "ideal_behavior": get("phoenix2pytest.ideal_behavior"),
        "tokens_total": attrs.get("llm.token_count.total", 0),
    }


# ---------- Step 2: Gemini classifier ----------

EXTRACTOR_SYSTEM_PROMPT = """You extract evidence from LLM traces that have already been labeled with a failure mode.

You receive a trace (user prompt + LLM output) and a known failure mode label. Trust the label as ground truth - your job is NOT to decide if it's a failure.

Your job: extract concrete details that will drive test generation.

Reply with ONLY a JSON object, no prose, no markdown fences:
{
  "evidence": "the specific phrase or pattern in the output that demonstrates the labeled failure",
  "expected_behavior": "what an aligned model should have done instead, in one sentence",
  "assertion_strategy": "concrete approach - substring_excluded | regex_excluded | format_must_match | answer_must_be_exact | refusal_marker_required",
  "key_strings_to_exclude": ["list of specific phrases the test should assert are NOT in output"],
  "key_patterns_required": ["list of phrases or regex patterns the test should assert ARE in output, if any"]
}

Failure mode definitions (for context only - don't re-classify):
- hallucination: model fabricates specific facts. Tests assert specific made-up strings absent.
- format_break: output violates strict format demand. Tests assert format with regex.
- off_topic_drift: output adds content beyond what was asked. Tests assert length / specific content.
- stale_data: model claims real-time data it cannot have. Tests assert refusal markers present.
- wrong_reasoning: model produces incorrect reasoning. Tests assert correct answer or clarification request.
- refusal_bug: model refuses something it should answer. Tests assert refusal markers absent.
"""


def extract_failure_details(client: genai.Client, trace_data: dict) -> dict:
    failure_mode = trace_data.get("failure_mode_label") or "unknown"
    user_msg = (
        f"FAILURE MODE LABEL: {failure_mode}\n\n"
        f"USER PROMPT:\n{trace_data['user_prompt']}\n\n"
        f"LLM OUTPUT:\n{trace_data['llm_output']}\n\n"
        f"Extract evidence and assertion strategy."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_msg,
        config=GenerateContentConfig(
            system_instruction=EXTRACTOR_SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )
    result = json.loads(response.text or "{}")
    result["failure_mode"] = failure_mode
    return result


# ---------- Step 3: Gemini test synthesiser ----------

SYNTHESISER_SYSTEM_PROMPT = """You generate pytest regression tests that catch specific LLM failure modes.

You receive: the user prompt that triggered the failure, evidence from the bad output, the failure mode, and concrete assertion strategy with strings to check for / against.

Generate a runnable pytest file that:
1. Calls Gemini 2.5 Flash via google-genai with the user prompt
2. Implements the assertion strategy precisely (substring_excluded / regex_excluded / format_must_match / answer_must_be_exact / refusal_marker_required)
3. Uses concrete string-level assertions (no LLM-as-judge)
4. Includes proper imports and a clear test function name following pattern `test_no_<failure_mode>_<short_context>`

Output ONLY runnable Python code. No prose. No markdown fences. No explanation.

Required template:

import os
import re
import pytest

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "phoenix2pytest-hackathon")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

from google import genai
from google.genai.types import HttpOptions


def _ask_gemini(prompt: str) -> str:
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text or ""


def test_no_<failure_mode>_<short_context>():
    response = _ask_gemini(\"\"\"<user prompt verbatim>\"\"\")
    # Concrete assertions implementing the assertion_strategy
    ...
"""


def synthesise_test(client: genai.Client, trace_data: dict, details: dict) -> str:
    user_msg = (
        f"USER PROMPT:\n{trace_data['user_prompt']}\n\n"
        f"FAILURE MODE: {details.get('failure_mode')}\n"
        f"EVIDENCE: {details.get('evidence')}\n"
        f"EXPECTED BEHAVIOR: {details.get('expected_behavior')}\n"
        f"ASSERTION STRATEGY: {details.get('assertion_strategy')}\n"
        f"STRINGS TO EXCLUDE: {json.dumps(details.get('key_strings_to_exclude', []))}\n"
        f"PATTERNS REQUIRED: {json.dumps(details.get('key_patterns_required', []))}\n\n"
        f"Generate the pytest file. Output only Python code."
    )

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=user_msg,
        config=GenerateContentConfig(system_instruction=SYNTHESISER_SYSTEM_PROMPT),
    )
    code = response.text or ""

    # Defensive: strip any accidental markdown fences
    code = re.sub(r"^```(?:python)?\s*", "", code.strip())
    code = re.sub(r"\s*```$", "", code)
    return code


# ---------- Step 4: write + run ----------


def write_test_file(failure_mode: str, code: str) -> Path:
    sanitized = re.sub(r"[^a-z0-9_]", "_", failure_mode.lower())
    target = GENERATED_DIR / f"test_{sanitized}.py"
    target.write_text(code, encoding="utf-8")
    return target


def run_pytest(target: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(target), "-v", "--tb=short"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


# ---------- Driver ----------


def main() -> int:
    load_dotenv()
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    print("=" * 70)
    print("Phoenix2Pytest: vertical slice end-to-end")
    print("=" * 70)

    spans = fetch_spans("phoenix2pytest-dryrun", limit=10)
    print(f"\nFetched {len(spans)} spans from Phoenix")

    # Pick the hallucination span as the highest-signal failure.
    target_span = next(
        (
            s
            for s in spans
            if (s.get("attributes") or {}).get("phoenix2pytest.failure_mode") == "hallucination"
        ),
        spans[0] if spans else None,
    )
    if not target_span:
        print("ERROR: no spans found", file=sys.stderr)
        return 1

    trace_data = extract_trace_data(target_span)
    print(f"\nTarget span: {trace_data['name']}")
    print(f"  prompt:  {trace_data['user_prompt'][:100]}...")
    print(f"  output:  {trace_data['llm_output'][:100]}...")
    print(f"  labeled: {trace_data['failure_mode_label']}")

    gemini = genai.Client(http_options=HttpOptions(api_version="v1"))

    print("\n--- Step 2: extract failure details ---")
    details = extract_failure_details(gemini, trace_data)
    print(json.dumps(details, indent=2))

    print("\n--- Step 3: synthesise pytest ---")
    code = synthesise_test(gemini, trace_data, details)
    print(code)

    print("\n--- Step 4: write + run ---")
    target = write_test_file(details.get("failure_mode", "unknown"), code)
    print(f"Wrote {target}")

    return_code, output = run_pytest(target)
    print("\n--- pytest output ---")
    print(output)
    print(f"\nExit code: {return_code}")
    print(
        "\nVerdict: RED (test caught regression - this is the value!)"
        if return_code != 0
        else "\nVerdict: GREEN (model passed - would mean LLM no longer fails)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
