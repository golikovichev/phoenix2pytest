# ruff: noqa: E501
# E501 disabled file-wide because SYSTEM_PROMPT is a multi-line text payload
# the LLM consumes verbatim. Re-wrapping its lines would change the prompt.
"""Synthesiser agent: turns a classified LLM failure trace into a runnable pytest file.

The synthesiser is the third step in the pipeline (Phoenix span fetch -> failure
extractor -> synthesiser). It accepts the trace data and the structured details
the extractor produced and asks Gemini 2.5 Pro to write a pytest module that
reproduces the prompt and asserts the failure is gone.

The output is plain Python source. Markdown fences around the response are
stripped defensively so the file can be written to disk and run as-is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-2.5-pro"

SYSTEM_PROMPT = """You generate pytest regression tests that catch specific LLM failure modes.

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


@dataclass
class TraceData:
    """Minimal view of a Phoenix span the synthesiser cares about."""

    user_prompt: str
    llm_output: str = ""
    span_id: str = ""


@dataclass
class FailureDetails:
    """Structured output from the extractor step that drives synthesis."""

    failure_mode: str
    evidence: str = ""
    expected_behavior: str = ""
    assertion_strategy: str = ""
    key_strings_to_exclude: list[str] | None = None
    key_patterns_required: list[str] | None = None

    def __post_init__(self) -> None:
        if self.key_strings_to_exclude is None:
            self.key_strings_to_exclude = []
        if self.key_patterns_required is None:
            self.key_patterns_required = []

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FailureDetails:
        return cls(
            failure_mode=str(payload.get("failure_mode") or "unknown"),
            evidence=str(payload.get("evidence") or ""),
            expected_behavior=str(payload.get("expected_behavior") or ""),
            assertion_strategy=str(payload.get("assertion_strategy") or ""),
            key_strings_to_exclude=list(payload.get("key_strings_to_exclude") or []),
            key_patterns_required=list(payload.get("key_patterns_required") or []),
        )


class GeminiClient(Protocol):
    """Minimal protocol so the synthesiser is testable without importing genai."""

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        """Return the model reply for the given system + user message pair."""
        ...


def build_user_message(trace: TraceData, details: FailureDetails) -> str:
    """Assemble the user-side prompt the model sees."""
    return (
        f"USER PROMPT:\n{trace.user_prompt}\n\n"
        f"FAILURE MODE: {details.failure_mode}\n"
        f"EVIDENCE: {details.evidence}\n"
        f"EXPECTED BEHAVIOR: {details.expected_behavior}\n"
        f"ASSERTION STRATEGY: {details.assertion_strategy}\n"
        f"STRINGS TO EXCLUDE: {json.dumps(details.key_strings_to_exclude or [])}\n"
        f"PATTERNS REQUIRED: {json.dumps(details.key_patterns_required or [])}\n\n"
        f"Generate the pytest file. Output only Python code."
    )


def build_user_message_for_group(traces: list[TraceData], details: FailureDetails) -> str:
    """Assemble the prompt for a group of traces that share one failure mode.

    Asks the model to emit a single pytest module that parametrises the test
    function over the user prompts in the group. The assertion strategy and
    failure mode are identical across the group, only the inputs differ.
    """
    prompts_json = json.dumps([t.user_prompt for t in traces], ensure_ascii=False, indent=2)
    return (
        f"USER PROMPTS (multiple inputs that all trigger this failure):\n"
        f"{prompts_json}\n\n"
        f"FAILURE MODE: {details.failure_mode}\n"
        f"EVIDENCE: {details.evidence}\n"
        f"EXPECTED BEHAVIOR: {details.expected_behavior}\n"
        f"ASSERTION STRATEGY: {details.assertion_strategy}\n"
        f"STRINGS TO EXCLUDE: {json.dumps(details.key_strings_to_exclude or [])}\n"
        f"PATTERNS REQUIRED: {json.dumps(details.key_patterns_required or [])}\n\n"
        f"Emit a single pytest module with one @pytest.mark.parametrize "
        f"function that covers ALL prompts above. The test signature should "
        f"accept the prompt as a parameter. Output only Python code."
    )


_FENCE_OPEN = re.compile(r"^```(?:python)?\s*", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$", re.MULTILINE)


def strip_markdown_fences(raw: str) -> str:
    """Remove any leading / trailing markdown fence the model may add.

    Many LLMs ignore the no-fences instruction and wrap the reply in ```python
    ... ``` anyway. Stripping defensively means the synthesiser still produces
    a parseable .py file when the model misbehaves.
    """
    cleaned = _FENCE_OPEN.sub("", raw.strip(), count=1)
    cleaned = _FENCE_CLOSE.sub("", cleaned, count=1)
    return cleaned.strip() + "\n"


def synthesise(
    trace: TraceData,
    details: FailureDetails,
    client: GeminiClient,
    *,
    model: str = DEFAULT_MODEL,
) -> str:
    """Produce the pytest source for a single failure trace.

    The function is pure given the client: same trace + same details + same
    model name reach the model with the same prompt. Determinism after that
    point depends on the underlying Gemini call.
    """
    user_msg = build_user_message(trace, details)
    raw = client.generate_text(model=model, system=SYSTEM_PROMPT, user=user_msg)
    return strip_markdown_fences(raw or "")


_SANITISE = re.compile(r"[^a-z0-9_]")


def _failure_mode_slug(failure_mode: str) -> str:
    """Lowercase + sanitise the failure mode into a filename-safe slug.

    Any character outside ``[a-z0-9_]`` becomes an underscore. Returns
    ``"unknown"`` when the input is empty or sanitises to an empty string.
    """
    return _SANITISE.sub("_", (failure_mode or "unknown").lower()) or "unknown"


def synthesise_many(
    items: list[tuple[TraceData, FailureDetails]],
    client: GeminiClient,
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, str]:
    """Synthesise tests for many trace + details pairs grouped by failure mode.

    Items that share the same failure_mode_slug are folded into a single group
    and the model is asked to emit a parametrised pytest function covering
    all of their user prompts. Items with a unique failure mode go through
    the plain single-trace prompt.

    Returns a mapping ``{failure_mode_slug: pytest_source}`` with one entry
    per distinct failure mode. Insertion order follows first occurrence of
    each failure mode in ``items``.
    """
    if not items:
        return {}

    # Group by slug, preserving insertion order and FailureDetails of the first
    # trace seen for each group. We assume the extractor classified consistently.
    groups: dict[str, tuple[list[TraceData], FailureDetails]] = {}
    for trace, details in items:
        slug = _failure_mode_slug(details.failure_mode)
        if slug in groups:
            groups[slug][0].append(trace)
        else:
            groups[slug] = ([trace], details)

    output: dict[str, str] = {}
    for slug, (traces, details) in groups.items():
        if len(traces) == 1:
            user_msg = build_user_message(traces[0], details)
        else:
            user_msg = build_user_message_for_group(traces, details)
        raw = client.generate_text(model=model, system=SYSTEM_PROMPT, user=user_msg)
        output[slug] = strip_markdown_fences(raw or "")
    return output


def write_test_file(failure_mode: str, code: str, target_dir: Path) -> Path:
    """Write the synthesised code to ``target_dir/test_<failure_mode>.py``.

    The directory is created if it does not yet exist. The failure mode is
    lowercased and any character outside ``[a-z0-9_]`` is replaced with an
    underscore so the filename is import-safe.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _failure_mode_slug(failure_mode)
    target = target_dir / f"test_{safe_name}.py"
    target.write_text(code, encoding="utf-8")
    return target


def write_test_files(codes: dict[str, str], target_dir: Path) -> list[Path]:
    """Write each entry in ``codes`` to ``target_dir/test_<slug>.py``.

    The slug is assumed to be pre-sanitised (typically from ``synthesise_many``
    output). Returns the written paths in iteration order.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for slug, code in codes.items():
        target = target_dir / f"test_{slug}.py"
        target.write_text(code, encoding="utf-8")
        paths.append(target)
    return paths
