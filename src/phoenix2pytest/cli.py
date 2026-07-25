"""Console entry point: turn flagged Phoenix traces into pytest files.

Two sources of traces:

* ``--from-file PATH`` reads a saved JSON traces file (the same shape the batch
  web endpoint accepts). Fully offline, no Phoenix or extraction step needed.
* ``--label MODE`` fetches spans from a live Phoenix project, keeps the ones a
  human annotated with a failure mode, and extracts the details with Gemini.

Both sources feed the same synthesiser and writer. The orchestration functions
take injected clients so the whole flow is testable without a network.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from phoenix2pytest import __version__
from phoenix2pytest.extractor import ExtractionError, extract_failure_details, extract_trace_data
from phoenix2pytest.loader import TraceLoadError, load_traces
from phoenix2pytest.synthesiser import (
    FailureDetails,
    GeminiClient,
    SynthesisError,
    TraceData,
    build_default_client,
    synthesise_many,
    write_test_files,
)

Pair = tuple[TraceData, FailureDetails]
SpanFetcher = Callable[[str | None], list[dict[str, Any]]]


class ConfigError(RuntimeError):
    """The live Phoenix path is missing required configuration or a dependency."""


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``phoenix2pytest`` command."""
    parser = argparse.ArgumentParser(
        prog="phoenix2pytest",
        description="Turn flagged Phoenix LLM-failure traces into pytest regression tests.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Only use traces annotated with this failure mode. Omit to use every annotated trace.",
    )
    parser.add_argument(
        "--from-file",
        dest="from_file",
        default=None,
        help="Read traces from a JSON file instead of a live Phoenix project.",
    )
    parser.add_argument(
        "--out",
        default="tests",
        help="Directory to write generated test files into (default: tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated tests to stdout instead of writing files.",
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="Generate paraphrase-tolerant tests that assert embedding similarity.",
    )
    parser.add_argument("--version", action="version", version=f"phoenix2pytest {__version__}")
    return parser


def spans_to_pairs(
    spans: list[dict[str, Any]],
    client: GeminiClient,
    *,
    label: str | None = None,
) -> list[Pair]:
    """Turn raw Phoenix spans into ``(TraceData, FailureDetails)`` pairs.

    Spans with no failure-mode annotation are skipped, since only labelled
    traces drive test generation. When ``label`` is given, only spans annotated
    with that exact failure mode are kept. Each surviving span's details are
    extracted with ``client``.
    """
    pairs: list[Pair] = []
    for span in spans:
        data = extract_trace_data(span)
        mode = data.get("failure_mode_label") or ""
        if not mode:
            continue
        if label is not None and mode != label:
            continue
        details = extract_failure_details(data, client)
        trace = TraceData(
            user_prompt=data.get("user_prompt", ""),
            llm_output=data.get("llm_output", ""),
            span_id=data.get("span_id", ""),
        )
        pairs.append((trace, details))
    return pairs


_EMBEDDER_CONFTEST = '''import pytest

from phoenix2pytest.assertions import build_default_embedder


@pytest.fixture
def embedder():
    """Embedder for paraphrase-tolerant tests.

    Returns the default google-genai embedder. Override it in your own conftest,
    or inject a stub in a test, to run offline and deterministically.
    """
    return build_default_embedder()
'''


def _ensure_embedder_conftest(out_dir: Path) -> None:
    """Write an ``embedder`` fixture next to paraphrase tests, without clobbering.

    Paraphrase tests take an ``embedder`` fixture; without one pytest fails at
    collection. This drops a ready conftest so the generated tests are runnable
    out of the box, but never overwrites a conftest the user already has.
    """
    conftest = out_dir / "conftest.py"
    if conftest.exists():
        print(
            f"note: {conftest} already exists; add an `embedder` fixture "
            "(see phoenix2pytest.assertions.build_default_embedder)",
            file=sys.stderr,
        )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    conftest.write_text(_EMBEDDER_CONFTEST, encoding="utf-8")
    print(f"wrote {conftest} (embedder fixture)")


def run(
    pairs: list[Pair],
    client: GeminiClient,
    *,
    out_dir: Path,
    dry_run: bool,
    paraphrase: bool,
) -> int:
    """Synthesise tests for ``pairs`` and either print or write them.

    Returns a process exit code: ``0`` on success, ``2`` when there were no
    traces to work with (so a CI job can tell "nothing flagged" from "ran and
    produced tests").
    """
    if not pairs:
        print("No flagged traces to synthesise.", file=sys.stderr)
        return 2
    codes = synthesise_many(pairs, client, paraphrase=paraphrase)
    if dry_run:
        for slug, code in codes.items():
            print(f"# test_{slug}.py")
            print(code)
        return 0
    written = write_test_files(codes, out_dir)
    for path in written:
        print(f"wrote {path}")
    if paraphrase:
        _ensure_embedder_conftest(out_dir)
    return 0


def _fetch_spans_from_phoenix(label: str | None) -> list[dict[str, Any]]:
    """Fetch spans from the live Phoenix project named by PHOENIX_PROJECT.

    Uses the arize-phoenix-client already in the dependency set. This boundary
    talks to a live server, so it is exercised by manual / e2e runs rather than
    the unit suite; the pure mapping it feeds (``spans_to_pairs``) is unit
    tested with injected spans.
    """
    import os

    missing = [name for name in ("PHOENIX_HOST", "PHOENIX_API_KEY") if not os.environ.get(name)]
    if missing:
        raise ConfigError(
            f"the live Phoenix path needs {' and '.join(missing)} set in the environment; "
            "set them, or pass --from-file to work offline"
        )
    try:
        from phoenix.client import Client  # local import: only needed for the live path
    except ImportError as exc:  # pragma: no cover - depends on optional install state
        raise ConfigError(
            "arize-phoenix-client is not importable; install phoenix2pytest with its "
            "dependencies, or pass --from-file to work offline"
        ) from exc

    client = Client(
        base_url=os.environ["PHOENIX_HOST"],
        api_key=os.environ["PHOENIX_API_KEY"],
    )
    project = os.environ.get("PHOENIX_PROJECT", "default")
    return list(client.spans.get_spans(project_identifier=project, limit=100))


def main(
    argv: list[str] | None = None,
    *,
    gemini: GeminiClient | None = None,
    fetch_spans: SpanFetcher | None = None,
) -> int:
    """Parse arguments and run the pipeline. Returns a process exit code.

    ``gemini`` and ``fetch_spans`` are injection seams for tests; in normal use
    the Gemini client is built from the environment and traces come from a live
    Phoenix project unless ``--from-file`` is given.
    """
    args = build_parser().parse_args(argv)
    client = gemini if gemini is not None else build_default_client()

    try:
        if args.from_file:
            pairs = load_traces(args.from_file)
        else:
            fetcher = fetch_spans if fetch_spans is not None else _fetch_spans_from_phoenix
            spans = fetcher(args.label)
            pairs = spans_to_pairs(spans, client, label=args.label)
        return run(
            pairs,
            client,
            out_dir=Path(args.out),
            dry_run=args.dry_run,
            paraphrase=args.paraphrase,
        )
    except (TraceLoadError, SynthesisError, ExtractionError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
