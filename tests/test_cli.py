"""Tests for the console CLI.

Every path runs offline: the Gemini client is a stub injected through the
GeminiClient protocol, and the live Phoenix fetch is replaced with an injected
callable that returns canned spans. No network, no subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path

from phoenix2pytest import cli
from phoenix2pytest.synthesiser import FailureDetails, TraceData


class _StubGemini:
    """Returns a minimal valid pytest module; records the last user prompt."""

    def __init__(self) -> None:
        self.user: str | None = None

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        self.user = user
        return "def test_generated():\n    assert True\n"


def _pairs() -> list[tuple[TraceData, FailureDetails]]:
    return [(TraceData(user_prompt="hi"), FailureDetails(failure_mode="hallucination"))]


def _write_traces_file(tmp_path: Path) -> Path:
    path = tmp_path / "traces.json"
    path.write_text(
        json.dumps(
            [{"trace": {"user_prompt": "hi"}, "details": {"failure_mode": "hallucination"}}]
        ),
        encoding="utf-8",
    )
    return path


# --- parser --------------------------------------------------------------


def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args([])
    assert args.out == "tests"
    assert args.dry_run is False
    assert args.paraphrase is False
    assert args.from_file is None


# --- run -----------------------------------------------------------------


def test_run_dry_run_prints_code_and_writes_nothing(tmp_path, capsys) -> None:
    out_dir = tmp_path / "tests"
    code = cli.run(_pairs(), _StubGemini(), out_dir=out_dir, dry_run=True, paraphrase=False)
    captured = capsys.readouterr()
    assert code == 0
    assert "def test_generated" in captured.out
    assert not out_dir.exists()


def test_run_writes_files(tmp_path) -> None:
    out_dir = tmp_path / "tests"
    code = cli.run(_pairs(), _StubGemini(), out_dir=out_dir, dry_run=False, paraphrase=False)
    assert code == 0
    written = list(out_dir.glob("test_*.py"))
    assert len(written) == 1
    assert "def test_generated" in written[0].read_text(encoding="utf-8")


def test_run_empty_pairs_returns_2(tmp_path, capsys) -> None:
    code = cli.run([], _StubGemini(), out_dir=tmp_path, dry_run=False, paraphrase=False)
    assert code == 2


def test_run_paraphrase_threads_into_prompt(tmp_path) -> None:
    client = _StubGemini()
    cli.run(_pairs(), client, out_dir=tmp_path / "t", dry_run=True, paraphrase=True)
    assert client.user is not None
    assert "assert_paraphrase_similar" in client.user


# --- spans_to_pairs ------------------------------------------------------


def test_spans_to_pairs_skips_unlabelled_and_extracts_labelled() -> None:
    labelled = {
        "context": {"span_id": "s1"},
        "attributes": {
            "input.value": "q",
            "output.value": "a",
            "phoenix2pytest.failure_mode": "stale_data",
        },
    }
    unlabelled = {"context": {"span_id": "s2"}, "attributes": {"input.value": "q2"}}

    class _ExtractGemini:
        def generate_text(self, *, model: str, system: str, user: str) -> str:
            return json.dumps({"evidence": "e", "assertion_strategy": "refusal_marker_required"})

    pairs = cli.spans_to_pairs([labelled, unlabelled], _ExtractGemini(), label=None)
    assert len(pairs) == 1
    trace, details = pairs[0]
    assert trace.user_prompt == "q"
    assert details.failure_mode == "stale_data"


def test_spans_to_pairs_label_filter_matches_one_mode() -> None:
    def span(mode: str, sid: str) -> dict:
        return {
            "context": {"span_id": sid},
            "attributes": {"input.value": "q", "phoenix2pytest.failure_mode": mode},
        }

    class _ExtractGemini:
        def generate_text(self, *, model: str, system: str, user: str) -> str:
            return json.dumps({"evidence": "e"})

    pairs = cli.spans_to_pairs(
        [span("hallucination", "a"), span("stale_data", "b")],
        _ExtractGemini(),
        label="stale_data",
    )
    assert len(pairs) == 1
    assert pairs[0][1].failure_mode == "stale_data"


# --- main ----------------------------------------------------------------


def test_main_from_file_dry_run(tmp_path, capsys) -> None:
    path = _write_traces_file(tmp_path)
    code = cli.main(["--from-file", str(path), "--dry-run"], gemini=_StubGemini())
    captured = capsys.readouterr()
    assert code == 0
    assert "def test_generated" in captured.out


def test_main_from_file_writes(tmp_path) -> None:
    path = _write_traces_file(tmp_path)
    out_dir = tmp_path / "out"
    code = cli.main(["--from-file", str(path), "--out", str(out_dir)], gemini=_StubGemini())
    assert code == 0
    assert list(out_dir.glob("test_*.py"))


def test_main_missing_file_returns_1(tmp_path, capsys) -> None:
    code = cli.main(["--from-file", str(tmp_path / "nope.json")], gemini=_StubGemini())
    assert code == 1
    captured = capsys.readouterr()
    assert "nope.json" in (captured.err + captured.out)


def test_main_live_label_uses_injected_fetch(tmp_path, capsys) -> None:
    spans = [
        {
            "context": {"span_id": "s1"},
            "attributes": {
                "input.value": "q",
                "phoenix2pytest.failure_mode": "hallucination",
            },
        }
    ]

    class _ExtractAndSynth:
        def generate_text(self, *, model: str, system: str, user: str) -> str:
            # extractor asks for JSON; synthesiser asks for code. Disambiguate
            # by the system prompt content.
            if "JSON object" in system:
                return json.dumps({"evidence": "e"})
            return "def test_generated():\n    assert True\n"

    code = cli.main(
        ["--label", "hallucination", "--dry-run"],
        gemini=_ExtractAndSynth(),
        fetch_spans=lambda label: spans,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "def test_generated" in captured.out


def test_main_live_missing_env_returns_1(monkeypatch, capsys) -> None:
    # No --from-file and no injected fetch: the real live fetch runs and must
    # fail cleanly (exit 1 + message) when Phoenix env vars are unset, not
    # crash with a bare KeyError traceback.
    monkeypatch.delenv("PHOENIX_HOST", raising=False)
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
    code = cli.main(["--label", "hallucination"], gemini=_StubGemini())
    captured = capsys.readouterr()
    assert code == 1
    assert "PHOENIX_HOST" in (captured.err + captured.out)


def test_main_live_extraction_error_returns_1(capsys) -> None:
    spans = [
        {
            "context": {"span_id": "s1"},
            "attributes": {"input.value": "q", "phoenix2pytest.failure_mode": "m"},
        }
    ]

    class _BadExtractor:
        def generate_text(self, *, model: str, system: str, user: str) -> str:
            return "sorry, not JSON"  # extractor will reject this

    code = cli.main(
        ["--label", "m", "--dry-run"],
        gemini=_BadExtractor(),
        fetch_spans=lambda label: spans,
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "error:" in (captured.err + captured.out)


def test_run_paraphrase_writes_embedder_conftest(tmp_path) -> None:
    out_dir = tmp_path / "tests"
    cli.run(_pairs(), _StubGemini(), out_dir=out_dir, dry_run=False, paraphrase=True)
    conftest = out_dir / "conftest.py"
    assert conftest.exists()
    body = conftest.read_text(encoding="utf-8")
    assert "def embedder" in body
    assert "build_default_embedder" in body


def test_run_paraphrase_does_not_overwrite_existing_conftest(tmp_path) -> None:
    out_dir = tmp_path / "tests"
    out_dir.mkdir()
    conftest = out_dir / "conftest.py"
    conftest.write_text("# my own conftest\n", encoding="utf-8")
    cli.run(_pairs(), _StubGemini(), out_dir=out_dir, dry_run=False, paraphrase=True)
    assert conftest.read_text(encoding="utf-8") == "# my own conftest\n"


def test_run_without_paraphrase_writes_no_conftest(tmp_path) -> None:
    out_dir = tmp_path / "tests"
    cli.run(_pairs(), _StubGemini(), out_dir=out_dir, dry_run=False, paraphrase=False)
    assert not (out_dir / "conftest.py").exists()
