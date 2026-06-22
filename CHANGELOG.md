# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow semantic versioning.

## [1.0.0] - 2026-06

First stable release. The pipeline (Phoenix trace -> Gemini extractor -> synthesiser -> pytest file) is stable within the documented scope; see the Limits section in `README.md`.

### Added
- Batch trace-to-test generation: synthesise tests for many annotated traces in one request, grouping by failure mode and folding shared modes into one parametrised test (`/batch` on the web UI and Cloud Run).
- `/health` probe for the web service.
- Generated code is validated as parseable Python (`ast.parse`) before it is returned; a model reply that is not valid Python raises `SynthesisError` and the web endpoints return HTTP 502 instead of writing a broken test file.
- Promoted demo scripts to top-level entry points and updated `pyproject` metadata.
- Cursor support bot fabrication case added to the real-harvested dataset.

### Changed
- Redesigned the demo page with a card layout, a shared Single/Batch nav bar, and a pre-filled runnable example.
- Documented the Content-Length parse fallback inside the body-size middleware.
- Revised `SECURITY.md` with supported versions and the reporting flow.

### Maintenance
- Pinned GitHub Actions to commit SHAs; added OpenSSF Scorecard analysis and least-privilege workflow permissions.
- Dependabot rolled actions/checkout from 4 to 6, codecov/codecov-action from 4 to 6, and github/codeql-action from 3 to 4.

## [0.1.0] - 2026-05

Initial alpha release for the Google Cloud Rapid Agent Hackathon (Arize track).

### Added
- Phoenix LLM trace ingest: read an Arize Phoenix trace export and walk the spans to surface failure patterns.
- Regression test extraction: emit a runnable pytest module that replays the failing input and asserts the corrected behaviour.
- Optional Gemini-assisted assertion generation behind a `--use-gemini` flag (off by default; works without any LLM call).
- FastAPI demo surface with a body-size middleware for the hosted walkthrough.
- pytest + ruff + pre-commit setup with a small anti-marker scanner for prose hygiene.
- CI matrix for Python 3.11, 3.12, and 3.13. CodeQL workflow and codecov upload included from the start.

### Known limitations
- Alpha. CLI flags are subject to change.
- Trace ingest currently assumes the Phoenix v3 export shape.
- The Gemini assertion path is best-effort and not yet covered by integration tests.

See `README.md` for the hackathon framing and stack notes.
