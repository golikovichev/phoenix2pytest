# Changelog

All notable changes to this project are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), but this is an early alpha so versions track the hackathon delivery cycle rather than strict semver.

## [Unreleased]

### Added
- Promoted demo scripts to top-level entry points and updated `pyproject` metadata so installs expose `phoenix-extract` and related commands.
- Cursor support bot fabrication case added to the real-harvested dataset (one more failure pattern documented end-to-end).

### Changed
- Documented the Content-Length parse fallback inside the body-size middleware.
- Revised `SECURITY.md` with supported versions and the reporting flow.

### Maintenance
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
