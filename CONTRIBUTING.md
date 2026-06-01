# Contributing

Thanks for your interest in phoenix2pytest. This is a small alpha project shipped during the Google Cloud Rapid Agent Hackathon, so the contribution flow is light.

## Reporting a bug

Open an issue with:

- What you ran (command + Python version)
- What you expected
- What happened instead
- A minimal Arize Phoenix trace export if the bug is parser-related (strip any project keys first)

## Suggesting a feature

Open an issue first so we can talk through the use case before you write code. The project scope is intentionally narrow (Phoenix LLM trace ingest, regression test extraction, optional Gemini-assisted assertions), so feature requests that pull it elsewhere will get a polite redirect.

## Submitting a pull request

1. Fork the repo and create a branch from `main`.
2. Make your changes. Keep the diff focused on one thing.
3. Add or update tests in `tests/`. The CI runs `pytest -v` on Python 3.11, 3.12, and 3.13.
4. Run the tests locally before pushing:
   ```bash
   pip install -e ".[dev]"
   pytest -v
   ```
5. Open the PR with a short description of what changed and why.

## Code style

- Python 3.11+. Type hints on public functions.
- Function and variable names in English, snake_case (e.g., `extract_trace_spans`).
- One responsibility per function. If a function grows past 30-40 lines, split it.
- The pre-commit config enforces ruff lint and a small anti-marker scanner. Run `pre-commit run --all-files` before opening a PR.

## Security

If you find something that could leak API keys or production trace data, please email me directly instead of opening a public issue. Address is on my GitHub profile. See also `SECURITY.md`.
