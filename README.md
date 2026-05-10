# phoenix2pytest

Turn production LLM failures into regression tests, automatically.

> **Status:** Alpha. In active development.

## What it does

Reads Arize Phoenix traces from your LLM application. Identifies failed conversations, classifies the failure mode, and generates pytest test cases that would have caught the failure. Production traffic feeds the regression suite without manual translation.

## How is this different from DeepEval, Opik, pytest-evals?

Existing eval frameworks assume you know what to test. You write evals up front, run them against your LLM, get scores.

`phoenix2pytest` runs the other direction. It reads traces from production, finds failures, and emits the tests for you.

| Tool | Direction |
|---|---|
| DeepEval / Opik / pytest-evals | spec → eval → run |
| **phoenix2pytest** | **trace → failure → synthesised test** |

Different mental model: closer to how traditional QA turns crash reports into repro tests.

## Quickstart

Coming soon.

## License

MIT.
