---
name: phoenix2pytest
description: Turn labeled LLM failure traces from an Arize Phoenix project into runnable pytest regression tests using the phoenix2pytest pipeline. Use when the user has an LLM application emitting OpenInference spans to Phoenix and wants a regression suite from real production failures, when extracting test cases from observed LLM bugs (hallucination, format break, off-topic drift, stale data, wrong reasoning, refusal bug), when bridging Phoenix-labeled traces into pytest-based suites for CI, when the user mentions Arize Phoenix MCP, OpenInference instrumentation, LLM observability, Gemini test synthesis, Vertex AI agent evaluation, or wants to react to LLM failures rather than predict them upfront.
license: MIT
metadata:
  category: "ai-testing"
  homepage: "https://github.com/golikovichev/phoenix2pytest"
  pypi: "https://github.com/golikovichev/phoenix2pytest"
  version: "0.1.0"
---

# phoenix2pytest

Read labeled-as-failure traces from an Arize Phoenix project, write pytest regression tests that catch them. The Phoenix trace stays the source of truth; the generated suite is committable code. Re-run when new failures land in your Phoenix project.

Full pipeline architecture, schema reference, ingestion script, web UI quickstart, comparison vs other eval frameworks, and Cloud Run deployment notes live in `REFERENCE.md` next to this file.

## Quick start

1. Install Phoenix client + Vertex AI Gemini deps (this project lists pinned versions):
   ```bash
   pip install -e .
   ```

2. Set Phoenix + Vertex AI credentials in `.env`:
   ```bash
   PHOENIX_BASE_URL=https://app.phoenix.arize.com
   PHOENIX_API_KEY=<your-phoenix-api-key>
   GOOGLE_CLOUD_PROJECT=<your-gcp-project>
   GOOGLE_CLOUD_LOCATION=us-central1
   GOOGLE_GENAI_USE_VERTEXAI=True
   ```

3. Ingest the demo dataset (51 traces across 6 failure modes) to populate your Phoenix project:
   ```bash
   python scripts/ingest_demo_dataset.py --project phoenix2pytest-demo
   ```
   This emits OpenInference spans for each trace.

4. Sanity-check the dataset shape before generating tests:
   ```bash
   python -m pytest tests/test_demo_dataset.py -v
   ```

5. Launch the web UI to browse failures and generate tests interactively:
   ```bash
   uvicorn phoenix2pytest.web:app --reload --port 8000
   ```
   Open http://127.0.0.1:8000, paste a Phoenix span ID or pick from the demo dataset, click Generate.

6. Validate the generated test before committing: run it once to confirm it **fails** against the original bad response (reproducing the regression) and **passes** after applying a fix. This confirms the assertion targets the right behavior.

7. The generated pytest module calls Gemini with the original prompt and asserts the specific broken behaviors are NOT in the response. Commit it. Re-run pytest in CI on every prompt edit or model upgrade.

## Example

Real `halluc_001` span captured in Phoenix shows Gemini fabricating page 47 quotes from The Great Gatsby:

```text
INPUT:   Tell me three specific lines of dialogue from page 47 of
         'The Great Gatsby' by F. Scott Fitzgerald. Quote them exactly.
OUTPUT:  1. "I was in the Third Division during the war," I told him...
         2. "I was in the Ninth Machine-Gun Battalion," he said approvingly...
         3. "Why, I thought you knew, old sport. I'm Gatsby."
LABEL:   phoenix2pytest.failure_mode = hallucination
```

The synthesised pytest assertion targets the exact fabricated strings:

```python
def test_no_hallucination_gatsby_page_47():
    response = _ask_gemini("Tell me three specific lines of dialogue from page 47...")
    assert "I was in the Third Division during the war," not in response
    assert "I was in the Ninth Machine-Gun Battalion," not in response
    assert "I thought you knew, old sport. I'm Gatsby." not in response
```

The test stays in CI. Next time someone edits the system prompt or the model gets re-quantised, this exact regression test catches it.

## Common errors

- **`PHOENIX_API_KEY not set`:** check `.env` is loaded by the script (uses `python-dotenv`).
- **Empty Phoenix project:** run step 3 first to populate the demo dataset.
- **Gemini quota exceeded:** 15 real-Gemini calls in the demo cost roughly $0.01; check your Vertex AI quota.
- **Web UI port conflict:** pass `--port 8001` to uvicorn.

Full error-handling tree, schema validation rules, and ingestion-script flags in `REFERENCE.md`.

## References

- Bundle: `REFERENCE.md` (pipeline architecture, schema reference, ingestion flags, web UI internals, Cloud Run deploy)
- Project: https://github.com/golikovichev/phoenix2pytest
- Release v0.1.0: https://github.com/golikovichev/phoenix2pytest/releases/tag/v0.1.0
