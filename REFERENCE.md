# phoenix2pytest reference

Detailed pipeline architecture, pydantic schema, demo dataset structure, ingestion script flags, web UI internals, Cloud Run deployment, and roadmap. SKILL.md keeps the quick-start and one Gatsby example. Read this file when you need the full surface.

## Pipeline architecture

```
Your LLM app (LangChain, LlamaIndex, custom)
    │ OpenInference OTEL spans
    ▼
Arize Phoenix project (cloud or self-hosted)
    │ Engineer reviews + labels
    │   .phoenix2pytest.failure_mode = hallucination | format_break | ...
    ▼
phoenix2pytest pipeline
    ├── 1. Fetch labeled trace via Phoenix MCP client
    ├── 2. Gemini Flash extractor: identify what specifically broke
    │      (evidence strings, expected behavior, assertion strategy)
    ├── 3. Gemini Pro synthesiser: write runnable pytest file
    └── 4. Write to generated_tests/test_<failure_mode>.py
    │
    ▼
Commit generated test, run in CI
```

A console-script CLI that runs steps 1-4 end-to-end is planned for a later release (see the README roadmap). In 1.0, the web UI at `phoenix2pytest.web:app` provides the same flow interactively.

## Demo dataset (`tests/data/demo_dataset.json`)

51 traces total, distributed across 6 failure modes per the canonical Literal type in `phoenix2pytest.schema.FailureMode`:

| Failure mode | Count | Description |
| --- | --- | --- |
| hallucination | 13 | Model fabricates specific facts (e.g. Gatsby page 47 quotes) |
| format_break | 10 | Output violates strict format demand (JSON shape, line count) |
| wrong_reasoning | 8 | Incorrect arithmetic, false logical deduction |
| refusal_bug | 8 | Refuses something it should answer |
| off_topic_drift | 6 | Adds content beyond what was asked |
| stale_real_time_data | 6 | Claims real-time data it cannot have |

Source distribution: 15 `real` (elicited from live Gemini calls during ingestion), 35 `synthetic` (hand-curated), 1 `real-harvested` (Reddit thread via Bright Data API).

6 traces are tagged `demo_featured: True`, one per failure mode, for use in demos.

## Pydantic schema

`phoenix2pytest.schema` exposes two pydantic v2 models. Both lock the wire shapes the pipeline passes between agents:

### `TraceScenario`

The trace data extracted from a Phoenix span, consumed by the failure-mode extractor.

```python
class TraceScenario(BaseModel):
    user_prompt: str              # non-empty, stripped
    llm_output: str               # non-empty, stripped
    failure_mode: FailureMode     # closed Literal
    ideal_behavior: str | None    # optional metadata
    model: str | None             # optional, e.g. "gemini-2.5-flash"
    span_id: str | None           # Phoenix span ID
    dataset_id: str | None        # demo dataset row ID
    tokens_total: int = 0         # ge=0
```

`extra="forbid"`. Blank strings on required fields are rejected; blank strings on optionals normalise to None.

### `ExtractorResponse`

The structured JSON returned by the Gemini Flash extractor, consumed by the test synthesiser.

```python
class ExtractorResponse(BaseModel):
    failure_mode: FailureMode
    evidence: str                                # non-empty
    expected_behavior: str                       # non-empty
    assertion_strategy: AssertionStrategy        # closed Literal
    key_strings_to_exclude: list[str] = []       # blank items filtered
    key_patterns_required: list[str] = []        # blank items filtered
```

`extra="forbid"`. Blank list items are filtered automatically.

### Closed vocabularies

```python
FailureMode = Literal[
    "hallucination", "format_break", "off_topic_drift",
    "stale_real_time_data", "wrong_reasoning", "refusal_bug",
]

AssertionStrategy = Literal[
    "substring_excluded", "regex_excluded",
    "format_must_match", "answer_must_be_exact",
    "refusal_marker_required",
]
```

`FAILURE_MODE_VALUES: tuple[str, ...] = get_args(FailureMode)` is the single source of truth for downstream consumers. `scripts.ingest_demo_dataset.VALID_FAILURE_MODES` derives from it.

## Ingestion script (`scripts/ingest_demo_dataset.py`)

| Flag | Default | Purpose |
| --- | --- | --- |
| `--project NAME` | `phoenix2pytest-demo` | Phoenix project name to emit spans into. |
| `--limit N` | none | Ingest first N entries only. Useful for quick smoke tests. |
| `--skip-real` | false | Skip `source: real` entries (avoids live Gemini calls, saves cost). |
| `--dry-run` | false | Validate the dataset without calling Phoenix or Gemini. |
| `--dataset PATH` | `tests/data/demo_dataset.json` | Override dataset path. |

Live Gemini calls for the 15 real entries cost roughly $0.01 total on gemini-2.5-flash. Synthetic and real-harvested entries cost nothing (the failed output is stored verbatim in the JSON).

## Web UI (`phoenix2pytest.web`)

FastAPI app exposing two routes:

- `GET /` - form page. Paste a Phoenix span ID + select failure mode, OR pick from the demo dataset.
- `POST /generate` - accepts JSON or form-encoded body, returns generated pytest test code as HTML preview OR raw JSON (driven by `Accept` header).

Run locally:

```bash
uvicorn phoenix2pytest.web:app --reload --port 8000
```

Includes input-validation guards (oversized body, missing fields, escaped HTML output), see `tests/test_web.py` for the full contract.

## Cloud Run deployment

The repo ships `Dockerfile` + `cloudbuild.yaml` for Cloud Run. Deploy from project root:

```bash
gcloud builds submit --config cloudbuild.yaml
gcloud run deploy phoenix2pytest --image gcr.io/<project>/phoenix2pytest \
    --region us-central1 --allow-unauthenticated \
    --set-env-vars PHOENIX_BASE_URL=...,GOOGLE_CLOUD_PROJECT=...
```

Phoenix API key + Gemini credentials live in Secret Manager mounts, not env vars. Region pinned to us-central1 to match Vertex AI Gemini availability.

Hosted demo URL ships in 0.2.1 (target 2026-06-25).

## Error handling

- **Phoenix MCP unreachable:** check `PHOENIX_BASE_URL` and `PHOENIX_API_KEY`. The MCP server requires the npx-installable Phoenix MCP package; install via `npx @arizeai/phoenix-mcp`.
- **Gemini quota exceeded:** ingestion makes 15 live Gemini calls. Bump quota in GCP console or use `--skip-real` for smoke tests.
- **Schema validation error on ingest:** `tests/test_demo_dataset.py::test_dataset_passes_validate_dataset` flags any malformed entry. Run it before committing dataset changes.
- **Web UI 422 on `/generate`:** check the JSON body has `user_prompt` and `failure_mode`. Field names match the pydantic schema.

## Roadmap

- **0.2.0** (target 2026-06-10): Wire vertical-slice pipeline into the package. CLI entry point `phoenix2pytest --project X --label hallucination`.
- **0.2.1** (target 2026-06-25): Cloud Run hosted demo URL.
- **0.2.x** (target 2026-06-30+): Tutorial cross-posts (Phoenix trace → pytest in 5 minutes; LangChain integration; LlamaIndex integration).
- **0.3.0** (target 2026-07-20, post-15.07 GT submission): More failure mode handlers (`tool_use_error`, `multi_turn_memory_leak`, `temperature_instability`).

## External links

- Project README and architecture write-up: https://github.com/golikovichev/phoenix2pytest
- Release v0.1.0: https://github.com/golikovichev/phoenix2pytest/releases/tag/v0.1.0
- Built for the Arize track of the Google Cloud Rapid Agent Hackathon: https://rapid-agent.devpost.com/
- Arize Phoenix documentation: https://docs.arize.com/phoenix
- OpenInference instrumentation spec: https://github.com/Arize-ai/openinference
