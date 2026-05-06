"""Smoke test: confirm Vertex AI + Gemini work.

Prerequisites:
    1. gcloud auth application-default login
    2. .env file in repo root with GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION,
       GOOGLE_GENAI_USE_VERTEXAI=True (or set as env vars).

Run:
    python scripts/smoke_test_gemini.py
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions


def main() -> int:
    load_dotenv()

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "phoenix2pytest-hackathon")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "True")

    print(f"Project:  {project}")
    print(f"Location: {location}")
    print(f"Vertex:   {use_vertex}")

    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = use_vertex

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with exactly the word 'pong' and nothing else.",
    )

    print(f"\nResponse: {response.text!r}")
    if response.usage_metadata:
        print(f"Tokens:   prompt={response.usage_metadata.prompt_token_count}, "
              f"output={response.usage_metadata.candidates_token_count}, "
              f"total={response.usage_metadata.total_token_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
