"""Smoke test: confirm Arize Phoenix client connection.

Prerequisites:
    .env file in repo root with PHOENIX_API_KEY and PHOENIX_BASE_URL set.

Run:
    python scripts/smoke_test_phoenix.py
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    api_key = os.environ.get("PHOENIX_API_KEY")
    base_url = os.environ.get("PHOENIX_BASE_URL")

    if not api_key:
        print("ERROR: PHOENIX_API_KEY not set in environment / .env", file=sys.stderr)
        return 1
    if not base_url:
        print("ERROR: PHOENIX_BASE_URL not set in environment / .env", file=sys.stderr)
        return 1

    print(f"Phoenix endpoint: {base_url}")
    print(f"API key:          {api_key[:6]}...{api_key[-4:]}")

    try:
        from phoenix.client import Client
    except ImportError:
        print("ERROR: arize-phoenix-client not installed. Run:", file=sys.stderr)
        print("    pip install arize-phoenix-client", file=sys.stderr)
        return 1

    client = Client(base_url=base_url, api_key=api_key)

    try:
        projects = client.projects.list()
        # Phoenix client may return dicts or objects depending on version.
        names = [
            (p.get("name") if isinstance(p, dict) else getattr(p, "name", str(p)))
            for p in projects
        ]
        print(f"\nProjects ({len(names)}): {names}")
        print("\n[OK] Phoenix connection works.")
        return 0
    except Exception as exc:
        print(f"\n[FAIL] Phoenix request error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
