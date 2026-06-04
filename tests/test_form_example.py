"""The landing form ships a ready-to-run example.

A judge opening the hosted demo should be able to click Generate without
hunting for valid input, so the form is pre-filled with an example that
satisfies the same required fields /generate enforces.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

import phoenix2pytest.web as web


def test_example_payload_is_valid_and_complete():
    """The baked-in example must parse and carry the fields /generate requires
    (trace.user_prompt and details.failure_mode), so the default click works."""
    trace = json.loads(web.EXAMPLE_TRACE_JSON)
    details = json.loads(web.EXAMPLE_DETAILS_JSON)
    assert trace.get("user_prompt")
    assert details.get("failure_mode")


def test_form_page_prefills_the_example():
    client = TestClient(web.app)
    body = client.get("/").text
    # The example is rendered into the textareas, not just shown as a placeholder.
    assert "What is the capital of France" in body
    assert "hallucination" in body
