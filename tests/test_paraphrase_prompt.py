"""The paraphrase flag threads a similarity instruction into the prompt.

When a caller asks for paraphrase-tolerant tests, the user message must tell
the model to assert embedding similarity via
``phoenix2pytest.assertions.assert_paraphrase_similar`` and take an ``embedder``
fixture, instead of an exact string match. These tests pin that wiring with a
capturing stub, so no live model is called.
"""

from __future__ import annotations

from phoenix2pytest.synthesiser import (
    FailureDetails,
    TraceData,
    build_user_message,
    build_user_message_for_group,
    synthesise,
    synthesise_many,
)


class _CaptureGemini:
    """Records the user message and returns a minimal valid test module."""

    def __init__(self) -> None:
        self.user: str | None = None

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        self.user = user
        return "def test_x():\n    assert True\n"


def test_single_message_paraphrase_mentions_helper_and_fixture() -> None:
    msg = build_user_message(
        TraceData(user_prompt="hi"),
        FailureDetails(failure_mode="hallucination"),
        paraphrase=True,
    )
    assert "assert_paraphrase_similar" in msg
    assert "embedder" in msg


def test_single_message_default_has_no_paraphrase_instruction() -> None:
    msg = build_user_message(
        TraceData(user_prompt="hi"),
        FailureDetails(failure_mode="hallucination"),
    )
    assert "assert_paraphrase_similar" not in msg


def test_group_message_paraphrase_mentions_helper() -> None:
    msg = build_user_message_for_group(
        [TraceData(user_prompt="a"), TraceData(user_prompt="b")],
        FailureDetails(failure_mode="m"),
        paraphrase=True,
    )
    assert "assert_paraphrase_similar" in msg


def test_synthesise_threads_paraphrase_into_prompt() -> None:
    client = _CaptureGemini()
    synthesise(
        TraceData(user_prompt="hi"),
        FailureDetails(failure_mode="m"),
        client,
        paraphrase=True,
    )
    assert client.user is not None
    assert "assert_paraphrase_similar" in client.user


def test_synthesise_many_threads_paraphrase_into_prompt() -> None:
    client = _CaptureGemini()
    synthesise_many(
        [(TraceData(user_prompt="hi"), FailureDetails(failure_mode="m"))],
        client,
        paraphrase=True,
    )
    assert client.user is not None
    assert "assert_paraphrase_similar" in client.user
