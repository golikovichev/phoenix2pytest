"""Tests for the paraphrase-tolerant assertion helpers.

The embedder is stubbed so these tests run offline and deterministically.
A real embedding backend is only constructed lazily by build_default_embedder
and is never called here.
"""

from __future__ import annotations

import math

import pytest

from phoenix2pytest.assertions import (
    assert_paraphrase_similar,
    cosine_similarity,
)


class _StubEmbedder:
    """Maps known strings to fixed vectors so similarity is deterministic."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed(self, text: str) -> list[float]:
        return self._table[text]


def test_cosine_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_minus_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_does_not_crash() -> None:
    # A zero vector has no direction; define similarity as 0.0 rather than NaN.
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_assert_paraphrase_similar_passes_and_returns_score() -> None:
    embedder = _StubEmbedder(
        {
            "The order shipped.": [1.0, 0.0, 0.0],
            "Your package is on its way.": [0.98, 0.199, 0.0],
        }
    )
    score = assert_paraphrase_similar(
        "The order shipped.",
        "Your package is on its way.",
        embedder=embedder,
        threshold=0.85,
    )
    assert score >= 0.85
    assert math.isclose(score, cosine_similarity([1.0, 0.0, 0.0], [0.98, 0.199, 0.0]))


def test_assert_paraphrase_similar_below_threshold_raises_with_score() -> None:
    embedder = _StubEmbedder(
        {
            "The order shipped.": [1.0, 0.0],
            "The weather is cold.": [0.0, 1.0],
        }
    )
    with pytest.raises(AssertionError) as exc:
        assert_paraphrase_similar(
            "The order shipped.",
            "The weather is cold.",
            embedder=embedder,
            threshold=0.85,
        )
    # The failure message must surface the actual score so a human can tune it.
    assert "0.0" in str(exc.value)


def test_assert_paraphrase_similar_threshold_boundary_is_inclusive() -> None:
    # Score exactly equal to the threshold passes (>=, not >).
    embedder = _StubEmbedder(
        {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
        }
    )
    # cosine here is 0.0; a threshold of 0.0 must pass.
    score = assert_paraphrase_similar("a", "b", embedder=embedder, threshold=0.0)
    assert score == 0.0
