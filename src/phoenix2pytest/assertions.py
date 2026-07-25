"""Paraphrase-tolerant assertions for generated regression tests.

A generated test can assert that a model reply *means* the same thing as a
reference answer instead of matching it byte for byte. The test takes an
``embedder`` (any object with an ``embed(text) -> list[float]`` method) so it
stays deterministic under a stub and never forces a network call at import
time. ``build_default_embedder`` wraps the google-genai embedding API and is
constructed lazily, only when a caller actually wants the live backend.
"""

from __future__ import annotations

import math
from typing import Protocol


class Embedder(Protocol):
    """Anything that turns text into a dense vector.

    Kept minimal so a test can inject a stub returning fixed vectors.
    """

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, in ``[-1.0, 1.0]``.

    Returns ``0.0`` when either vector has zero magnitude: a zero vector has no
    direction, so ``0.0`` is more useful to a caller than a ``NaN`` from a
    division by zero. Raises :class:`ValueError` when the lengths differ, since
    comparing vectors from different embedding spaces is a caller bug.
    """
    if len(a) != len(b):
        raise ValueError(
            f"vector length mismatch: {len(a)} != {len(b)}; "
            "both texts must be embedded by the same model"
        )
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def assert_paraphrase_similar(
    actual: str,
    expected: str,
    *,
    embedder: Embedder,
    threshold: float = 0.85,
) -> float:
    """Assert ``actual`` is a paraphrase of ``expected`` by embedding similarity.

    Embeds both strings with ``embedder``, computes their cosine similarity,
    and raises :class:`AssertionError` when the score is below ``threshold``.
    The boundary is inclusive: a score equal to ``threshold`` passes. Returns
    the score on success so a caller can log or tune it.

    The failure message carries the score and the threshold so a human reading
    a CI log can decide whether the model drifted or the threshold is too tight.
    """
    score = cosine_similarity(embedder.embed(actual), embedder.embed(expected))
    if score < threshold:
        raise AssertionError(
            f"paraphrase similarity {score:.4f} is below threshold {threshold:.4f}\n"
            f"  actual:   {actual!r}\n"
            f"  expected: {expected!r}"
        )
    return score


def build_default_embedder(
    *,
    model: str = "text-embedding-004",
    api_key: str | None = None,
) -> Embedder:
    """Build an :class:`Embedder` backed by the google-genai embedding API.

    Imported and constructed lazily so importing this module never requires the
    genai client or a network call. The API key falls back to the
    ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` environment variables that
    google-genai reads by default.
    """
    from google import genai  # local import: keep module import side-effect free

    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    class _GenaiEmbedder:
        def embed(self, text: str) -> list[float]:
            result = client.models.embed_content(model=model, contents=text)
            embeddings = getattr(result, "embeddings", None) or []
            if not embeddings:
                raise ValueError(f"embedding API returned no vector for text: {text!r}")
            return list(embeddings[0].values)

    return _GenaiEmbedder()
