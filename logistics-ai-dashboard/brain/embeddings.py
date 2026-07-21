"""
brain/embeddings.py — the embedding backend for semantic search. Offline-first
and model-agnostic behind a tiny port.

* ``LocalHashingEmbedder`` (default) is a fully-offline, deterministic, stateless
  vectoriser (scikit-learn HashingVectorizer). No model download, no network — so
  SupChainMate runs on air-gapped servers with open-source SLMs out of the box.
* ``RouterEmbedder`` delegates to the provider-agnostic AI Router (NVIDIA / any
  configured provider) when a business prefers hosted embeddings; the whole index
  must use one embedder so the vector space is consistent.

Swap the embedder in one place (``default_embedder``) and re-index — the retriever
and store are agnostic to which backend produced the vectors.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

EMBED_DIM = 1024


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, text: str) -> np.ndarray: ...


class LocalHashingEmbedder:
    """Offline, deterministic embeddings — the default for on-prem deployments."""
    name = "local-hashing"

    def __init__(self, dim: int = EMBED_DIM) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer
        self.dim = dim
        self._vec = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm="l2",
            ngram_range=(1, 2), stop_words="english")

    def embed(self, text: str) -> np.ndarray:
        v = self._vec.transform([text or ""]).toarray()[0].astype("float32")
        return v


class RouterEmbedder:
    """Hosted embeddings via the provider-agnostic AI Router (optional)."""
    name = "ai-router"

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        from ai import AI
        resp = AI.embed([text or ""])
        vecs = getattr(resp, "vectors", None) or getattr(resp, "embeddings", None)
        if not vecs:
            raise RuntimeError("router returned no embedding")
        v = np.asarray(vecs[0], dtype="float32")
        n = np.linalg.norm(v)
        return v / n if n else v


_default: Embedder | None = None


def default_embedder() -> Embedder:
    """The active embedding backend (offline local by default)."""
    global _default
    if _default is None:
        _default = LocalHashingEmbedder()
    return _default


def set_embedder(embedder: Embedder) -> None:
    global _default
    _default = embedder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # vectors are L2-normalised, so cosine == dot product
    return float(np.dot(a, b))
