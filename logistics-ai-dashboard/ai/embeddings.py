"""
ai/embeddings.py
Embedding service for RAG and semantic search. Routes to the embedding
capability (nemotron-3-embed-1b) and degrades to None so callers can fall
back to lexical retrieval when no embedding key is set.
"""

from __future__ import annotations

from typing import Optional

import config
from ai.router import AI

_log = config.get_logger(__name__)


def available() -> bool:
    return AI.status().get("embedding", False)


def embed_documents(texts: list[str]) -> Optional[list[list[float]]]:
    """Embed corpus passages. Returns None when embeddings are unavailable."""
    if not texts:
        return []
    resp = AI.embed(texts, input_type="passage")
    if not resp.ok:
        _log.info("Document embedding unavailable: %s", resp.error)
        return None
    return resp.vectors


def embed_query(text: str) -> Optional[list[float]]:
    resp = AI.embed([text], input_type="query")
    if not resp.ok or not resp.vectors:
        return None
    return resp.vectors[0]
