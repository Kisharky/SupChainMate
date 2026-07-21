"""
brain/retriever.py — hybrid semantic + lexical retrieval over the memory store.
Cosine similarity on the offline vector index, blended with a lexical overlap
signal so exact terms (SKU codes, supplier names, policy numbers) still rank.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from brain.embeddings import cosine, default_embedder
from brain.schemas import MemoryKind, MemoryRecord, RetrievalResult
from brain.store import MemoryStore

SEMANTIC_WEIGHT = 0.7


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


class Retriever:
    def __init__(self, store: Optional[MemoryStore] = None) -> None:
        self.store = store or MemoryStore()
        self.embedder = default_embedder()

    def search(self, query: str, kinds: Optional[list[MemoryKind]] = None,
               top_k: int = 6) -> list[RetrievalResult]:
        corpus = self.store.all(kinds)
        if not corpus:
            return []
        qv = self.embedder.embed(query)
        qtok = _tokens(query)
        scored: list[RetrievalResult] = []
        for rec, vec in corpus:
            sem = cosine(qv, vec) if vec.shape == qv.shape else 0.0
            rtok = _tokens(rec.title + " " + rec.content)
            lex = len(qtok & rtok) / max(len(qtok), 1)
            score = SEMANTIC_WEIGHT * sem + (1 - SEMANTIC_WEIGHT) * lex
            scored.append(RetrievalResult(record=rec, score=score, semantic=sem, lexical=lex))
        scored.sort(key=lambda r: r.score, reverse=True)
        return [r for r in scored if r.score > 0.02][:top_k]
