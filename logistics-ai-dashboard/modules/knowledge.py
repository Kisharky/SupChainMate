"""
modules/knowledge.py
Knowledge Base — thin adapter over the enterprise RAG pipeline in `ai/rag.py`.

The full pipeline (intelligent chunking → embedding generation → cached
vector index → semantic + hybrid retrieval → citation → reasoning) lives in
the AI layer. This module preserves the historical public API used by the
app and the copilot tools so nothing downstream had to change.
"""

from __future__ import annotations

from ai import rag


def chunk_text(text: str, name: str) -> list[dict]:
    """Backward-compatible chunk view: [{doc, text}]."""
    return [{"doc": c.doc, "text": c.text} for c in rag.chunk_document(text, name)]


def retrieve(query: str, top_k: int = rag.TOP_K) -> list[dict]:
    """Top-k passages via the RAG retriever (hybrid semantic+lexical / TF-IDF)."""
    return rag.retrieve(query, top_k=top_k)


def answer(query: str) -> dict:
    """Grounded, cited answer via the RAG pipeline."""
    return rag.answer(query)


def kb_stats() -> dict:
    return rag.kb_stats()
