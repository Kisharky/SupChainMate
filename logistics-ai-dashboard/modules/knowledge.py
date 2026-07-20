"""
modules/knowledge.py
Knowledge Base — RAG over the business's own documents (SOPs, procurement
policies, contracts, supplier manuals, warehouse procedures).

Retrieval is TF-IDF cosine similarity (scikit-learn, fully offline);
generation is optional: with a Groq key the top passages are composed into
a grounded answer with citations, without one the retrieved passages are
returned extractively. Retrieval never fabricates — every answer points at
the source chunks it came from.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

import config
from modules import groq_ai, store

_log = config.get_logger(__name__)

CHUNK_CHARS = 900
TOP_K = 4
MIN_SCORE = 0.05


def chunk_text(text: str, name: str) -> list[dict]:
    """Paragraph-aware chunks of ~CHUNK_CHARS with document attribution."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", str(text)) if p.strip()]
    chunks: list[dict] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) > CHUNK_CHARS and buf:
            chunks.append({"doc": name, "text": buf.strip()})
            buf = ""
        buf += p + "\n\n"
    if buf.strip():
        chunks.append({"doc": name, "text": buf.strip()})
    return chunks


def _all_chunks() -> list[dict]:
    chunks: list[dict] = []
    for doc in store.load_documents():
        chunks += chunk_text(doc["content"], doc["name"])
    return chunks


def _embedding_retrieve(query: str, chunks: list[dict], top_k: int) -> Optional[list[dict]]:
    """Semantic retrieval via the embedding capability (nemotron-3-embed-1b).
    Returns None when embeddings are unavailable so the caller falls back."""
    try:
        from ai import embeddings as ai_embeddings
        if not ai_embeddings.available():
            return None
        doc_vecs = ai_embeddings.embed_documents([c["text"] for c in chunks])
        q_vec = ai_embeddings.embed_query(query)
        if doc_vecs is None or q_vec is None:
            return None
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity([q_vec], doc_vecs).ravel()
    except Exception as e:  # any embedding failure → lexical fallback
        _log.info("Embedding retrieval unavailable (%s) — lexical fallback", e)
        return None
    order = np.argsort(sims)[::-1][:top_k]
    return [{**chunks[i], "score": round(float(sims[i]), 3), "retriever": "embedding"}
            for i in order if sims[i] >= MIN_SCORE]


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Top-k chunks: semantic (embeddings) when available, else TF-IDF +
    char-n-gram cosine. Returns [{doc, text, score, retriever}]."""
    chunks = _all_chunks()
    if not chunks or not str(query).strip():
        return []
    semantic = _embedding_retrieve(query, chunks, top_k)
    if semantic is not None:
        return semantic
    corpus = [c["text"] for c in chunks]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        def _sims(**kwargs) -> np.ndarray:
            try:
                matrix = TfidfVectorizer(**kwargs).fit_transform(corpus + [query])
                return cosine_similarity(matrix[-1], matrix[:-1]).ravel()
            except ValueError:  # e.g. query entirely of stop words
                return np.zeros(len(corpus))

        # Word n-grams for precision; character n-grams so inflections
        # ("approval"/"approvals") and small corpora still match.
        sims = np.maximum(
            _sims(stop_words="english", ngram_range=(1, 2)),
            _sims(analyzer="char_wb", ngram_range=(3, 5)))
    except ImportError:
        return []
    order = np.argsort(sims)[::-1][:top_k]
    return [{**chunks[i], "score": round(float(sims[i]), 3), "retriever": "tfidf"}
            for i in order if sims[i] >= MIN_SCORE]


_RAG_SYSTEM = (
    "Answer the question using ONLY the numbered source passages provided. "
    "Cite passages like [1], [2] after each claim. If the passages don't "
    "contain the answer, say so plainly. Max 5 sentences.")


def answer(query: str) -> dict:
    """
    Answer a question from the knowledge base (RAG).
    Retrieval → context builder → reasoning model (via the AI Router,
    capability reasoning.operations; falls back to Groq/extractive).
    Returns {answer, passages, engine}.
    """
    passages = retrieve(query)
    if not passages:
        return {"answer": ("Nothing in the knowledge base matches that question — "
                           "upload the relevant SOP, policy, or contract first."),
                "passages": [], "engine": "extractive"}

    context_block = "\n\n".join(
        f"[{i+1}] (from {p['doc']}):\n{p['text'][:1200]}"
        for i, p in enumerate(passages))
    try:
        from ai import AI, Capability
        resp = AI.ask(Capability.REASONING_OPERATIONS, "knowledge_qa", context=None,
                      system=_RAG_SYSTEM,
                      user=f"Question: {query}\n\nSources:\n{context_block}")
        if resp.ok and resp.text:
            engine = resp.model_used or ("groq" if resp.fell_back else "reasoning")
            return {"answer": resp.text, "passages": passages, "engine": engine}
    except Exception as e:  # RAG must degrade to extractive, never fail
        _log.info("RAG reasoning unavailable (%s) — extractive answer", e)

    best = passages[0]
    return {"answer": (f"Most relevant passage (from {best['doc']}, "
                       f"similarity {best['score']:.2f}):\n\n{best['text'][:800]}"),
            "passages": passages, "engine": "extractive"}


def kb_stats() -> dict:
    docs = store.load_documents()
    return {"documents": len(docs),
            "chunks": len(_all_chunks()),
            "names": [d["name"] for d in docs]}
