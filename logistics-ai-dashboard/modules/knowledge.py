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


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Top-k chunks by TF-IDF cosine similarity: [{doc, text, score}]."""
    chunks = _all_chunks()
    if not chunks or not str(query).strip():
        return []
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
    return [{**chunks[i], "score": round(float(sims[i]), 3)}
            for i in order if sims[i] >= MIN_SCORE]


def answer(query: str) -> dict:
    """
    Answer a question from the knowledge base.
    Returns {answer, passages, engine} — engine is 'groq' (grounded
    composition) or 'extractive' (passages only, zero keys needed).
    """
    passages = retrieve(query)
    if not passages:
        return {"answer": ("Nothing in the knowledge base matches that question — "
                           "upload the relevant SOP, policy, or contract first."),
                "passages": [], "engine": "extractive"}

    if groq_ai.is_available():
        context_block = "\n\n".join(
            f"[{i+1}] (from {p['doc']}):\n{p['text'][:1200]}"
            for i, p in enumerate(passages))
        composed = groq_ai._call(
            messages=[
                {"role": "system", "content":
                    "Answer the question using ONLY the numbered source passages "
                    "provided. Cite passages like [1], [2] after each claim. If the "
                    "passages don't contain the answer, say so plainly. Max 5 sentences."},
                {"role": "user", "content": f"Question: {query}\n\nSources:\n{context_block}"},
            ],
            max_tokens=350, temperature=0.2)
        if composed and not composed.startswith("["):
            return {"answer": composed, "passages": passages, "engine": "groq"}

    best = passages[0]
    return {"answer": (f"Most relevant passage (from {best['doc']}, "
                       f"similarity {best['score']:.2f}):\n\n{best['text'][:800]}"),
            "passages": passages, "engine": "extractive"}


def kb_stats() -> dict:
    docs = store.load_documents()
    return {"documents": len(docs),
            "chunks": len(_all_chunks()),
            "names": [d["name"] for d in docs]}
