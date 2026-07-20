"""
ai/rag.py
Enterprise RAG pipeline for supply chain documents (invoices, POs,
contracts, SOPs, Incoterms, shipping docs, ERP manuals, policies).

Pipeline:
    documents → intelligent chunking (overlap) → embedding generation
    → cached vector index → semantic retrieval → hybrid ranking
    → citation generation → context builder → reasoning model

Every stage degrades gracefully: with no embedding key, retrieval falls
back to lexical (TF-IDF + char-n-gram); with no reasoning key, answers are
extractive. Chunk embeddings are cached in SQLite by content hash, so
re-embedding is avoided and only new/changed chunks are sent to the model.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from ai.router import AI
from ai.types import Capability
from modules import store

_log = config.get_logger(__name__)

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150         # carry context across chunk boundaries
TOP_K = 4
MIN_SCORE = 0.05
SEMANTIC_WEIGHT = 0.65      # hybrid rank: semantic vs lexical
_RETRIEVAL_TTL_S = 300.0
_retrieval_cache: dict[str, tuple[float, list]] = {}


@dataclass
class Chunk:
    doc: str
    text: str
    hash: str


@dataclass
class Citation:
    index: int
    doc: str
    snippet: str
    score: float
    retriever: str


# ── 1. Intelligent chunking (paragraph-aware, with overlap) ────────────────────

def chunk_document(text: str, name: str) -> list[Chunk]:
    """Split into ~CHUNK_CHARS chunks on paragraph boundaries, overlapping
    CHUNK_OVERLAP chars so a fact split across a boundary stays retrievable."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", str(text)) if p.strip()]
    chunks: list[Chunk] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) > CHUNK_CHARS and buf:
            chunks.append(_mk_chunk(buf.strip(), name))
            buf = buf[-CHUNK_OVERLAP:] if len(buf) > CHUNK_OVERLAP else ""
        buf += p + "\n\n"
    if buf.strip():
        chunks.append(_mk_chunk(buf.strip(), name))
    return chunks


def _mk_chunk(text: str, name: str) -> Chunk:
    return Chunk(doc=name, text=text,
                 hash=hashlib.sha1(f"{name}|{text}".encode()).hexdigest()[:16])


def _all_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in store.load_documents():
        chunks += chunk_document(doc["content"], doc["name"])
    return chunks


# ── 2/3. Embedding generation + cached vector index ───────────────────────────

def _index_conn() -> Optional[sqlite3.Connection]:
    conn = store._conn()
    if conn is None:
        return None
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS doc_embeddings (
            chunk_hash TEXT PRIMARY KEY, doc TEXT, vector TEXT, dim INTEGER)""")
        return conn
    except sqlite3.Error:
        conn.close()
        return None


def _cached_vectors(hashes: list[str]) -> dict[str, list[float]]:
    conn = _index_conn()
    if conn is None or not hashes:
        return {}
    try:
        q = f"SELECT chunk_hash, vector FROM doc_embeddings WHERE chunk_hash IN ({','.join('?' * len(hashes))})"
        return {h: json.loads(v) for h, v in conn.execute(q, hashes).fetchall()}
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


def _store_vectors(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    conn = _index_conn()
    if conn is None:
        return
    try:
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO doc_embeddings (chunk_hash, doc, vector, dim) "
                "VALUES (?,?,?,?)",
                [(c.hash, c.doc, json.dumps(v), len(v))
                 for c, v in zip(chunks, vectors)])
    except sqlite3.Error as e:
        _log.warning("vector index write failed: %s", e)
    finally:
        conn.close()


def build_index(chunks: list[Chunk]) -> Optional[dict[str, list[float]]]:
    """Ensure every chunk has a cached embedding; embed only the misses.
    Returns {hash: vector} or None if embeddings are unavailable."""
    from ai import embeddings as ai_embeddings
    if not ai_embeddings.available():
        return None
    cached = _cached_vectors([c.hash for c in chunks])
    missing = [c for c in chunks if c.hash not in cached]
    if missing:
        vecs = ai_embeddings.embed_documents([c.text for c in missing])
        if vecs is None:
            return None
        _store_vectors(missing, vecs)
        cached.update({c.hash: v for c, v in zip(missing, vecs)})
    return cached


# ── 4/5. Semantic retrieval + hybrid ranking ──────────────────────────────────

def _lexical_scores(query: str, corpus: list[str]) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        def _sims(**kw) -> np.ndarray:
            try:
                m = TfidfVectorizer(**kw).fit_transform(corpus + [query])
                return cosine_similarity(m[-1], m[:-1]).ravel()
            except ValueError:
                return np.zeros(len(corpus))
        return np.maximum(_sims(stop_words="english", ngram_range=(1, 2)),
                          _sims(analyzer="char_wb", ngram_range=(3, 5)))
    except ImportError:
        return np.zeros(len(corpus))


def _corpus_fingerprint(chunks: list[Chunk]) -> str:
    return hashlib.sha1("".join(c.hash for c in chunks).encode()).hexdigest()[:12]


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Hybrid retrieval: semantic (cached embeddings) blended with lexical,
    else lexical-only. Cached per (query, corpus) for a short TTL."""
    chunks = _all_chunks()
    if not chunks or not str(query).strip():
        return []
    from ai import embeddings as ai_embeddings
    # Availability is part of the key so the cache busts if an embedding key
    # is added/removed and the retrieval mode changes.
    mode = "sem" if ai_embeddings.available() else "lex"
    ckey = hashlib.sha1(
        f"{query}|{_corpus_fingerprint(chunks)}|{top_k}|{mode}".encode()).hexdigest()
    hit = _retrieval_cache.get(ckey)
    if hit and hit[0] > time.monotonic():
        return hit[1]

    corpus = [c.text for c in chunks]
    lexical = _lexical_scores(query, corpus)

    index = build_index(chunks)
    if index is not None:
        from ai import embeddings as ai_embeddings
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = ai_embeddings.embed_query(query)
        doc_vecs = [index.get(c.hash) for c in chunks]
        if q_vec is not None and all(v is not None for v in doc_vecs):
            semantic = cosine_similarity([q_vec], doc_vecs).ravel()
            combined = SEMANTIC_WEIGHT * _norm(semantic) + (1 - SEMANTIC_WEIGHT) * _norm(lexical)
            retriever = "hybrid"
        else:
            combined, retriever = lexical, "tfidf"
    else:
        combined, retriever = lexical, "tfidf"

    order = np.argsort(combined)[::-1][:top_k]
    results = [{"doc": chunks[i].doc, "text": chunks[i].text,
                "score": round(float(combined[i]), 3), "retriever": retriever}
               for i in order if combined[i] >= MIN_SCORE]
    _retrieval_cache[ckey] = (time.monotonic() + _RETRIEVAL_TTL_S, results)
    return results


def _norm(a: np.ndarray) -> np.ndarray:
    """Min-max to [0,1]. A uniform array (e.g. a single candidate) maps to 1
    when its similarity is positive — never zero out the only relevant chunk."""
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        return (a - lo) / (hi - lo)
    return np.ones_like(a) if hi > 0 else np.zeros_like(a)


# ── 6. Citations + answer composition ─────────────────────────────────────────

_RAG_SYSTEM = (
    "Answer the question using ONLY the numbered source passages provided. "
    "Cite passages like [1], [2] after each claim. If the passages don't "
    "contain the answer, say so plainly. Max 5 sentences.")


def citations(passages: list[dict]) -> list[Citation]:
    return [Citation(index=i + 1, doc=p["doc"], snippet=p["text"][:200],
                     score=p["score"], retriever=p.get("retriever", "?"))
            for i, p in enumerate(passages)]


def answer(query: str) -> dict:
    """Full RAG answer: retrieve → cite → reason. Returns
    {answer, passages, citations, engine, retriever}."""
    passages = retrieve(query)
    if not passages:
        return {"answer": ("Nothing in the knowledge base matches that question — "
                           "upload the relevant SOP, policy, or contract first."),
                "passages": [], "citations": [], "engine": "extractive",
                "retriever": "none"}

    context_block = "\n\n".join(
        f"[{i+1}] (from {p['doc']}):\n{p['text'][:1200]}"
        for i, p in enumerate(passages))
    cites = citations(passages)
    try:
        resp = AI.ask(Capability.REASONING_OPERATIONS, "knowledge_qa", context=None,
                      system=_RAG_SYSTEM,
                      user=f"Question: {query}\n\nSources:\n{context_block}")
        if resp.ok and resp.text:
            engine = resp.model_used or ("groq" if resp.fell_back else "reasoning")
            return {"answer": resp.text, "passages": passages, "citations": cites,
                    "engine": engine, "retriever": passages[0]["retriever"]}
    except Exception as e:
        _log.info("RAG reasoning unavailable (%s) — extractive", e)

    best = passages[0]
    return {"answer": (f"Most relevant passage (from {best['doc']}, "
                       f"similarity {best['score']:.2f}):\n\n{best['text'][:800]}"),
            "passages": passages, "citations": cites, "engine": "extractive",
            "retriever": best["retriever"]}


def kb_stats() -> dict:
    docs = store.load_documents()
    chunks = _all_chunks()
    indexed = len(_cached_vectors([c.hash for c in chunks])) if chunks else 0
    return {"documents": len(docs), "chunks": len(chunks),
            "indexed_chunks": indexed, "names": [d["name"] for d in docs]}
