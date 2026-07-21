"""Tests for the Decision Brain — fully offline (local embeddings, SQLite store)."""

from brain import BRAIN, MemoryKind
from brain.embeddings import LocalHashingEmbedder, cosine
from brain.retriever import Retriever
from brain.store import MemoryStore


def test_local_embedder_is_offline_and_deterministic():
    e = LocalHashingEmbedder()
    a = e.embed("expedite air freight policy")
    b = e.embed("expedite air freight policy")
    assert a.shape == (e.dim,)
    assert cosine(a, b) > 0.999            # deterministic
    assert cosine(a, e.embed("returns and restocking")) < cosine(a, b)  # discriminates


def test_store_roundtrip_and_stats():
    s = MemoryStore()
    rid = s.add(MemoryKind.KNOWLEDGE, "SLA policy", "On-time target is 98% for tier-1 SKUs.")
    assert s.has(rid)
    assert s.stats()["total"] >= 1
    assert s.stats()["embedder"] == "local-hashing"


def test_semantic_recall_ranks_relevant_first():
    BRAIN.add_knowledge("Expedite policy", "Air freight authorised up to $5,000 for critical SKUs below 3 days of cover.")
    BRAIN.add_knowledge("Returns policy", "Returns accepted within 30 days, 10% restocking fee.")
    hits = BRAIN.recall("what is our air freight expedite policy?", top_k=3)
    assert hits
    assert "expedite" in hits[0].record.title.lower()
    # semantic + lexical are both surfaced
    assert hits[0].semantic >= 0 and hits[0].lexical >= 0


def test_kind_filter():
    BRAIN.record_decision("Cut logistics cost", "Re-route network; $X saved.", {"run_id": "t"})
    hits = BRAIN.recall("logistics cost decision", kinds=["decision"], top_k=5)
    assert all(h.record.kind == MemoryKind.DECISION for h in hits)


def test_context_for_planner_returns_compact_context():
    BRAIN.add_knowledge("Holding cost policy", "Target inventory turns of 8x; review slow movers monthly.")
    ctx = BRAIN.context_for("reduce inventory holding cost", top_k=3)
    assert ctx["n"] >= 1
    assert isinstance(ctx["context"], str) and ctx["context"]
    assert len(ctx["citations"]) == ctx["n"]


def test_answer_is_offline_extractive_with_citations():
    BRAIN.add_knowledge("Fuel surcharge", "Fuel surcharge is 5% passed through on all lanes.")
    a = BRAIN.answer("what is the fuel surcharge?")
    assert a["answer"]
    assert a["citations"]
