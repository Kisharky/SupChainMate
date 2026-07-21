"""
brain/brain.py — the Decision Brain: SupChainMate's long-term memory + knowledge
system (Hermes-style retrieval-augmented memory).

It stores every Planner decision, company knowledge, past recommendations and
their outcomes, and user feedback / approvals — and retrieves the most relevant
of them whenever the Planner needs context. It is offline-first (local vector
store + embeddings) and model-agnostic (any synthesis flows through the
provider-agnostic AI Router). It consumes the existing systems read-only and
never modifies them.
"""

from __future__ import annotations

from typing import Optional

from brain.retriever import Retriever
from brain.schemas import MemoryKind, RetrievalResult
from brain.store import MemoryStore


class DecisionBrain:
    def __init__(self, store: Optional[MemoryStore] = None,
                 retriever: Optional[Retriever] = None) -> None:
        self.store = store or MemoryStore()
        self.retriever = retriever or Retriever(self.store)

    # ---- write (typed helpers) ----
    def record_decision(self, objective: str, summary: str, metadata: dict) -> str:
        return self.store.add(MemoryKind.DECISION, f"Decision · {objective}", summary,
                              metadata=metadata, source="planner")

    def add_knowledge(self, title: str, content: str, doc_type: str = "document",
                      source: str = "user") -> str:
        return self.store.add(MemoryKind.KNOWLEDGE, title, content,
                              metadata={"doc_type": doc_type}, source=source)

    def record_recommendation(self, title: str, content: str, metadata: dict, source="trust") -> str:
        return self.store.add(MemoryKind.RECOMMENDATION, title, content, metadata=metadata, source=source)

    def record_outcome(self, title: str, content: str, metadata: dict) -> str:
        return self.store.add(MemoryKind.OUTCOME, title, content, metadata=metadata, source="outcome")

    def record_feedback(self, title: str, content: str, metadata: dict) -> str:
        return self.store.add(MemoryKind.FEEDBACK, title, content, metadata=metadata, source="user")

    def record_approval(self, title: str, content: str, metadata: dict) -> str:
        return self.store.add(MemoryKind.APPROVAL, title, content, metadata=metadata, source="audit")

    def record_entity(self, name: str, content: str, kind: str = "supplier") -> str:
        return self.store.add(MemoryKind.ENTITY, f"{kind}: {name}", content,
                              metadata={"entity_type": kind}, source="crm")

    # ---- read ----
    def recall(self, query: str, kinds: Optional[list[str]] = None, top_k: int = 6) -> list[RetrievalResult]:
        mk = [MemoryKind(k) for k in kinds] if kinds else None
        return self.retriever.search(query, kinds=mk, top_k=top_k)

    def context_for(self, objective: str, top_k: int = 5) -> dict:
        """Relevant past decisions + knowledge for the Planner, as compact context."""
        hits = self.recall(objective, top_k=top_k)
        lines = [f"[{h.record.kind.value}] {h.record.title}: "
                 f"{h.record.content[:200]}" for h in hits]
        return {"objective": objective,
                "context": "\n".join(lines),
                "citations": [h.to_dict() for h in hits],
                "n": len(hits)}

    def answer(self, query: str, top_k: int = 6) -> dict:
        """Retrieve then reason (model-agnostic via the AI Router, extractive fallback)."""
        hits = self.recall(query, top_k=top_k)
        if not hits:
            return {"answer": "No relevant memory found yet. Ingest knowledge or run the "
                              "Planner to populate the Decision Brain.", "citations": []}
        context = "\n\n".join(f"[{i+1}] ({h.record.kind.value}) {h.record.title}\n{h.record.content[:500]}"
                              for i, h in enumerate(hits))
        answer = self._synthesise(query, context) or self._extractive(hits)
        return {"answer": answer, "citations": [h.to_dict() for h in hits],
                "engine": "ai-router" if self._ai_ok else "extractive"}

    _ai_ok = False

    def _synthesise(self, query: str, context: str) -> Optional[str]:
        try:
            from ai import AI
            prompt = (f"Answer the question using ONLY the memory below; cite sources as [n]. "
                      f"If the memory is insufficient, say so.\n\nMEMORY:\n{context}\n\n"
                      f"QUESTION: {query}\nANSWER:")
            resp = AI.ask("reasoning.operations", task="brain_answer", user=prompt)
            text = getattr(resp, "text", None)
            if text and len(text) > 30:
                self._ai_ok = True
                return text.strip()
        except Exception:  # noqa: BLE001
            pass
        return None

    def _extractive(self, hits: list[RetrievalResult]) -> str:
        top = hits[0].record
        return (f"Most relevant memory — {top.title}: {top.content[:400]}"
                + (f" (and {len(hits) - 1} more related record(s))." if len(hits) > 1 else "."))

    def stats(self) -> dict:
        return self.store.stats()

    def recent(self, limit: int = 30) -> list[dict]:
        return [r.to_dict() for r in self.store.recent(limit)]

    # ---- integration: one-time read-only ingest of existing state ----
    def ingest_existing(self) -> dict:
        counts = {k.value: 0 for k in MemoryKind}

        # Company knowledge — the RAG knowledge base / uploaded documents
        try:
            from modules import store as sstore
            for doc in sstore.load_documents():
                if self.add_knowledge(doc.get("name", "document"), doc.get("content", ""),
                                      source="kb"):
                    counts[MemoryKind.KNOWLEDGE.value] += 1
        except Exception:  # noqa: BLE001
            pass

        # Past Planner decisions
        try:
            from planner import PLANNER
            for run in PLANNER.history(100):
                self.record_decision(run["objective"], run.get("recommendation", ""),
                                     {"run_id": run["id"], "capabilities": run.get("capabilities", []),
                                      "ts": run.get("ts")})
                counts[MemoryKind.DECISION.value] += 1
        except Exception:  # noqa: BLE001
            pass

        # Recommendations + their decisions (approvals / outcomes)
        try:
            from modules import store as sstore
            for rec in sstore.load_recommendations(limit=300):
                impact = (rec.get("impact") or {}).get("cost_savings_yr")
                self.record_recommendation(
                    rec.get("title", "recommendation"),
                    rec.get("action", ""),
                    {"rec_key": rec.get("rec_key"), "status": rec.get("status"),
                     "confidence": rec.get("confidence"), "impact_usd": impact})
                counts[MemoryKind.RECOMMENDATION.value] += 1
                if rec.get("status") in ("APPROVED", "MODIFIED", "REJECTED", "ESCALATED"):
                    self.record_approval(
                        f"{rec.get('status')} · {rec.get('title', '')}",
                        f"{rec.get('decided_by', 'user')} {rec.get('status', '').lower()} "
                        f"this recommendation. Note: {rec.get('note') or '—'}",
                        {"rec_key": rec.get("rec_key"), "status": rec.get("status"),
                         "decided_by": rec.get("decided_by")})
                    counts[MemoryKind.APPROVAL.value] += 1
        except Exception:  # noqa: BLE001
            pass

        return {"ingested": counts, "stats": self.stats()}


class _BrainFacade:
    """Process-wide singleton, lazily built (mirrors ai.AI / optimize.OPT / PLANNER)."""
    def __init__(self) -> None:
        self._brain: Optional[DecisionBrain] = None

    @property
    def brain(self) -> DecisionBrain:
        if self._brain is None:
            self._brain = DecisionBrain()
        return self._brain

    def __getattr__(self, item):
        return getattr(self.brain, item)


BRAIN = _BrainFacade()
