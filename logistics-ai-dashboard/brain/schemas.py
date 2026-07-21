"""
brain/schemas.py — framework-free contracts for the Decision Brain, the long-term
memory + knowledge system. No business logic; these are the records that flow
into the memory store and out of the retriever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MemoryKind(str, Enum):
    DECISION = "decision"           # an executive decision made by the Planner
    KNOWLEDGE = "knowledge"         # SOPs, policies, contracts, reports, notes
    RECOMMENDATION = "recommendation"
    OUTCOME = "outcome"             # a realised result tied to a recommendation/decision
    FEEDBACK = "feedback"           # user feedback
    APPROVAL = "approval"           # approve / reject / schedule / escalate history
    ENTITY = "entity"               # supplier / customer information


@dataclass
class MemoryRecord:
    id: str
    kind: MemoryKind
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind.value, "title": self.title,
                "content": self.content, "metadata": self.metadata,
                "ts": self.ts, "source": self.source}


@dataclass
class RetrievalResult:
    record: MemoryRecord
    score: float
    semantic: float
    lexical: float

    def to_dict(self) -> dict:
        d = self.record.to_dict()
        d.update(score=round(self.score, 4), semantic=round(self.semantic, 4),
                 lexical=round(self.lexical, 4),
                 snippet=(self.record.content[:280] + ("…" if len(self.record.content) > 280 else "")))
        return d
