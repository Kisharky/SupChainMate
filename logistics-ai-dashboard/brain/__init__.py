"""
brain/ — the Decision Brain: SupChainMate's long-term memory + knowledge system.

Stores every Planner decision, company knowledge (SOPs, policies, contracts,
reports, supplier/customer info, notes), past recommendations and outcomes, and
user feedback / approvals — and retrieves the relevant ones on demand via
semantic search over a local, offline vector store. Model-agnostic (synthesis
runs through the provider-agnostic AI Router) and fully offline-capable.

    from brain import BRAIN
    BRAIN.add_knowledge("Expedite policy", "Air freight authorised up to $5,000 ...")
    hits = BRAIN.recall("what is our expedite policy?")
"""

from brain.brain import BRAIN, DecisionBrain
from brain.schemas import MemoryKind, MemoryRecord, RetrievalResult

__all__ = ["BRAIN", "DecisionBrain", "MemoryKind", "MemoryRecord", "RetrievalResult"]
