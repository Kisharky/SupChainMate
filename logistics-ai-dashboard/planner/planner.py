"""
planner/planner.py — the executive decision orchestrator. Given a business
objective it: understands it → discovers the required capabilities → builds an
execution graph → executes existing systems (concurrently where independent) →
merges results → produces ONE executive Decision → records it to memory.

It contains NO business logic: every domain computation happens inside an
existing system, reached through a registered capability. Structurally this
mirrors ai.AI and optimize.OPT — a thin facade over injected collaborators.
"""

from __future__ import annotations

from typing import Optional

from planner.aggregator import Aggregator
from planner.capabilities import register_default_capabilities
from planner.executor import Executor
from planner.graph import ExecutionGraph
from planner.memory import PlannerMemory
from planner.registry import CapabilityRegistry
from planner.schemas import Decision


class Planner:
    def __init__(self, registry: Optional[CapabilityRegistry] = None,
                 executor: Optional[Executor] = None,
                 aggregator: Optional[Aggregator] = None,
                 memory: Optional[PlannerMemory] = None) -> None:
        self.registry = registry or register_default_capabilities(CapabilityRegistry())
        self.executor = executor or Executor(self.registry)
        self.aggregator = aggregator or Aggregator()
        self.memory = memory or PlannerMemory()

    def plan(self, objective: str, context: Optional[dict] = None, record: bool = True) -> Decision:
        capabilities = self.registry.select(objective)          # discover
        layers = ExecutionGraph(capabilities).build()           # graph
        results = self.executor.run(layers, context or {})      # execute
        decision = self.aggregator.aggregate(objective, results, layers, capabilities)  # merge
        if record:
            decision.run_id = self.memory.record(
                objective=objective, graph=layers,
                capabilities=[c.name for c in capabilities],
                outputs={n: r.to_dict() for n, r in results.items()},
                recommendation=decision.executive_summary,
                predicted={"financial_impact": decision.financial_impact,
                           "confidence": decision.confidence})
        return decision

    def capabilities(self) -> list[dict]:
        return self.registry.meta()

    def history(self, limit: int = 20) -> list[dict]:
        return self.memory.recent(limit)


class _PlannerFacade:
    """Process-wide singleton, lazily built (mirrors ai.AI / optimize.OPT)."""
    def __init__(self) -> None:
        self._planner: Optional[Planner] = None

    def configure(self, planner: Planner) -> None:
        self._planner = planner

    @property
    def planner(self) -> Planner:
        if self._planner is None:
            self._planner = Planner()
        return self._planner

    def plan(self, objective: str, context: Optional[dict] = None) -> Decision:
        return self.planner.plan(objective, context)

    def capabilities(self) -> list[dict]:
        return self.planner.capabilities()

    def history(self, limit: int = 20) -> list[dict]:
        return self.planner.history(limit)

    def register(self, capability) -> None:
        """Extensibility: register a future capability at runtime — the Planner
        picks it up on the next plan() with no core change."""
        self.planner.registry.register(capability)


PLANNER = _PlannerFacade()
