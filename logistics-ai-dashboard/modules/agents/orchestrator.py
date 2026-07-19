"""
modules/agents/orchestrator.py
The Agent Orchestrator: runs multi-step workflows, passes context between
agents, routes recommendations into the Decision Center (human approval),
and audit-logs every step.

Workflows are ordered pipelines; an agent's declared `depends_on` controls
which upstream outputs it may read. The orchestrator validates that every
dependency appears earlier in the pipeline before running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config
from modules import store, trust
from modules.agents.base import AgentResult, BaseAgent
from modules.agents.domain import (DemandForecastAgent, ExecutiveAgent,
                                   InventoryAgent, LogisticsAgent,
                                   ProcurementAgent, SupplierRiskAgent,
                                   SustainabilityAgent, WarehouseAgent)

_log = config.get_logger(__name__)


@dataclass
class WorkflowRun:
    workflow: str
    results: list[AgentResult] = field(default_factory=list)
    recommendations_created: int = 0

    @property
    def total_ms(self) -> int:
        return sum(r.duration_ms for r in self.results)


class Orchestrator:
    """Coordinates registered agents through named workflows."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._workflows: dict[str, list[str]] = {}

    # ── Registration (open/closed: extend by registering, not editing) ────────
    def register(self, agent: BaseAgent) -> "Orchestrator":
        self._agents[agent.name] = agent
        return self

    def define_workflow(self, name: str, agent_names: list[str]) -> "Orchestrator":
        missing = [a for a in agent_names if a not in self._agents]
        if missing:
            raise ValueError(f"Workflow '{name}' references unknown agents: {missing}")
        # every declared dependency must run earlier in the pipeline
        seen: set[str] = set()
        for a in agent_names:
            unmet = [d for d in self._agents[a].depends_on
                     if d in agent_names and d not in seen]
            if unmet:
                raise ValueError(f"Workflow '{name}': agent '{a}' depends on {unmet} "
                                 f"which run after it")
            seen.add(a)
        self._workflows[name] = agent_names
        return self

    @property
    def workflows(self) -> dict[str, list[str]]:
        return dict(self._workflows)

    @property
    def agents(self) -> dict[str, BaseAgent]:
        return dict(self._agents)

    # ── Execution ─────────────────────────────────────────────────────────────
    def run_workflow(self, name: str, shared: dict[str, Any],
                     sync_to_decision_center: bool = True) -> WorkflowRun:
        if name not in self._workflows:
            raise ValueError(f"Unknown workflow: {name}")
        run = WorkflowRun(workflow=name)
        upstream: dict[str, dict[str, Any]] = {}
        upstream_conf: dict[str, float] = {}

        # Agent memory: expose every agent's previous persisted outputs so
        # downstream reasoning can compare run-over-run.
        shared = dict(shared)
        shared["memory"] = store.last_agent_runs()

        store.log_event("orchestrator", "workflow_started", details=name)
        _log.info("Workflow '%s': %s", name, " -> ".join(self._workflows[name]))

        for agent_name in self._workflows[name]:
            agent = self._agents[agent_name]
            result = agent.execute(shared, upstream, upstream_conf)
            run.results.append(result)
            upstream[agent_name] = result.outputs
            upstream_conf[agent_name] = result.confidence
            if result.ok and result.outputs:
                store.save_agent_run(name, agent_name, result.confidence, result.outputs)
            store.log_event(
                "orchestrator", "agent_run",
                details=(f"{name}/{agent_name}: confidence {result.confidence:.0f}%, "
                         f"{len(result.recommendations)} rec(s), {result.duration_ms} ms"
                         + (f", ERROR: {result.error}" if result.error else "")))
            if sync_to_decision_center and result.recommendations:
                run.recommendations_created += trust.sync_recommendations(
                    result.recommendations)

        store.log_event("orchestrator", "workflow_completed",
                        details=(f"{name}: {len(run.results)} agents, "
                                 f"{run.recommendations_created} new recommendation(s), "
                                 f"{run.total_ms} ms"))
        return run


def build_default_orchestrator() -> Orchestrator:
    """The standard eight-agent roster and its three built-in workflows."""
    orch = Orchestrator()
    for agent in (DemandForecastAgent(), InventoryAgent(), ProcurementAgent(),
                  LogisticsAgent(), SupplierRiskAgent(), WarehouseAgent(),
                  SustainabilityAgent(), ExecutiveAgent()):
        orch.register(agent)
    orch.define_workflow("planning_chain",
                         ["demand_forecast", "inventory", "procurement", "executive"])
    orch.define_workflow("logistics_review",
                         ["logistics", "supplier_risk", "sustainability", "executive"])
    orch.define_workflow("full_control_tower",
                         ["demand_forecast", "inventory", "procurement", "logistics",
                          "supplier_risk", "warehouse", "sustainability", "executive"])
    return orch
