"""
planner/schemas.py — framework-free contracts for the Planner (the executive
decision orchestrator). No business logic lives here; these are the shapes that
flow between the registry, graph, executor, aggregator, and memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# A capability handler is a pure adapter: it receives the accumulated context
# (base inputs + upstream outputs) and returns a normalised result dict:
#   {summary: str, findings: list[str], metrics: dict, impact_usd: float|None,
#    confidence: float}
# The handler is the ONLY place that knows a concrete service's shape.
Handler = Callable[[dict], dict]


@dataclass
class Capability:
    """Self-describing unit of intelligence the Planner can invoke.

    Registering one (with its metadata + handler) makes it immediately
    discoverable — the Planner never hardcodes a call to it.
    """
    name: str
    description: str
    required_inputs: list[str]
    outputs: list[str]
    dependencies: list[str]
    confidence: float                 # nominal 0–1 self-confidence
    priority: int                     # lower runs earlier when unordered
    handler: Handler
    keywords: list[str] = field(default_factory=list)

    def to_meta(self) -> dict:
        return {"name": self.name, "description": self.description,
                "required_inputs": self.required_inputs, "outputs": self.outputs,
                "dependencies": self.dependencies, "confidence": self.confidence,
                "priority": self.priority, "keywords": self.keywords}


@dataclass
class TaskResult:
    capability: str
    ok: bool
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    impact_usd: Optional[float] = None
    confidence: float = 0.5
    duration_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {"capability": self.capability, "ok": self.ok, "summary": self.summary,
                "findings": self.findings, "metrics": self.metrics,
                "impact_usd": self.impact_usd, "confidence": round(self.confidence, 2),
                "duration_ms": self.duration_ms, "error": self.error}


@dataclass
class Decision:
    """The single executive decision the Planner produces."""
    objective: str
    executive_summary: str
    key_findings: list[str]
    recommended_actions: list[dict]          # {action, impact_usd, confidence}
    financial_impact: dict
    operational_impact: dict
    risks: list[str]
    confidence: float
    evidence: list[str]
    kpis: list[dict]                         # {name, value}
    assumptions: list[str]
    next_steps: list[str]
    capabilities: list[str]
    graph: list[list[str]]                   # execution layers
    tasks: list[dict] = field(default_factory=list)
    run_id: str = ""

    def to_dict(self) -> dict:
        return {
            "objective": self.objective, "executive_summary": self.executive_summary,
            "key_findings": self.key_findings, "recommended_actions": self.recommended_actions,
            "financial_impact": self.financial_impact, "operational_impact": self.operational_impact,
            "risks": self.risks, "confidence": round(self.confidence, 1), "evidence": self.evidence,
            "kpis": self.kpis, "assumptions": self.assumptions, "next_steps": self.next_steps,
            "capabilities": self.capabilities, "graph": self.graph, "tasks": self.tasks,
            "run_id": self.run_id,
        }


@dataclass
class PlannerRun:
    id: str
    objective: str
    graph: list[list[str]]
    capabilities: list[str]
    outputs: dict
    recommendation: str
    predicted_outcome: dict
    actual_outcome: Optional[dict]
    ts: str
