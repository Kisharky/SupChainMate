"""
modules/agents/base.py
Agent contracts: context scoping, results, and the execution template.

Design (SOLID):
  - Single responsibility: one agent = one business domain; the base class
    owns only the cross-cutting mechanics (scoping, timing, logging, audit).
  - Open/closed: new agents subclass BaseAgent and register with the
    orchestrator — nothing else changes.
  - Interface segregation: agents declare `required_context` and
    `depends_on`; the ScopedContext enforces that an agent can only read
    what it declared — "access only to relevant tools and data" is a
    runtime guarantee, not a convention.
  - Dependency inversion: agents depend on the AgentContext abstraction,
    never on Streamlit or the database. Persistence of recommendations and
    audit events happens in the orchestrator via the trust/store layers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import config
from modules.trust import Impact, Recommendation

_log = config.get_logger(__name__)


class ContextAccessError(KeyError):
    """An agent touched context it did not declare — a scoping violation."""


class ScopedContext:
    """Read-only view of the shared context, limited to declared keys."""

    def __init__(self, data: dict[str, Any], allowed: set[str],
                 upstream: dict[str, dict[str, Any]],
                 upstream_confidence: Optional[dict[str, float]] = None):
        self._data = data
        self._allowed = allowed
        self.upstream = upstream  # outputs of declared dependency agents only
        self.upstream_confidence = upstream_confidence or {}

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._allowed:
            raise ContextAccessError(
                f"Context key '{key}' was not declared in required_context "
                f"(allowed: {sorted(self._allowed)})")
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._allowed and key in self._data


@dataclass
class AgentResult:
    """The uniform output every agent produces."""
    agent: str
    objective: str
    findings: list[str] = field(default_factory=list)   # explainable reasoning steps
    confidence: float = 50.0                            # 0-100
    confidence_basis: str = ""
    impact: Impact = field(default_factory=Impact)
    recommendations: list[Recommendation] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)  # passed downstream
    requires_approval: bool = False
    duration_ms: int = 0
    error: Optional[str] = None
    ai_narrative: Optional[str] = None   # LLM narrative when AI reasoning is enabled

    @property
    def ok(self) -> bool:
        return self.error is None


class BaseAgent(ABC):
    """Template-method base: scoping, timing, failure containment, and — when
    enabled — AI reasoning through the router (never a model directly)."""

    name: str = "agent"
    objective: str = ""
    required_context: list[str] = []
    depends_on: list[str] = []     # upstream agent names whose outputs it may read
    reasoning_capability: str = "reasoning.operations"   # router picks the model

    @abstractmethod
    def run(self, ctx: ScopedContext) -> AgentResult:
        """Domain logic. Must not perform I/O beyond the provided context."""

    def execute(self, shared: dict[str, Any],
                upstream: dict[str, dict[str, Any]],
                upstream_confidence: Optional[dict[str, float]] = None,
                ai_enabled: bool = False) -> AgentResult:
        """Scope the context, run with timing, contain failures, and
        optionally add an AI narrative via the router."""
        scoped = ScopedContext(
            shared, set(self.required_context),
            {k: v for k, v in upstream.items() if k in self.depends_on},
            {k: v for k, v in (upstream_confidence or {}).items()
             if k in self.depends_on})
        t0 = time.perf_counter()
        try:
            result = self.run(scoped)
        except ContextAccessError:
            raise  # scoping violations are programming errors — surface them
        except Exception as e:  # domain failures are contained, not fatal
            _log.warning("Agent %s failed: %s", self.name, e)
            result = AgentResult(agent=self.name, objective=self.objective,
                                 findings=[f"Agent failed: {e}"],
                                 confidence=20.0, confidence_basis="agent error",
                                 error=str(e))
        if ai_enabled and result.ok:
            result.ai_narrative = self._ai_narrative(result, upstream)
        result.duration_ms = round((time.perf_counter() - t0) * 1000)
        result.requires_approval = result.requires_approval or bool(result.recommendations)
        result.confidence = float(min(max(result.confidence, 0.0), 100.0))
        _log.info("Agent %s: confidence %.0f, %d finding(s), %d rec(s), %d ms",
                  self.name, result.confidence, len(result.findings),
                  len(result.recommendations), result.duration_ms)
        return result

    def _ai_narrative(self, result: "AgentResult",
                      upstream: dict[str, dict[str, Any]]) -> Optional[str]:
        """Ask the router to narrate this agent's deterministic findings. The
        agent never picks a model — it names a capability; the router decides.
        Numbers come from `result`, never from the model."""
        try:
            from ai import AI
            context = {"agent": self.name, "objective": self.objective,
                       "findings": result.findings, "outputs": result.outputs,
                       "confidence": result.confidence,
                       "upstream": {k: upstream.get(k) for k in self.depends_on}}
            resp = AI.ask(self.reasoning_capability, f"{self.name}_review", context)
            return resp.text if resp.ok and resp.text else None
        except Exception as e:  # AI enrichment must never break an agent
            _log.warning("AI narrative for %s failed: %s", self.name, e)
            return None
