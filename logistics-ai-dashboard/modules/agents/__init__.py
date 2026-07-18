"""
modules/agents — the multi-agent orchestration layer.

Eight single-responsibility domain agents wrap the existing deterministic
engines (forecasting, inventory, tender, control tower, carbon, network)
and an Orchestrator runs them as multi-step workflows with shared context.
Recommendations produced by agents flow into the Decision Center (trust
layer) for human approval; every agent run is audit-logged.
"""

from modules.agents.base import AgentResult, BaseAgent, ScopedContext  # noqa: F401
from modules.agents.orchestrator import Orchestrator, build_default_orchestrator  # noqa: F401
