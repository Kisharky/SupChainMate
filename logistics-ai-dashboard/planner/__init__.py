"""
planner/ — the executive decision orchestrator (an AI Decision OS layer).

Sits ABOVE the existing architecture (AI Router, Optimization Engine, Commercial
Intelligence, Decision Workspace, domain services) and coordinates them into a
single executive decision. Contains no business logic: it discovers registered
capabilities, builds an execution graph, runs the existing systems, and merges
their outputs.

    from planner import PLANNER
    decision = PLANNER.plan("Reduce inventory holding cost by 10%")
"""

from planner.planner import PLANNER, Planner
from planner.registry import CapabilityRegistry
from planner.schemas import Capability, Decision, TaskResult

__all__ = ["PLANNER", "Planner", "CapabilityRegistry", "Capability", "Decision", "TaskResult"]
