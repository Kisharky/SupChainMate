"""
optimize/ — a pluggable optimization layer beneath the domain agents.

Agents and business logic describe a *problem* (routing, allocation); the engine
resolves it to a concrete solver — NVIDIA cuOpt when configured, a local
heuristic otherwise — exactly as the ai/ layer resolves capabilities to models.
"""

from optimize.engine import OPT, OptimizationEngine
from optimize.skills import optimize_delivery_route, optimize_supply_allocation
from optimize.types import (AllocationProblem, OptimizationResult, ProblemKind,
                           RoutingProblem, Stop)

__all__ = [
    "OPT", "OptimizationEngine", "optimize_delivery_route",
    "optimize_supply_allocation", "RoutingProblem", "AllocationProblem",
    "OptimizationResult", "ProblemKind", "Stop",
]
