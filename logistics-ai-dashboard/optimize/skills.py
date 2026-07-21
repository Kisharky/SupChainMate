"""
optimize/skills.py — the agent-facing "optimization skills". Domain agents call
these instead of touching a solver directly, so the optimization layer sits
*beneath* the agents (agents reason; the skill solves). Each skill takes plain
business inputs and returns a solver-agnostic OptimizationResult.
"""

from __future__ import annotations

from typing import Iterable

from optimize.engine import OPT
from optimize.types import OptimizationResult, Stop


def optimize_delivery_route(stops: Iterable[dict], depot: int = 0,
                            round_trip: bool = True) -> OptimizationResult:
    """Best visiting order over hubs/stops. `stops` = [{name, lat, lon, demand?}]."""
    parsed = [Stop(name=str(s.get("name", f"Stop {i}")),
                   lat=float(s["lat"]), lon=float(s["lon"]),
                   demand=float(s.get("demand", 0) or 0))
              for i, s in enumerate(stops)]
    if len(parsed) < 2:
        return OptimizationResult("routing", "none", False, 0, 0, 0,
                                  detail="need at least 2 stops")
    return OPT.optimize_route(parsed, depot=depot, round_trip=round_trip)


def optimize_supply_allocation(sources: list[str], sinks: list[str],
                               cost: list[list[float]], supply: list[float],
                               demand: list[float]) -> OptimizationResult:
    """Least-cost assignment of supply across sinks (transportation problem)."""
    return OPT.optimize_allocation(sources, sinks, cost, supply, demand)
