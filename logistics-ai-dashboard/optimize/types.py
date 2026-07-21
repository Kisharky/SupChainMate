"""
optimize/types.py — framework-free data contracts for the optimization layer.

Mirrors the AI layer's philosophy: business/agent code describes a *problem*
(routing, allocation) and never picks a solver. The engine resolves the problem
kind to a concrete solver (NVIDIA cuOpt, or a local fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProblemKind(str, Enum):
    ROUTING = "routing"        # vehicle routing / tour over stops
    ALLOCATION = "allocation"  # transportation / assignment (supply → demand)


@dataclass(frozen=True)
class Stop:
    name: str
    lat: float
    lon: float
    demand: float = 0.0


@dataclass
class RoutingProblem:
    """Optimise the visiting order of `stops` from a depot."""
    stops: list[Stop]
    depot: int = 0
    round_trip: bool = True
    vehicles: int = 1
    kind: ProblemKind = field(default=ProblemKind.ROUTING, init=False)


@dataclass
class AllocationProblem:
    """Assign supply across sinks to minimise total cost (transportation LP)."""
    sources: list[str]
    sinks: list[str]
    cost: list[list[float]]      # cost[i][j] = ship one unit source i → sink j
    supply: list[float]
    demand: list[float]
    kind: ProblemKind = field(default=ProblemKind.ALLOCATION, init=False)


@dataclass
class RouteLeg:
    from_idx: int
    to_idx: int
    from_name: str
    to_name: str
    distance_km: float


@dataclass
class Assignment:
    source: str
    sink: str
    units: float
    cost: float


@dataclass
class OptimizationResult:
    kind: str
    solver: str                       # which solver produced this
    solved: bool
    objective: float                  # total distance (km) or total cost
    baseline: float                   # naive/as-given objective for comparison
    improvement_pct: float            # % better than baseline
    order: list[int] = field(default_factory=list)          # routing visiting order
    legs: list[RouteLeg] = field(default_factory=list)      # routing legs
    assignments: list[Assignment] = field(default_factory=list)  # allocation
    fell_back: bool = False
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "solver": self.solver, "solved": self.solved,
            "objective": round(self.objective, 1), "baseline": round(self.baseline, 1),
            "improvement_pct": round(self.improvement_pct, 1),
            "order": self.order, "fell_back": self.fell_back, "detail": self.detail,
            "legs": [{"from": l.from_name, "to": l.to_name,
                      "from_idx": l.from_idx, "to_idx": l.to_idx,
                      "distance_km": round(l.distance_km, 1)} for l in self.legs],
            "assignments": [{"source": a.source, "sink": a.sink,
                             "units": round(a.units, 1), "cost": round(a.cost, 1)}
                            for a in self.assignments],
        }


@dataclass(frozen=True)
class SolverSpec:
    """Maps a problem kind to a primary solver and a fallback (pluggable)."""
    kind: ProblemKind
    solver: str
    fallback: Optional[str] = None
