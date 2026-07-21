"""
optimize/engine.py — the pluggable optimization router. Resolves a problem kind
to its primary solver, and falls back down the declared chain when the primary
isn't configured or can't solve. Structurally identical to ai/router.py, but for
optimization: agents/business code call OPT.solve(problem) and never name a solver.
"""

from __future__ import annotations

from typing import Optional

from optimize.registry import SolverRegistry
from optimize.types import (AllocationProblem, OptimizationResult, ProblemKind,
                           RoutingProblem, Stop)


class OptimizationEngine:
    def __init__(self, solvers: Optional[dict] = None,
                 registry: Optional[SolverRegistry] = None) -> None:
        if solvers is None:
            from optimize.solvers.cuopt import CuOptSolver
            from optimize.solvers.local import LocalSolver
            solvers = {"cuopt": CuOptSolver(), "local": LocalSolver()}
        self.solvers = solvers
        self.registry = registry or SolverRegistry()

    def solve(self, problem) -> OptimizationResult:
        spec = self.registry.resolve(problem.kind)
        if spec is None:
            return OptimizationResult(problem.kind.value, "none", False, 0, 0, 0,
                                      detail="no solver registered")
        chain = [spec.solver] + ([spec.fallback] if spec.fallback else [])
        last: Optional[OptimizationResult] = None
        for i, name in enumerate(chain):
            solver = self.solvers.get(name)
            if solver is None or not solver.supports(problem.kind):
                continue
            if not solver.is_configured():
                continue
            result = solver.solve(problem)
            if result.solved:
                result.fell_back = i > 0
                return result
            last = result
        # Nothing in the chain solved — force the local solver as a last resort.
        local = self.solvers.get("local")
        if local is not None and local.supports(problem.kind):
            r = local.solve(problem)
            r.fell_back = True
            r.detail = (last.detail + " → " if last else "") + r.detail
            return r
        return last or OptimizationResult(problem.kind.value, "none", False, 0, 0, 0,
                                          detail="no solver could handle the problem")

    def status(self) -> dict:
        return {
            "plan": self.registry.plan(),
            "solvers": {name: {"configured": s.is_configured()}
                        for name, s in self.solvers.items()},
        }


class _OptFacade:
    """Process-wide singleton, lazily built (mirrors ai.AI)."""
    def __init__(self) -> None:
        self._engine: Optional[OptimizationEngine] = None

    def configure(self, engine: OptimizationEngine) -> None:
        self._engine = engine

    @property
    def engine(self) -> OptimizationEngine:
        if self._engine is None:
            self._engine = OptimizationEngine()
        return self._engine

    def solve(self, problem) -> OptimizationResult:
        return self.engine.solve(problem)

    def optimize_route(self, stops: list[Stop], depot: int = 0,
                       round_trip: bool = True) -> OptimizationResult:
        return self.solve(RoutingProblem(stops=stops, depot=depot, round_trip=round_trip))

    def optimize_allocation(self, sources, sinks, cost, supply, demand) -> OptimizationResult:
        return self.solve(AllocationProblem(sources=sources, sinks=sinks, cost=cost,
                                            supply=supply, demand=demand))

    def status(self) -> dict:
        return self.engine.status()


OPT = _OptFacade()
