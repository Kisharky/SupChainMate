"""
optimize/solvers/cuopt.py — NVIDIA cuOpt adapter. Wraps the existing
``modules.nvidia_api.cuopt_optimize`` (GPU VRP over NIM) behind the solver port.
Returns ``solved=False`` when the key is missing or the service is unreachable,
so the engine transparently falls back to the local solver.
"""

from __future__ import annotations

from optimize.types import (OptimizationResult, ProblemKind, RouteLeg,
                            RoutingProblem)


class CuOptSolver:
    name = "cuopt"

    def supports(self, kind: ProblemKind) -> bool:
        # cuOpt excels at routing here; allocation is routed to the local solver.
        return kind == ProblemKind.ROUTING

    def is_configured(self) -> bool:
        try:
            import config
            return config.get_env("NVIDIA_CUOPT_API_KEY") is not None
        except Exception:  # noqa: BLE001
            return False

    def solve(self, problem) -> OptimizationResult:
        if not isinstance(problem, RoutingProblem):
            return OptimizationResult(problem.kind.value, self.name, False, 0, 0, 0,
                                      detail="cuOpt adapter handles routing only")
        stops = problem.stops
        lats = [s.lat for s in stops]
        lons = [s.lon for s in stops]
        demands = [max(1, int(s.demand)) for s in stops]
        try:
            from modules import nvidia_api
            out = nvidia_api.cuopt_optimize(lats, lons, demands)
        except Exception as exc:  # noqa: BLE001
            return OptimizationResult(ProblemKind.ROUTING.value, self.name, False,
                                      0, 0, 0, detail=f"cuOpt call failed: {exc}")
        if not out or not out.get("success"):
            return OptimizationResult(ProblemKind.ROUTING.value, self.name, False,
                                      0, 0, 0,
                                      detail=(out or {}).get("summary", "cuOpt unavailable"))
        # Map cuOpt's route(s) into an ordered tour + legs.
        order: list[int] = []
        for route in out.get("routes", []):
            order += [i for i in route.get("stops", []) if i not in order]
        if not order:
            order = list(range(len(stops)))
        obj = float(out.get("total_cost_km", 0.0))
        baseline = float(out.get("naive_cost_km", obj))
        legs = []
        seq = order + ([order[0]] if problem.round_trip else [])
        # distances recomputed locally for display consistency
        from optimize.solvers.local import _haversine
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            legs.append(RouteLeg(a, b, stops[a].name, stops[b].name,
                                 _haversine(stops[a].lat, stops[a].lon, stops[b].lat, stops[b].lon)))
        imp = (baseline - obj) / baseline * 100 if baseline > 0 else 0.0
        return OptimizationResult(
            kind=ProblemKind.ROUTING.value, solver=self.name, solved=True,
            objective=obj, baseline=baseline, improvement_pct=max(0.0, imp),
            order=order, legs=legs, detail="NVIDIA cuOpt capacitated VRP")
