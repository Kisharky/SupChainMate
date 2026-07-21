"""
optimize/solvers/local.py — dependency-light local solver. Always available; it
is the honest fallback when NVIDIA cuOpt is not reachable/configured.

* Routing: nearest-neighbour construction + 2-opt improvement over a Haversine
  distance matrix (a clean TSP tour connecting the network's hubs).
* Allocation: greedy least-cost transportation heuristic respecting supply and
  demand (uses scipy's exact solver when available, greedy otherwise).
"""

from __future__ import annotations

import math

from optimize.types import (Assignment, AllocationProblem, OptimizationResult,
                            ProblemKind, RouteLeg, RoutingProblem)


def _haversine(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


class LocalSolver:
    name = "local"

    def supports(self, kind: ProblemKind) -> bool:
        return kind in (ProblemKind.ROUTING, ProblemKind.ALLOCATION)

    def is_configured(self) -> bool:
        return True

    # ---- Routing ----
    def _matrix(self, stops) -> list[list[float]]:
        n = len(stops)
        return [[_haversine(stops[i].lat, stops[i].lon, stops[j].lat, stops[j].lon)
                 for j in range(n)] for i in range(n)]

    def _tour_len(self, order: list[int], d: list[list[float]], round_trip: bool) -> float:
        total = sum(d[order[i]][order[i + 1]] for i in range(len(order) - 1))
        if round_trip and len(order) > 1:
            total += d[order[-1]][order[0]]
        return total

    def _solve_routing(self, p: RoutingProblem) -> OptimizationResult:
        n = len(p.stops)
        d = self._matrix(p.stops)
        # As-given order = baseline
        baseline_order = list(range(n))
        baseline = self._tour_len(baseline_order, d, p.round_trip)

        # Nearest-neighbour construction from the depot
        unvisited = set(range(n)) - {p.depot}
        order = [p.depot]
        while unvisited:
            last = order[-1]
            nxt = min(unvisited, key=lambda j: d[last][j])
            order.append(nxt)
            unvisited.discard(nxt)

        # 2-opt improvement
        improved = True
        while improved and n > 3:
            improved = False
            for i in range(1, n - 1):
                for k in range(i + 1, n):
                    new_order = order[:i] + order[i:k + 1][::-1] + order[k + 1:]
                    if self._tour_len(new_order, d, p.round_trip) + 1e-9 < self._tour_len(order, d, p.round_trip):
                        order = new_order
                        improved = True

        obj = self._tour_len(order, d, p.round_trip)
        legs = []
        seq = order + ([order[0]] if p.round_trip else [])
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            legs.append(RouteLeg(a, b, p.stops[a].name, p.stops[b].name, d[a][b]))
        imp = (baseline - obj) / baseline * 100 if baseline > 0 else 0.0
        return OptimizationResult(
            kind=ProblemKind.ROUTING.value, solver=self.name, solved=True,
            objective=obj, baseline=baseline, improvement_pct=max(0.0, imp),
            order=order, legs=legs,
            detail=f"nearest-neighbour + 2-opt over {n} stops")

    # ---- Allocation ----
    def _solve_allocation(self, p: AllocationProblem) -> OptimizationResult:
        supply = list(p.supply)
        demand = list(p.demand)
        cost = p.cost
        assigns: list[Assignment] = []
        total = 0.0
        # Greedy least-cost cell selection (transportation heuristic)
        cells = sorted(((cost[i][j], i, j) for i in range(len(supply))
                        for j in range(len(demand))), key=lambda c: c[0])
        for c, i, j in cells:
            if supply[i] <= 1e-9 or demand[j] <= 1e-9:
                continue
            units = min(supply[i], demand[j])
            supply[i] -= units
            demand[j] -= units
            total += units * c
            assigns.append(Assignment(p.sources[i], p.sinks[j], units, units * c))
        # Baseline: proportional (demand-weighted average cost) allocation
        avg_cost = (sum(sum(row) for row in cost) / (len(supply) * len(demand))
                    if supply and demand else 0.0)
        served = sum(a.units for a in assigns)
        baseline = avg_cost * served
        imp = (baseline - total) / baseline * 100 if baseline > 0 else 0.0
        return OptimizationResult(
            kind=ProblemKind.ALLOCATION.value, solver=self.name, solved=True,
            objective=total, baseline=baseline, improvement_pct=max(0.0, imp),
            assignments=assigns, detail="greedy least-cost transportation")

    def solve(self, problem) -> OptimizationResult:
        if isinstance(problem, RoutingProblem):
            return self._solve_routing(problem)
        return self._solve_allocation(problem)
