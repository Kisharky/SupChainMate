"""Tests for the pluggable optimization layer (optimize/)."""

from optimize import (OPT, OptimizationEngine, optimize_delivery_route,
                     optimize_supply_allocation)
from optimize.registry import SolverRegistry
from optimize.solvers.local import LocalSolver
from optimize.types import ProblemKind


HUBS = [
    {"name": "A", "lat": -23.1, "lon": -47.0}, {"name": "B", "lat": -22.9, "lon": -43.2},
    {"name": "C", "lat": -19.9, "lon": -43.9}, {"name": "D", "lat": -30.0, "lon": -51.2},
    {"name": "E", "lat": -25.4, "lon": -49.3}, {"name": "F", "lat": -15.8, "lon": -47.9},
]


def test_routing_improves_on_baseline():
    r = optimize_delivery_route(HUBS, round_trip=True)
    assert r.solved
    assert r.objective <= r.baseline          # never worse than as-given
    assert r.improvement_pct >= 0
    assert sorted(r.order) == list(range(len(HUBS)))  # visits every stop once
    assert len(r.legs) == len(HUBS)           # round trip closes the loop


def test_routing_needs_two_stops():
    r = optimize_delivery_route(HUBS[:1])
    assert not r.solved


def test_allocation_respects_supply_and_demand():
    r = optimize_supply_allocation(
        sources=["DC-A", "DC-B"], sinks=["S1", "S2", "S3"],
        cost=[[4, 6, 8], [9, 3, 5]], supply=[100, 120], demand=[60, 90, 70])
    assert r.solved
    assert r.objective <= r.baseline
    # every unit of demand is met and no source over-ships
    served = {s: 0.0 for s in ("S1", "S2", "S3")}
    shipped = {s: 0.0 for s in ("DC-A", "DC-B")}
    for a in r.assignments:
        served[a.sink] += a.units
        shipped[a.source] += a.units
    assert served == {"S1": 60, "S2": 90, "S3": 70}
    assert shipped["DC-A"] <= 100 and shipped["DC-B"] <= 120


def test_engine_falls_back_to_local_when_cuopt_absent():
    # cuOpt is unconfigured in the test env, so routing must fall back to local.
    res = optimize_delivery_route(HUBS)
    assert res.solver == "local"
    assert res.fell_back is True

    # An engine with only the local solver still solves via the last-resort path.
    eng = OptimizationEngine(solvers={"local": LocalSolver()}, registry=SolverRegistry())
    from optimize.types import RoutingProblem, Stop
    direct = eng.solve(RoutingProblem(stops=[Stop(h["name"], h["lat"], h["lon"]) for h in HUBS]))
    assert direct.solved and direct.solver == "local"


def test_status_reports_plan():
    st = OPT.status()
    assert st["plan"]["routing"] in ("cuopt", "local")
    assert "local" in st["solvers"]
    assert st["solvers"]["local"]["configured"] is True
