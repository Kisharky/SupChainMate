"""Tests for the Planner orchestration core (framework-free, no domain deps)."""

from planner.aggregator import Aggregator
from planner.executor import Executor
from planner.graph import ExecutionGraph
from planner.planner import Planner
from planner.registry import CapabilityRegistry
from planner.schemas import Capability


def _cap(name, deps, impact=None, kw=None, prio=1):
    return Capability(
        name=name, description=f"{name} capability", required_inputs=[], outputs=[name],
        dependencies=deps, confidence=0.8, priority=prio, keywords=kw or [name],
        handler=lambda ctx, n=name, i=impact: {"summary": f"{n} ran", "metrics": {"v": 1},
                                               "impact_usd": i, "confidence": 0.8})


def _registry():
    r = CapabilityRegistry()
    r.register(_cap("forecast", [], kw=["forecast", "demand"]))
    r.register(_cap("inventory", ["forecast"], impact=5000, kw=["inventory", "stock", "holding"]))
    r.register(_cap("profit", ["inventory"], impact=9000, kw=["profit", "margin", "cost"]))
    r.register(_cap("leakage", [], impact=3000, kw=["leakage", "revenue"]))
    return r


def test_select_pulls_transitive_dependencies():
    r = _registry()
    caps = {c.name for c in r.select("reduce inventory holding cost")}
    # 'inventory' and 'profit' match; deps forecast + inventory pulled in
    assert {"forecast", "inventory", "profit"} <= caps


def test_graph_orders_dependencies_into_layers():
    r = _registry()
    caps = r.select("reduce inventory holding cost")
    layers = ExecutionGraph(caps).build()
    flat = [n for layer in layers for n in layer]
    assert flat.index("forecast") < flat.index("inventory") < flat.index("profit")
    # forecast has no deps → first layer
    assert "forecast" in layers[0]


def test_graph_detects_cycles():
    r = CapabilityRegistry()
    r.register(_cap("a", ["b"]))
    r.register(_cap("b", ["a"]))
    try:
        ExecutionGraph(r.all()).build()
        assert False, "expected cycle error"
    except ValueError:
        pass


def test_executor_runs_all_and_propagates_context():
    r = _registry()
    caps = r.select("reduce inventory holding cost")
    layers = ExecutionGraph(caps).build()
    results = Executor(r).run(layers, {})
    assert all(res.ok for res in results.values())
    assert set(results) == {c.name for c in caps}


def test_aggregate_merges_into_one_decision():
    r = _registry()
    caps = r.select("reduce inventory holding cost")
    layers = ExecutionGraph(caps).build()
    results = Executor(r).run(layers, {})
    d = Aggregator().aggregate("reduce inventory holding cost", results, layers, caps)
    assert d.financial_impact["identified_usd"] == 14000  # 5000 + 9000
    assert d.confidence == 80
    assert d.executive_summary
    assert d.graph == layers


def test_planner_records_to_memory_and_returns_run_id():
    p = Planner(registry=_registry())
    d = p.plan("reduce inventory holding cost")
    assert d.run_id
    assert d.capabilities
    assert p.history(5)


def test_new_capability_is_discovered_without_core_changes():
    r = _registry()
    r.register(_cap("carbon_optimizer", [], impact=1200, kw=["carbon", "emissions", "sustainability"]))
    caps = {c.name for c in r.select("cut carbon emissions")}
    assert "carbon_optimizer" in caps
