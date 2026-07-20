"""
Multi-agent layer tests: context scoping enforcement, each domain agent on
synthetic data, orchestrator pipelines (context passing, dependency
validation, audit logging, Decision Center routing).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import control_tower, cost_audit, decisions, store, tracking
from modules.agents import build_default_orchestrator
from modules.agents.base import AgentResult, BaseAgent, ContextAccessError, ScopedContext
from modules.agents.domain import (DemandForecastAgent, ExecutiveAgent,
                                   InventoryAgent, LogisticsAgent,
                                   ProcurementAgent, SupplierRiskAgent,
                                   SustainabilityAgent, WarehouseAgent)
from modules.agents.orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "agents.db"))


@pytest.fixture(scope="module")
def shared_context():
    """A full synthetic context resembling a loaded session."""
    rng = np.random.default_rng(6)
    n = 800
    order = pd.date_range("2025-01-01", periods=n, freq="4h")
    promised = order + pd.Timedelta(days=7)
    late = rng.random(n) < 0.15
    delivered = promised + pd.to_timedelta(np.where(late, 3, -1), unit="D")
    ships_raw = pd.DataFrame({
        "order_id": [f"o{i:05d}" for i in range(n)],
        "order_purchase_timestamp": order,
        "order_estimated_delivery_date": promised,
        "order_delivered_customer_date": delivered,
        "order_status": "delivered",
        "carrier": rng.choice(["Alpha", "Beta", "Gamma"], n, p=[0.55, 0.3, 0.15]),
        "status": "Delivered",
        "transport_mode": "road",
    })
    ships_raw["freight_cost"] = np.where(ships_raw["carrier"] == "Alpha", 9.0, 13.0) \
        * rng.uniform(0.9, 1.1, n)
    ships = control_tower.prepare_shipments(ships_raw, None)

    t = np.arange(240)
    daily = pd.DataFrame({
        "ds": pd.date_range("2025-01-01", periods=240, freq="D"),
        "y": np.clip(100 + 0.2 * t + 15 * np.sin(2 * np.pi * t / 7)
                     + rng.normal(0, 5, 240), 0, None)})
    fc = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=247, freq="D"),
                       "yhat": 110.0})

    profile = decisions.DemandProfile(100.0, 20.0, 7.0, 2.0, 36500.0, 700.0, 7)
    outputs = decisions.run_decision_engine(profile, service_level=0.95)
    sku_plan = pd.DataFrame({
        "SKU": ["W1"], "ABC": ["A"], "Avg Daily": [12.0], "Svc Level": ["95%"],
        "Safety Stock": [40], "Reorder Point": [120], "EOQ": [300],
        "Order Every (d)": [25.0], "Est. Savings/yr ($)": [800.0],
        "Current Stock": [10.0], "Status": ["🔴 ORDER NOW"]})

    return {
        "daily_df": daily, "forecast_df": fc, "days": 7,
        "demand_profile": profile, "decision_outputs": outputs,
        "sku_plan": sku_plan, "service_level": 0.95,
        "history_days": 240, "avg_lead_time": 7.0,
        "shipments": ships, "kpis": control_tower.shipment_kpis(ships),
        "scorecard": control_tower.carrier_scorecard(ships),
        "audit": cost_audit.run_audit(ships),
        "centroid_stats": pd.DataFrame({"cluster": [0, 1], "customers": [400, 400],
                                        "avg_dist_km": [80.0, 150.0],
                                        "max_dist_km": [200.0, 350.0],
                                        "efficiency_score": [75.0, 55.0]}),
        "n_clusters": 2, "shipment_weight_kg": 20.0,
        "health": {"score": 82.0, "grade": "B"},
    }


# ── Scoping enforcement ───────────────────────────────────────────────────────

def test_scoped_context_blocks_undeclared_access():
    class NosyAgent(BaseAgent):
        name, objective = "nosy", "test"
        required_context = ["allowed_key"]

        def run(self, ctx: ScopedContext) -> AgentResult:
            ctx.get("secret_key")  # not declared → must raise
            return AgentResult(agent=self.name, objective=self.objective)

    with pytest.raises(ContextAccessError):
        NosyAgent().execute({"allowed_key": 1, "secret_key": 2}, {})


def test_upstream_limited_to_dependencies(shared_context):
    proc = ProcurementAgent()
    result = proc.execute(shared_context,
                          {"inventory": {"urgent_skus": ["X"]},
                           "sustainability": {"total_tco2e": 99}})
    assert result.ok
    # Sustainability's outputs were not declared, so procurement never saw them
    assert "sustainability" not in [f for f in result.findings if "99" in f]


# ── Each agent individually ───────────────────────────────────────────────────

@pytest.mark.parametrize("agent_cls", [
    DemandForecastAgent, LogisticsAgent, SupplierRiskAgent,
    WarehouseAgent, SustainabilityAgent,
])
def test_independent_agents_run(agent_cls, shared_context):
    result = agent_cls().execute(shared_context, {})
    assert result.ok, result.error
    assert result.findings and 20 <= result.confidence <= 95
    assert result.confidence_basis
    assert isinstance(result.outputs, dict)


def test_inventory_agent_uses_demand_context(shared_context):
    result = InventoryAgent().execute(
        shared_context, {"demand_forecast": {"growth_pct": 25.0}},
        {"demand_forecast": 80.0})
    assert result.ok
    assert any("growth" in f for f in result.findings)
    assert result.recommendations  # policy + urgent SKU
    assert result.requires_approval
    assert result.outputs["urgent_skus"]


def test_supplier_risk_flags_concentration(shared_context):
    result = SupplierRiskAgent().execute(shared_context, {})
    assert result.outputs["concentration_pct"] > 40
    assert any(r.category == "CONCENTRATION RISK" for r in result.recommendations)


def test_executive_confidence_bounded_by_weakest(shared_context):
    ex = ExecutiveAgent()
    result = ex.execute(shared_context,
                        {"demand_forecast": {"horizon_demand": 700, "growth_pct": 1.0,
                                             "champion": "X", "champion_mape": 30,
                                             "horizon_days": 7},
                         "inventory": {"reorder_point": 100, "eoq": 200, "urgent_skus": []}},
                        {"demand_forecast": 71.0, "inventory": 88.0})
    assert result.confidence == pytest.approx(71.0)
    assert "weakest" in result.confidence_basis
    assert result.outputs["brief"]


# ── Orchestrator ──────────────────────────────────────────────────────────────

def test_workflow_validation():
    orch = Orchestrator().register(InventoryAgent()).register(DemandForecastAgent())
    with pytest.raises(ValueError):
        orch.define_workflow("bad", ["ghost_agent"])
    with pytest.raises(ValueError):  # inventory depends on demand_forecast (later)
        orch.define_workflow("wrong_order", ["inventory", "demand_forecast"])
    orch.define_workflow("ok", ["demand_forecast", "inventory"])
    assert "ok" in orch.workflows


def test_full_workflow_passes_context_and_audits(shared_context):
    orch = build_default_orchestrator()
    run = orch.run_workflow("full_control_tower", shared_context)
    assert [r.agent for r in run.results] == [
        "demand_forecast", "inventory", "procurement", "logistics",
        "supplier_risk", "warehouse", "sustainability", "knowledge", "executive"]
    assert all(r.ok for r in run.results)
    # Executive coordinates: its dependency list covers every specialist
    from modules.agents.domain import ExecutiveAgent
    assert set(ExecutiveAgent.depends_on) == {
        "demand_forecast", "inventory", "procurement", "logistics",
        "supplier_risk", "warehouse", "sustainability", "knowledge"}
    # context passing: executive brief mentions upstream agents' numbers
    brief = run.results[-1].outputs["brief"]
    assert "Demand:" in brief and "Inventory:" in brief and "Risk:" in brief
    # recommendations routed to the Decision Center
    assert run.recommendations_created > 0
    from modules import trust
    assert len(trust.pending()) == run.recommendations_created
    # audit trail captured the run
    events = [e["event"] for e in store.load_audit_log()]
    assert "workflow_started" in events and "workflow_completed" in events
    assert events.count("agent_run") == 9


def test_knowledge_agent_reports_kb_coverage(shared_context):
    from modules.agents.domain import KnowledgeAgent
    from modules import store
    # empty KB → low confidence, honest finding
    empty = KnowledgeAgent().execute(shared_context, {})
    assert empty.ok and empty.outputs["policies_found"] == 0
    assert "empty" in empty.findings[0].lower()
    # with a matching policy → coverage found, cited
    store.add_document("sla_policy.txt",
                       "Carrier SLA on-time performance must exceed 95%. Carriers "
                       "grading below this face review.")
    covered = KnowledgeAgent().execute(shared_context, {})
    assert covered.outputs["policies_found"] >= 1
    assert covered.confidence > empty.confidence


def test_planning_chain_inter_agent_flow(shared_context):
    orch = build_default_orchestrator()
    run = orch.run_workflow("planning_chain", shared_context,
                            sync_to_decision_center=False)
    proc = next(r for r in run.results if r.agent == "procurement")
    # procurement saw inventory's urgent SKUs (which saw demand's forecast)
    assert proc.outputs["po_lines"] >= 1
    assert any("urgent reorder" in f for f in proc.findings)
