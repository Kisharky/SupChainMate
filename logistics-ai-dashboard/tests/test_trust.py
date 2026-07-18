"""
Trust-layer tests: recommendation builders, confidence bounds, the
approve/reject/modify workflow, persistence dedupe, and the audit trail.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import decisions, store, trust


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "trust.db"))


@pytest.fixture()
def profile():
    return decisions.DemandProfile(
        avg_daily_demand=100.0, std_daily_demand=20.0,
        avg_lead_time_days=7.0, std_lead_time_days=2.0,
        annual_demand=36500.0, horizon_forecast=700.0, horizon_days=7)


@pytest.fixture()
def outputs(profile):
    return decisions.run_decision_engine(profile, service_level=0.95)


# ── Builders ──────────────────────────────────────────────────────────────────

def test_decision_engine_recommendation(profile, outputs):
    recs = trust.from_decision_engine(profile, outputs, history_days=365, service_level=0.95)
    assert len(recs) == 1
    r = recs[0]
    assert 20 <= r.confidence <= 95 and r.confidence_basis
    assert len(r.drivers) >= 4
    assert any("Z = " in d.evidence for d in r.drivers)
    assert r.impact.cost_savings_yr == outputs.savings_vs_current
    assert r.impact.stockout_risk_pct == pytest.approx(5.0)
    assert r.impact.service_level_pct == pytest.approx(95.0)
    # short history must reduce confidence
    short = trust.from_decision_engine(profile, outputs, history_days=30, service_level=0.95)[0]
    assert short.confidence < r.confidence


def test_sku_builder_only_flags_urgent():
    plan = pd.DataFrame({
        "SKU": ["A", "B"], "ABC": ["A", "C"], "Avg Daily": [10.0, 2.0],
        "Svc Level": ["95%", "87%"], "Safety Stock": [30, 5],
        "Reorder Point": [100, 20], "EOQ": [200, 40],
        "Order Every (d)": [20.0, 20.0], "Est. Savings/yr ($)": [500.0, 50.0],
        "Current Stock": [10.0, 500.0],
        "Status": ["🔴 ORDER NOW", "🟢 OK"],
    })
    recs = trust.from_sku_engine(plan, avg_lead_time_days=7.0)
    assert len(recs) == 1 and "Reorder A" in recs[0].title
    assert recs[0].impact.stockout_risk_pct > 50  # 1 day of cover vs 7-day LT


def test_carrier_builder_needs_material_gap():
    score = pd.DataFrame({
        "Carrier": ["Good", "Bad"], "Shipments": [5000, 3000],
        "On-Time %": [96.0, 80.0], "Late": [200, 600],
        "Avg Delay (days)": [2.0, 5.0], "Avg ML Risk %": [10.0, 10.0],
        "Grade": ["A", "D"],
    })
    recs = trust.from_carrier_scorecard(score, None)
    assert len(recs) == 1 and "Shift 25%" in recs[0].title
    # sub-material gap → no recommendation
    score2 = score.copy()
    score2["On-Time %"] = [95.0, 93.5]
    assert trust.from_carrier_scorecard(score2, None) == []


def test_audit_builder_threshold():
    audit = {"kpis": {"flagged_value": 5000.0, "flagged_count": 40,
                      "total_spend": 100000.0, "audited_charges": 9000,
                      "outlier_overcharge": 3000.0, "duplicate_value": 500.0,
                      "late_premium_value": 1500.0, "retender_opportunity": 0.0}}
    recs = trust.from_cost_audit(audit)
    assert len(recs) == 1 and recs[0].impact.cost_savings_yr == 5000.0
    audit["kpis"]["flagged_value"] = 50.0
    assert trust.from_cost_audit(audit) == []


# ── Workflow + persistence + audit ────────────────────────────────────────────

def test_workflow_and_audit(profile, outputs):
    recs = trust.from_decision_engine(profile, outputs, 365, 0.95)
    created = trust.sync_recommendations(recs)
    assert created == 1
    # dedupe: same key must not create twice
    assert trust.sync_recommendations(recs) == 0

    pending = trust.pending()
    assert len(pending) == 1 and pending[0]["status"] == "PENDING"
    key = pending[0]["rec_key"]

    assert trust.decide(key, "MODIFIED", note="approve at half qty")
    assert trust.pending() == []
    hist = trust.history()
    assert hist[0]["status"] == "MODIFIED" and hist[0]["note"] == "approve at half qty"

    kpis = trust.summary_kpis()
    assert kpis["pending"] == 0 and kpis["approved"] == 1
    assert kpis["approved_savings"] == pytest.approx(outputs.savings_vs_current)

    log = store.load_audit_log()
    events = [e["event"] for e in log]
    assert events[0] == "recommendation_modified"       # newest first
    assert "recommendation_created" in events
    assert all(e["ts"] for e in log)


def test_invalid_decision_rejected():
    with pytest.raises(ValueError):
        trust.decide("abc", "PENDING")
    with pytest.raises(ValueError):
        trust.decide("abc", "MAYBE")


def test_generate_all_composes(profile, outputs):
    ctx = {"demand_profile": profile, "decision_outputs": outputs,
           "history_days": 365, "service_level": 0.95,
           "sku_plan": None, "avg_lead_time": 7.0,
           "scorecard": None, "audit": None}
    recs = trust.generate_all(ctx)
    assert len(recs) == 1 and recs[0].category == "INVENTORY POLICY"
    # keys are stable across regeneration
    assert recs[0].key == trust.generate_all(ctx)[0].key
