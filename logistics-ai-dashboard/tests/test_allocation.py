"""
Carrier allocation + dispute lifecycle tests.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import allocation, disputes, store


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "alloc.db"))


@pytest.fixture()
def scorecard():
    return pd.DataFrame({
        "Carrier": ["Cheap", "Premium", "Green"],
        "Shipments": [6000, 3000, 1000],
        "On-Time %": [88.0, 97.0, 92.0],
        "Late": [700, 90, 80],
        "Avg Delay (days)": [4.0, 1.0, 2.0],
        "Avg ML Risk %": [12.0, 5.0, 8.0],
        "Grade": ["C", "A", "B"],
        "Avg Cost/Shipment ($)": [7.0, 15.0, 10.0],
    })


@pytest.fixture()
def carbon_table():
    return pd.DataFrame({"Carrier": ["Cheap", "Premium", "Green"],
                         "kg CO2e/shipment": [2.0, 2.0, 0.5]})


def test_profiles_and_shares(scorecard, carbon_table):
    prof = allocation.build_carrier_profiles(scorecard, carbon_table)
    assert prof["Current Share %"].sum() == pytest.approx(100.0, abs=0.5)
    assert "kg CO2e/shipment" in prof.columns
    assert allocation.build_carrier_profiles(scorecard.head(1)) is None


def test_weight_extremes_pick_the_right_winner(scorecard, carbon_table):
    prof = allocation.build_carrier_profiles(scorecard, carbon_table)
    cost_only = allocation.allocation_scores(
        prof, {"cost": 1, "service": 0, "emissions": 0, "reliability": 0})
    assert cost_only.iloc[0]["Carrier"] == "Cheap"
    service_only = allocation.allocation_scores(
        prof, {"cost": 0, "service": 1, "emissions": 0, "reliability": 0})
    assert service_only.iloc[0]["Carrier"] == "Premium"
    green_only = allocation.allocation_scores(
        prof, {"cost": 0, "service": 0, "emissions": 1, "reliability": 0})
    assert green_only.iloc[0]["Carrier"] == "Green"


def test_concentration_cap_and_share_sum(scorecard, carbon_table):
    prof = allocation.build_carrier_profiles(scorecard, carbon_table)
    scored = allocation.allocation_scores(prof)
    assert (scored["Recommended Share %"] <= allocation.MAX_SHARE_PCT + 0.6).all()
    assert scored["Recommended Share %"].sum() == pytest.approx(100.0, abs=1.0)


def test_impact_and_recommendation(scorecard, carbon_table):
    prof = allocation.build_carrier_profiles(scorecard, carbon_table)
    scored = allocation.allocation_scores(prof)
    imp = allocation.allocation_impact(scored, 10000)
    assert imp["total_shift_pts"] > 0
    assert "on_time_current" in imp and "cost_current" in imp
    rec = allocation.build_recommendation(scored, imp, allocation.DEFAULT_WEIGHTS)
    assert rec is not None and rec.category == "CARRIER ALLOCATION"
    assert 20 <= rec.confidence <= 95 and len(rec.drivers) >= 3
    # no-shift case → no proposal
    flat = scored.copy()
    flat["Recommended Share %"] = flat["Current Share %"]
    flat["Shift (pts)"] = 0.0
    assert allocation.build_recommendation(
        flat, allocation.allocation_impact(flat, 10000),
        allocation.DEFAULT_WEIGHTS) is None


# ── Disputes ──────────────────────────────────────────────────────────────────

def _flagged():
    return pd.DataFrame({
        "shipment_id": ["S1", "S2", "S3"],
        "carrier": ["Cheap", "Cheap", "Premium"],
        "reason": ["OUTLIER — above carrier IQR cap"] * 2 + ["POTENTIAL DUPLICATE charge"],
        "freight_cost": [50.0, 40.0, 12.0],
        "overcharge_est": [35.0, 25.0, 12.0],
    })


def test_dispute_lifecycle_and_kpis():
    assert disputes.open_disputes_from_audit(_flagged()) == 3
    # dedupe on re-raise
    assert disputes.open_disputes_from_audit(_flagged()) == 0

    rows = disputes.list_disputes()
    assert len(rows) == 3 and rows[0]["amount"] == 35.0  # sorted by amount
    key = rows[0]["dispute_key"]

    assert disputes.transition(key, "SENT")
    assert disputes.transition(key, "RESOLVED", recovered=30.0)
    with pytest.raises(ValueError):     # RESOLVED is terminal
        disputes.transition(key, "SENT")
    with pytest.raises(ValueError):     # OPEN cannot jump to RESOLVED
        disputes.transition(disputes.list_disputes(status="OPEN")[0]["dispute_key"],
                            "RESOLVED")

    k = disputes.dispute_kpis()
    assert k["total"] == 3 and k["recovered"] == 30.0
    assert k["recovery_rate_pct"] == pytest.approx(30.0 / 35.0 * 100, rel=1e-3)

    events = [e["event"] for e in store.load_audit_log()]
    assert "disputes_opened" in events and "dispute_resolved" in events
