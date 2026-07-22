"""Tests for Customer 360 — an aggregator over existing modules."""
from api import customers as cu


def test_list_reuses_commercial_accounts():
    from api import commercial_intel as ci
    lst = cu.list_customers()["customers"]
    assert lst
    assert {c["id"] for c in lst} == {a["id"] for a in ci._accounts()}
    for c in lst:
        assert {"industry", "country", "health_score", "risk_band", "revenue"} <= c.keys()


def test_detail_shape_and_reused_figures():
    from api import commercial_intel as ci
    cid = cu.list_customers()["customers"][0]["id"]
    d = cu.detail(cid)
    assert d["ok"]
    # Commercial revenue is the SAME figure the commercial module computes (not recomputed).
    acct = next(a for a in ci._accounts() if a["id"] == cid)
    assert d["commercial"]["revenue"] == acct["revenue"]
    assert len(d["exec_summary"]["bullets"]) == 5
    assert d["exec_summary"]["generated_by"] == "Decision Brain"
    assert len(d["risk"]["dimensions"]) == 6


def test_section_shapes():
    cid = cu.list_customers()["customers"][0]["id"]
    assert cu.orders(cid)["orders"]
    sh = cu.shipments(cid)
    assert sh["history"] and sh["map"]["points"]
    fc = cu.forecast(cid)
    assert fc["historical"] and fc["predicted"] and 0 <= fc["stockout_probability"] <= 100
    assert cu.recommendations(cid)["recommendations"]
    assert cu.timeline(cid)["events"]


def test_brain_is_real_recall():
    cid = cu.list_customers()["customers"][0]["id"]
    b = cu.brain(cid)
    assert "groups" in b and isinstance(b["total"], int)  # real Decision Brain recall


def test_chat_grounds_in_account_figures():
    cid = cu.list_customers()["customers"][0]["id"]
    r = cu.chat(cid, "How profitable is this customer?")
    assert r["ok"] and r["answer"]
    assert "revenue" in r["context"].lower()


def test_unknown_customer_is_safe():
    assert cu.detail("nope")["ok"] is False
    assert cu.orders("nope")["orders"] == []
    assert cu.chat("nope", "hi")["ok"] is True  # graceful
