"""
Module test suite — run from logistics-ai-dashboard/ with:  python -m pytest tests/ -q

Covers the deterministic core: control tower, cost audit, health check,
tender toolkit, alerts, persistence, agent routing, and store connectors
(connectors are exercised against mocked HTTP responses — no network).
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import (agent, alerts, carbon, connect, control_tower, cost_audit,
                     doc_intel, health_check, ingestion, retail, store, tender)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def shipments():
    """Small synthetic shipment board with known properties."""
    n = 400
    rng = np.random.default_rng(7)
    order = pd.date_range("2025-01-01", periods=n, freq="6h")
    promised = order + pd.Timedelta(days=7)
    late_mask = rng.random(n) < 0.2
    delivered = promised + pd.to_timedelta(np.where(late_mask, 3, -1), unit="D")
    df = pd.DataFrame({
        "order_id": [f"ord{i:05d}" for i in range(n)],
        "order_purchase_timestamp": order,
        "order_estimated_delivery_date": promised,
        "order_delivered_customer_date": delivered,
        "order_status": "delivered",
        "carrier": rng.choice(["Alpha", "Beta"], n, p=[0.6, 0.4]),
        "status": "Delivered",
    })
    costs = np.where(df["carrier"] == "Alpha", 10.0, 14.0) * rng.uniform(0.9, 1.1, n)
    costs[:4] *= 5  # guaranteed outliers
    df["freight_cost"] = costs.round(2)
    return control_tower.prepare_shipments(df, delay_model=None)


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "test.db"))
    return store


# ── Control tower ─────────────────────────────────────────────────────────────

def test_shipment_health_classification(shipments):
    assert set(shipments["health"]) <= {
        "ON TRACK", "AT RISK", "LATE", "DELIVERED ON TIME", "DELIVERED LATE", "CANCELLED"}
    late_share = (shipments["health"] == "DELIVERED LATE").mean()
    assert 0.1 < late_share < 0.3  # built with ~20% late


def test_shipment_kpis(shipments):
    k = control_tower.shipment_kpis(shipments)
    assert k["total"] == len(shipments)
    assert 70 < k["on_time_pct"] < 90
    assert k["late"] == (shipments["health"] == "DELIVERED LATE").sum()


def test_carrier_scorecard(shipments):
    score = control_tower.carrier_scorecard(shipments)
    assert set(score["Carrier"]) == {"Alpha", "Beta"}
    assert score["Shipments"].sum() == len(shipments)
    assert all(g in "ABCD—" for g in score["Grade"])


def test_scorecard_none_without_carrier(shipments):
    assert control_tower.carrier_scorecard(shipments.drop(columns=["carrier"])) is None


# ── Cost audit ────────────────────────────────────────────────────────────────

def test_cost_audit_flags_outliers(shipments):
    audit = cost_audit.run_audit(shipments)
    assert audit["kpis"]["outlier_overcharge"] > 0
    assert (audit["flagged"]["reason"].str.contains("OUTLIER")).any()
    assert audit["kpis"]["total_spend"] == pytest.approx(
        shipments["freight_cost"].sum(), rel=1e-6)


def test_cost_audit_none_without_costs(shipments):
    assert cost_audit.run_audit(shipments.drop(columns=["freight_cost"])) is None


# ── Health check ──────────────────────────────────────────────────────────────

def test_health_check_scores(shipments):
    k = control_tower.shipment_kpis(shipments)
    audit = cost_audit.run_audit(shipments)
    outputs = SimpleNamespace(total_optimized_cost=10000, savings_vs_current=500)
    hc = health_check.run_health_check(shipments, k, audit, outputs, delay_risk=10.0)
    assert 0 <= hc["score"] <= 100
    assert hc["grade"] in "ABCDF"
    assert hc["difot"] is not None
    names = {d["dimension"] for d in hc["dimensions"]}
    assert {"Delivery performance", "Cost discipline", "Data quality"} <= names
    assert "HEALTH CHECK" in health_check.health_report(hc)


def test_health_check_empty():
    hc = health_check.run_health_check()
    assert hc["score"] == 0.0 and hc["dimensions"] == []


# ── Tender ────────────────────────────────────────────────────────────────────

def test_tender_pack(shipments):
    pack = tender.build_tender_pack(shipments)
    assert pack["stats"]["total_shipments"] == len(shipments)
    assert "REQUEST FOR PROPOSAL" in pack["rfp_text"]
    assert len(pack["carriers"]) == 2


def test_rate_shift(shipments):
    audit = cost_audit.run_audit(shipments)
    sim = tender.simulate_rate_shift(audit["by_carrier"], "Beta", "Alpha", 50)
    assert sim["cost_delta"] < 0  # Beta is dearer, shifting to Alpha saves
    assert tender.simulate_rate_shift(audit["by_carrier"], "Alpha", "Alpha", 50) is None


# ── Alerts ────────────────────────────────────────────────────────────────────

def test_retail_digest_counts():
    prods = [retail.product_dict("Low Stock", 20, 14, 25, "Medium", 1),
             retail.product_dict("Plenty", 5, 7, 8, "Low", 900)]
    rows = [retail.tracker_row(p) for p in prods]
    digest, n = alerts.build_retail_digest(prods, rows)
    assert n == 1 and "Low Stock" in digest


def test_email_unconfigured(monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.setattr(alerts, "_env", lambda name: None)
    ok, msg = alerts.send_email("a@b.com", "s", "b")
    assert not ok and "SMTP not configured" in msg


# ── Persistence ───────────────────────────────────────────────────────────────

def test_store_roundtrip(tmp_db):
    prods = [{"name": "X", "units_per_week": 5}]
    assert tmp_db.save_retail_products(prods)
    assert tmp_db.load_retail_products() == prods
    assert tmp_db.save_setting("k", {"a": 1})
    assert tmp_db.load_setting("k") == {"a": 1}
    assert tmp_db.load_setting("missing", "dflt") == "dflt"


def test_kpi_snapshots(tmp_db):
    tmp_db.save_kpi_snapshot({"health_score": 80})
    tmp_db.save_kpi_snapshot({"health_score": 85})
    snaps = tmp_db.load_kpi_snapshots()
    assert [s["health_score"] for s in snaps] == [80, 85]  # oldest first
    assert all("ts" in s for s in snaps)


# ── Ingestion platform detection ──────────────────────────────────────────────

def test_detect_store_platform():
    shopify = pd.DataFrame(columns=["Name", "Created at", "Lineitem quantity"])
    woo = pd.DataFrame(columns=["order_id", "order_date", "product_name", "qty"])
    other = pd.DataFrame(columns=["date", "quantity"])
    assert ingestion.detect_store_platform(shopify) == "Shopify"
    assert ingestion.detect_store_platform(woo) == "WooCommerce"
    assert ingestion.detect_store_platform(other) is None


# ── Agent offline routing ─────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected", [
    ("draft an sla email to Alpha", "draft_carrier_email"),
    ("list at-risk shipments", "get_at_risk_shipments"),
    ("generate the reorder plan", "generate_reorder_plan"),
    ("audit our invoices", "freight_cost_audit"),
    ("run a supply chain health check", "supply_chain_health_check"),
    ("prepare an rfp", "generate_tender_pack"),
    ("summary of exceptions", "exception_summary"),
])
def test_agent_offline_routing(shipments, query, expected):
    ctx = {"shipments": shipments,
           "scorecard": control_tower.carrier_scorecard(shipments),
           "kpis": control_tower.shipment_kpis(shipments),
           "metrics": {},
           "decision_outputs": SimpleNamespace(
               eoq=100, reorder_point=50, safety_stock=20,
               total_optimized_cost=1000, savings_vs_current=100),
           "exec_plan": pd.DataFrame({"Action": ["x"]}),
           "delay_risk": 10.0}
    result = agent.run_agent(query, ctx, client=False)
    assert result["actions"] == [expected]
    assert result["engine"] == "offline"


# ── Carbon lens ───────────────────────────────────────────────────────────────

def test_carbon_mode_factors_ordering():
    d = 100.0
    assert (carbon.shipment_co2_kg(d, 20, "sea")
            < carbon.shipment_co2_kg(d, 20, "rail")
            < carbon.shipment_co2_kg(d, 20, "road")
            < carbon.shipment_co2_kg(d, 20, "air"))
    # unknown mode falls back to road
    assert carbon.shipment_co2_kg(d, 20, "hoverboard") == carbon.shipment_co2_kg(d, 20, "road")


def test_carrier_emissions(shipments):
    df = shipments.copy()
    df["transport_mode"] = np.where(df["carrier"] == "Alpha", "rail", "air")
    out = carbon.carrier_emissions(df, avg_distance_km=200, weight_kg=20)
    assert list(out["Carrier"]) == ["Alpha", "Beta"]  # rail sorts greener than air
    assert out["Shipments"].sum() == len(df)
    notes = carbon.carbon_insights(out)
    assert any("more CO2e" in n for n in notes)


def test_zone_emissions_and_route_savings():
    cent = pd.DataFrame({"cluster": [0, 1], "customers": [100, 50],
                         "avg_dist_km": [120.0, 300.0]})
    zones = carbon.zone_emissions(cent, weight_kg=20)
    assert len(zones) == 2 and (zones["Total tCO2e"] > 0).all()
    assert carbon.network_avg_distance_km(cent) == pytest.approx(180.0)
    assert carbon.route_savings_co2(1000) == pytest.approx(0.85)


# ── Document intelligence ─────────────────────────────────────────────────────

def test_doc_intel_sample_roundtrip(shipments, monkeypatch):
    monkeypatch.setattr(doc_intel.groq_ai, "is_available", lambda: False)
    text = doc_intel.sample_invoice(shipments, inflate=False)
    fields, engine = doc_intel.extract_fields(
        text, shipments["carrier"].dropna().unique().tolist())
    assert engine == "offline"
    assert fields["invoice_number"] and fields["total_amount"] > 0
    assert fields["carrier"] in {"Alpha", "Beta"}
    result = doc_intel.reconcile(fields, shipments)
    assert result["verdict"] == "OK TO PAY"
    assert len(result["matched"]) == 3


def test_doc_intel_flags_inflated_invoice(shipments, monkeypatch):
    monkeypatch.setattr(doc_intel.groq_ai, "is_available", lambda: False)
    text = doc_intel.sample_invoice(shipments, inflate=True)
    fields, _ = doc_intel.extract_fields(
        text, shipments["carrier"].dropna().unique().tolist())
    result = doc_intel.reconcile(fields, shipments)
    assert result["verdict"] == "REVIEW — RATE MISMATCH"
    assert any("tolerance" in f for f in result["findings"])


def test_doc_intel_unknown_carrier(shipments, monkeypatch):
    monkeypatch.setattr(doc_intel.groq_ai, "is_available", lambda: False)
    text = "Ghost Logistics\nInvoice Number: INV-1\nTOTAL DUE: $500.00\n"
    fields, _ = doc_intel.extract_fields(text, ["Alpha", "Beta"])
    fields["carrier"] = "Ghost Logistics"
    result = doc_intel.reconcile(fields, shipments)
    assert result["verdict"] == "REVIEW — UNKNOWN CARRIER"


def test_doc_intel_pdf_text_extraction(tmp_path):
    txt, msg = doc_intel.extract_text(b"Invoice Number: INV-9\nTOTAL: $10.00", "inv.txt")
    assert txt and "INV-9" in txt


# ── Store connectors (mocked HTTP) ────────────────────────────────────────────

class _FakeResp:
    def __init__(self, payload, status=200, headers=None):
        self._payload, self.status_code, self.headers = payload, status, headers or {}
    def json(self):
        return self._payload
    def raise_for_status(self):
        if self.status_code >= 400:
            raise connect.requests.exceptions.HTTPError(str(self.status_code))


def test_shopify_fetch_success():
    page = {"orders": [
        {"created_at": "2025-06-01T10:00:00-04:00", "line_items": [{"quantity": 2}, {"quantity": 1}]},
        {"created_at": "2025-06-02T11:00:00-04:00", "line_items": [{"quantity": 4}]},
    ]}
    with patch.object(connect.requests, "get", return_value=_FakeResp(page)):
        df, msg = connect.fetch_shopify_orders("https://mystore.myshopify.com/", "shpat_x")
    assert df is not None and len(df) == 2
    assert list(df["quantity"]) == [3, 4]
    assert "Imported 2" in msg


def test_shopify_bad_token():
    with patch.object(connect.requests, "get", return_value=_FakeResp({}, status=401)):
        df, msg = connect.fetch_shopify_orders("mystore.myshopify.com", "bad")
    assert df is None and "401" in msg


def test_woocommerce_fetch_success():
    page = [{"date_created": "2025-06-01T10:00:00", "line_items": [{"quantity": 5}]}]
    with patch.object(connect.requests, "get", return_value=_FakeResp(page)):
        df, msg = connect.fetch_woocommerce_orders("myshop.com", "ck_x", "cs_y")
    assert df is not None and int(df["quantity"].iloc[0]) == 5


def test_connect_missing_credentials():
    df, _ = connect.fetch_shopify_orders("", "")
    assert df is None
    df, _ = connect.fetch_woocommerce_orders("site.com", "", "")
    assert df is None
