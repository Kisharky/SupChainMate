"""
Intelligence-layer tests: agent memory, RAG knowledge base, event-driven
automation, Executive Copilot tools, and the ERP/REST connectors (mocked).
"""

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import agent, connect, events, knowledge, store, trust


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "intel.db"))


class _FakeResp:
    def __init__(self, payload, status=200, headers=None):
        self._payload, self.status_code, self.headers = payload, status, headers or {}
    def json(self):
        return self._payload
    def raise_for_status(self):
        if self.status_code >= 400:
            raise connect.requests.exceptions.HTTPError(str(self.status_code))


# ── Agent memory ──────────────────────────────────────────────────────────────

def test_memory_roundtrip_and_before_latest():
    store.save_agent_run("wf", "logistics", 80.0, {"on_time_pct": 94.0})
    store.save_agent_run("wf", "logistics", 82.0, {"on_time_pct": 92.5})
    latest = store.last_agent_runs()
    assert latest["logistics"]["outputs"]["on_time_pct"] == 92.5
    prev = store.last_agent_runs(before_latest=True)
    assert prev["logistics"]["outputs"]["on_time_pct"] == 94.0


def test_executive_reports_memory_deltas():
    from modules.agents.domain import _memory_deltas
    memory = {"logistics": {"outputs": {"on_time_pct": 94.0, "late": 100}}}
    upstream = {"logistics": {"on_time_pct": 92.5, "late": 100}}
    deltas = _memory_deltas(memory, upstream)
    assert deltas == ["logistics.on_time_pct: 94.0 → 92.5"]


# ── Knowledge base (RAG) ──────────────────────────────────────────────────────

def test_kb_retrieval_and_extractive_answer(monkeypatch):
    monkeypatch.setattr(knowledge.groq_ai, "is_available", lambda: False)
    store.add_document("procurement_policy.txt",
                       "Supplier lead times must not exceed 21 days.\n\n"
                       "All purchase orders above $10,000 require director approval.\n\n"
                       "Freight invoices are payable within 14 days of receipt.")
    store.add_document("warehouse_sop.txt",
                       "Forklift operators must complete safety training annually.")
    hits = knowledge.retrieve("what is the maximum supplier lead time?")
    assert hits and hits[0]["doc"] == "procurement_policy.txt"
    assert "21 days" in hits[0]["text"]
    res = knowledge.answer("what is the maximum supplier lead time?")
    assert res["engine"] == "extractive" and "21 days" in res["answer"]
    assert res["passages"]


def test_kb_no_match():
    res = knowledge.answer("what colour is the moon?")
    assert res["passages"] == [] and "Nothing in the knowledge base" in res["answer"]
    assert knowledge.kb_stats()["documents"] == 0


def test_kb_chunking():
    chunks = knowledge.chunk_text("para one\n\n" + "x" * 1200 + "\n\npara three", "d")
    assert len(chunks) >= 2 and all(c["doc"] == "d" for c in chunks)


# ── Event-driven automation ───────────────────────────────────────────────────

def _event_ctx(**over):
    score = pd.DataFrame({"Carrier": ["Good", "Bad"], "Shipments": [500, 300],
                          "On-Time %": [96.0, 78.0], "Late": [10, 80],
                          "Avg Delay (days)": [1.0, 6.0], "Avg ML Risk %": [5.0, 9.0],
                          "Grade": ["A", "D"]})
    plan = pd.DataFrame({"SKU": ["A"], "Status": ["🔴 ORDER NOW"]})
    daily = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=100, freq="D"),
                          "y": 100.0})
    fc = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=107, freq="D"),
                       "yhat": 140.0})
    ctx = {"scorecard": score, "sku_plan": plan, "daily_df": daily,
           "forecast_df": fc, "days": 7, "kpis": {"at_risk": 30}}
    ctx.update(over)
    return ctx


def test_event_detectors():
    fired = events.detect(_event_ctx())
    names = {e["name"] for e in fired}
    assert names == {"supplier_delay_detected", "inventory_below_threshold",
                     "demand_spike", "shipment_risk_surge"}
    calm = events.detect(_event_ctx(
        scorecard=None, sku_plan=None, kpis={"at_risk": 0},
        forecast_df=pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=107, freq="D"),
                                  "yhat": 100.0})))
    assert calm == []


def test_event_triggers_workflow_once_per_workflow():
    class _StubRun:
        workflow, results, recommendations_created = "wf", [], 0

    class _StubOrch:
        def __init__(self):
            self.calls = []
        def run_workflow(self, name, ctx):
            self.calls.append(name)
            r = _StubRun(); r.workflow = name
            return r

    orch = _StubOrch()
    fired = events.detect(_event_ctx())
    results = events.run_triggered(orch, fired, _event_ctx())
    # 4 events but only 2 distinct workflows → each runs once
    assert sorted(orch.calls) == ["logistics_review", "planning_chain"]
    assert len(results) == 2
    ev_log = [e["event"] for e in store.load_audit_log()]
    assert ev_log.count("event_detected") == 4
    assert ev_log.count("event_triggered_workflow") == 2


# ── Executive Copilot tools ───────────────────────────────────────────────────

def test_pending_decisions_tool():
    txt, arts = agent._TOOL_FUNCS["get_pending_decisions"]({})
    assert "clear" in txt
    trust.sync_recommendations([trust.Recommendation(
        source="Planner", category="TEST", title="Do the thing", action="do it",
        confidence=80.0, impact=trust.Impact(cost_savings_yr=1234.0))])
    txt, arts = agent._TOOL_FUNCS["get_pending_decisions"]({})
    assert "1 recommendation" in txt and "$1,234" in txt and len(arts) == 1


def test_business_deltas_tool():
    txt, _ = agent._TOOL_FUNCS["business_deltas"]({})
    assert "No agent runs" in txt
    store.save_agent_run("wf", "inventory", 80, {"eoq": 200.0})
    txt, _ = agent._TOOL_FUNCS["business_deltas"]({})
    assert "second run is needed" in txt
    store.save_agent_run("wf", "inventory", 82, {"eoq": 250.0})
    txt, arts = agent._TOOL_FUNCS["business_deltas"]({})
    assert "inventory.eoq" in arts[0]["data"]


def test_copilot_routing_new_tools(monkeypatch):
    monkeypatch.setattr(knowledge.groq_ai, "is_available", lambda: False)
    store.add_document("policy.txt", "Orders above $10,000 require director approval.")
    ctx = {"metrics": {}}
    r = agent.run_agent("What actions should I approve today?", ctx, client=False)
    assert r["actions"] == ["get_pending_decisions"]
    r = agent.run_agent("What changed since the last run?", ctx, client=False)
    assert r["actions"] == ["business_deltas"]
    r = agent.run_agent("What does our policy say about approvals?", ctx, client=False)
    assert r["actions"] == ["ask_knowledge_base"]
    assert "director approval" in r["reply"]


def test_workers_still_cover_all_tools():
    tools_in_workers = {t for w in agent.WORKERS.values() for t in w["tools"]}
    schema_tools = {t["function"]["name"] for t in agent.TOOLS_SCHEMA}
    assert tools_in_workers == schema_tools
    assert "Executive" in agent.WORKERS


# ── ERP / REST connectors (mocked) ────────────────────────────────────────────

def test_erpnext_fetch_success():
    page = {"data": [{"transaction_date": "2025-06-01", "total_qty": 5},
                     {"transaction_date": "2025-06-02", "total_qty": 3}]}
    with patch.object(connect.requests, "get", return_value=_FakeResp(page)):
        df, msg = connect.fetch_erpnext_orders("erp.acme.com", "key", "secret")
    assert df is not None and len(df) == 2 and "ERPNext" in msg


def test_erpnext_bad_credentials():
    with patch.object(connect.requests, "get", return_value=_FakeResp({}, status=403)):
        df, msg = connect.fetch_erpnext_orders("erp.acme.com", "k", "s")
    assert df is None and "403" in msg


def test_generic_rest_fetch_with_path():
    payload = {"data": {"orders": [
        {"created": "2025-06-01", "units": 4},
        {"created": "2025-06-02", "units": 2},
        {"note": "no date here"}]}}
    with patch.object(connect.requests, "get", return_value=_FakeResp(payload)):
        df, msg = connect.fetch_rest_orders(
            "https://api.acme.com/orders", "data.orders", "created", "units")
    assert df is not None and len(df) == 2
    assert float(df["quantity"].sum()) == 6.0
    with patch.object(connect.requests, "get", return_value=_FakeResp(payload)):
        df, msg = connect.fetch_rest_orders(
            "https://api.acme.com/orders", "data.missing", "created")
    assert df is None and "not found" in msg
