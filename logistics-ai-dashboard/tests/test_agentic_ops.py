"""Tests for Agentic Ops Workflows (detect → diagnose → decide → execute → report)."""
from api import agentic_ops as ao


def test_shape_and_summary():
    d = ao.workflows()
    assert d["source"] == "representative"
    assert d["loop"] == ["detect", "diagnose", "decide", "execute", "report"]
    assert len(d["workflows"]) == 3
    s = d["summary"]
    assert s["workflows_run"] == 3
    assert s["total_saved"] == sum(w["saved_usd"] for w in d["workflows"])
    assert s["auto_resolved"] == sum(1 for w in d["workflows"] if w["auto"])


def test_every_workflow_has_the_full_loop():
    for w in ao.workflows()["workflows"]:
        assert [s["phase"] for s in w["steps"]] == ["detect", "diagnose", "decide", "execute", "report"]
        assert w["one_liner"] and w["guardrails"]
        assert w["status"] in ("resolved", "awaiting_approval", "monitoring")
        for st in w["steps"]:
            assert st["actor"] in ("agent", "human")


def test_signature_workflows_present():
    kinds = {w["kind"] for w in ao.workflows()["workflows"]}
    assert {"otif_drift", "reroute", "capacity_lock"} <= kinds


def test_resolved_workflow_has_all_steps_done():
    resolved = next(w for w in ao.workflows()["workflows"] if w["auto"])
    assert all(s["done"] for s in resolved["steps"])
    awaiting = next(w for w in ao.workflows()["workflows"] if w["status"] == "awaiting_approval")
    assert not all(s["done"] for s in awaiting["steps"])   # execute/report pend on the human
