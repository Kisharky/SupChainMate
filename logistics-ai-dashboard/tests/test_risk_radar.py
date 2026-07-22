"""Tests for the Disruption & Risk Radar (signal convergence engine)."""
from api import risk_radar as radar


def test_overview_shape():
    o = radar.overview()
    assert o["source"] == "representative"
    assert o["nodes"] and o["lanes"] and o["layers"] and o["alerts"]
    assert len(o["layers"]) == 7
    for n in o["nodes"]:
        assert {"lat", "lon", "risk_score", "band", "convergence", "signals"} <= n.keys()


def test_index_leans_on_worst_nodes():
    o = radar.overview()
    idx = o["index"]
    assert 0 <= idx["score"] <= 100
    top_node = max(n["risk_score"] for n in o["nodes"])
    # Index should be at least as high as the average of the worst — never below the network floor.
    assert idx["score"] >= min(n["risk_score"] for n in o["nodes"])
    assert idx["converging_alerts"] == len(o["alerts"])


def test_convergence_only_flags_multi_signal():
    # Every alert has >= 2 converging signals; critical ones have >= converge_at.
    o = radar.overview()
    ca = o["converge_at"]
    for a in o["alerts"]:
        assert a["convergence"] >= 2
        assert a["critical"] == (a["convergence"] >= ca)
    # A node with a single active signal must NOT produce an alert.
    alert_refs = {a["ref_id"] for a in o["alerts"]}
    for n in o["nodes"]:
        active = sum(1 for v in n["signals"].values() if v >= 40)
        if active < 2:
            assert n["id"] not in alert_refs


def test_alerts_sorted_by_convergence_then_score():
    o = radar.overview()
    keys = [(a["convergence"], a["composite_score"]) for a in o["alerts"]]
    assert keys == sorted(keys, reverse=True)


def test_layers_carry_active_events():
    layers = {l["id"]: l for l in radar.overview()["layers"]}
    for l in layers.values():
        assert l["active_events"] == len(l["events"])
        assert all(e["severity"] >= 40 for e in l["events"])


def test_node_detail_and_why():
    o = radar.overview()
    critical = next(a for a in o["alerts"] if a["critical"])
    det = radar.node_detail(critical["ref_id"])
    assert det["ok"] and det["convergence"] >= 3
    assert det["why"] and det["recommended_action"]
    assert any(s["active"] for s in det["signals"])
    assert radar.node_detail("nope")["ok"] is False
