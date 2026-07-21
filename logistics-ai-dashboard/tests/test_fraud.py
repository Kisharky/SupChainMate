"""Tests for Fraud & Anomaly Detection (representative signals)."""
from api import fraud


def test_overview_shape():
    o = fraud.overview()
    assert o["source"] == "representative"
    assert o["alerts"] and o["entities"] and o["checks"]
    for a in o["alerts"]:
        assert {"id", "type", "severity", "entity", "detail", "recommended_action",
                "amount_at_risk", "confidence", "status"} <= a.keys()
        assert a["severity"] in ("high", "medium", "low")


def test_alerts_sorted_high_severity_first():
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    ranks = [sev_rank[a["severity"]] for a in fraud.overview()["alerts"]]
    assert ranks == sorted(ranks)


def test_summary_matches_alerts_and_entities():
    o = fraud.overview()
    s = o["summary"]
    open_alerts = [a for a in o["alerts"] if a["status"] != "resolved"]
    assert s["open_alerts"] == len(open_alerts)
    assert s["high_severity"] == sum(1 for a in open_alerts if a["severity"] == "high")
    assert s["amount_at_risk"] == sum(a["amount_at_risk"] for a in open_alerts)
    assert s["entities_flagged"] == sum(1 for e in o["entities"] if e["risk_score"] >= 60)


def test_entity_tiers():
    for e in fraud.overview()["entities"]:
        assert e["tier"] in ("Critical", "High", "Medium", "Low")
        assert 0 <= e["risk_score"] <= 100
