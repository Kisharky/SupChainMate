"""Tests for Freight Operations (vetting, matching, quoting, triage)."""
from api import freight


def test_overview_shape():
    o = freight.overview()
    assert o["source"] == "representative"
    assert o["carriers"] and o["loads"] and o["triage"] and o["roadmap"]
    for c in o["carriers"]:
        assert {"mc_number", "authority_status", "insurance_status", "risk_score", "recommendation"} <= c.keys()


def test_carriers_sorted_by_risk_desc():
    scores = [c["risk_score"] for c in freight.overview()["carriers"]]
    assert scores == sorted(scores, reverse=True)


def test_high_risk_carrier_fails_checks():
    o = freight.overview()
    worst = o["carriers"][0]
    assert worst["risk_severity"] == "high"
    det = freight.carrier_detail(worst["id"])
    assert det["ok"] and any(not c["ok"] for c in det["checks"])
    assert "Block" in det["recommendation"] or worst["risk_score"] >= 60


def test_clean_carrier_passes():
    o = freight.overview()
    clean = next(c for c in o["carriers"] if c["flag_count"] == 0 and c["insurance_status"] == "valid")
    det = freight.carrier_detail(clean["id"])
    assert det["ok"] and det["flags"] == []


def test_unknown_carrier():
    assert freight.carrier_detail("MC-0")["ok"] is False


def test_load_matches_ranked_and_capped():
    for l in freight.overview()["loads"]:
        fits = [m["fit_score"] for m in l["matches"]]
        assert fits == sorted(fits, reverse=True)
        assert len(l["matches"]) <= 3


def test_quote_breakdown_and_margin():
    q = freight.quote("A", "B", "Reefer", 400)
    assert q["all_in_rate"] > q["carrier_cost"] > 0
    assert q["margin_usd"] == round(q["all_in_rate"] - q["carrier_cost"], 2)
    assert len(q["breakdown"]) == 3
    # Unknown equipment falls back to Dry Van.
    assert freight.quote("A", "B", "Spaceship", 400)["equipment"] == "Dry Van"


def test_triage_classifies_all_emails():
    triage = freight.overview()["triage"]
    types = {e["type"] for e in triage}
    assert {"load_tender", "quote_request", "check_call", "invoice", "claim"} <= types
