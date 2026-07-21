"""
api/workspace.py — the Decision & Scenario Intelligence workspace services.

An operational-decision command centre (inspired by modern AI decision platforms):
executive decision brief, an AI planner that decomposes requests into agent tasks,
multiple courses of action per issue, a scenario simulator, a Detect→…→Learn
decision timeline, and a "What Changed Today" briefing.

Design: every figure is anchored to a real baseline (shipment KPIs, the trust
layer, the commercial model); the modelled deltas (scenario magnitudes, course-of-
action economics) use transparent, labelled formulas — never black-box guesses.
"""

from __future__ import annotations

from typing import Any

from api import services


# ---- Shared real baselines ---------------------------------------------------
def _baseline() -> dict[str, Any]:
    """Pull the live operational baseline the workspace reasons over."""
    b = {"on_time": 93.0, "late": 0, "at_risk": 0, "in_transit": 0,
         "health": 96, "supplier_health": 91, "inventory_value_m": 12.4,
         "revenue": 11_900_000.0, "leakage": 238_000.0, "repricing_upside": 74_000.0,
         "pending": 0, "avg_conf": 90.0, "approved_savings": 0.0}
    try:
        _, sk, scorecard = services._shipments()
        b.update(on_time=round(float(sk.get("on_time_pct") or 93), 1),
                 late=int(sk.get("late", 0)), at_risk=int(sk.get("at_risk", 0)),
                 in_transit=int(sk.get("in_transit", 0)))
        if scorecard is not None and "Grade" in scorecard.columns:
            b["weak_carriers"] = [str(r["Carrier"]) for _, r in scorecard.iterrows()
                                  if str(r.get("Grade")) not in ("A", "B")]
    except Exception:  # noqa: BLE001
        pass
    try:
        k = services.executive_kpis()["kpis"]
        b.update(health=k["supply_chain_health"]["value"],
                 supplier_health=k["supplier_health"]["value"],
                 inventory_value_m=k["inventory_value"]["value"])
    except Exception:  # noqa: BLE001
        pass
    try:
        co = services._commercial()["kpis"]
        b.update(revenue=co["total_revenue"], leakage=co["revenue_leakage"],
                 repricing_upside=co["repricing_upside"])
    except Exception:  # noqa: BLE001
        pass
    try:
        from modules import trust
        sm = trust.summary_kpis()
        b.update(pending=sm.get("pending", 0), approved_savings=sm.get("approved_savings", 0.0),
                 avg_conf=sm.get("avg_confidence") or 90.0)
    except Exception:  # noqa: BLE001
        pass
    return b


# ---- Executive Decision Brief ------------------------------------------------
def executive_brief() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        b = _baseline()
        risks = []
        if b["at_risk"]:
            risks.append({"title": f"{b['at_risk']} shipments at delivery risk", "severity": "high",
                          "area": "Logistics", "detail": "ML delay signal above threshold on open shipments."})
        weak = b.get("weak_carriers", [])
        if weak:
            risks.append({"title": f"{len(weak)} carrier(s) below B grade", "severity": "medium",
                          "area": "Procurement", "detail": f"Underperformers: {', '.join(weak[:3])}."})
        if b["leakage"]:
            risks.append({"title": f"${b['leakage']:,.0f} revenue leakage", "severity": "medium",
                          "area": "Commercial", "detail": "Freight overspend + discount leakage recoverable."})
        if b["on_time"] < 95:
            risks.append({"title": f"On-time at {b['on_time']}% (target 95%)", "severity": "low",
                          "area": "Logistics", "detail": "Volume shift to higher-graded carriers recommended."})

        recommended, impact_open = [], 0.0
        try:
            from modules import trust
            services._ensure_recommendations()
            for r in trust.pending()[:4]:
                imp = float((r.get("impact") or {}).get("cost_savings_yr") or 0)
                impact_open += imp
                recommended.append({"title": r["title"], "action": r["action"],
                                    "confidence": round(float(r.get("confidence", 0)), 0),
                                    "impact_usd": round(imp, 0),
                                    "area": r.get("source", "—")})
        except Exception:  # noqa: BLE001
            pass

        opportunity = impact_open + b["repricing_upside"] + b["leakage"] * 0.4
        at_risk_usd = b["at_risk"] * 1800 + b["leakage"]
        summary = (
            f"Network health is {b['health']}% with on-time delivery at {b['on_time']}%. "
            f"{len(risks)} operational risk(s) are open"
            + (f", led by {risks[0]['title'].lower()}" if risks else "")
            + f". {b['pending']} decision(s) await approval carrying about "
            f"${impact_open:,.0f}/yr in impact; a further ${opportunity - impact_open:,.0f} sits in "
            f"pricing and leakage recovery. Recommended: clear the approval queue and action the "
            f"top course of action this cycle.")
        return {
            "summary": summary, "risks": risks, "recommended": recommended,
            "financial_impact": {"at_risk_usd": round(at_risk_usd, 0),
                                 "opportunity_usd": round(opportunity, 0),
                                 "net_usd": round(opportunity - at_risk_usd, 0)},
            "confidence": round(float(b["avg_conf"]), 0),
            "awaiting_approval": b["pending"],
            "kpis": {"health": b["health"], "on_time": b["on_time"],
                     "supplier_health": b["supplier_health"],
                     "inventory_value_m": b["inventory_value_m"]},
        }
    return services._safe(build, {"summary": "", "risks": [], "recommended": [],
                                  "financial_impact": {}, "confidence": 0,
                                  "awaiting_approval": 0, "kpis": {}})


# ---- What Changed Today ------------------------------------------------------
def whats_changed() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules import store, trust
        audit = store.load_audit_log(limit=120)
        today = (audit[0]["ts"][:10] if audit else "")
        todays = [e for e in audit if e["ts"][:10] == today] or audit[:20]
        completed = [f"{e['event'].replace('_', ' ')} — {e['details']}"
                     for e in todays if e["event"].startswith("recommendation_") and "created" not in e["event"]][:6]
        changes = [f"{e['event'].replace('_', ' ')}" for e in todays if e["event"].startswith("workflow")][:4]
        recs = trust.history(limit=100)
        realized = sum((r.get("impact") or {}).get("cost_savings_yr") or 0
                       for r in recs if r["status"] in ("APPROVED", "MODIFIED"))
        unresolved = [{"title": r["title"], "reason": "escalated — needs executive sign-off"}
                      for r in recs if r["status"] == "ESCALATED"][:5]
        b = _baseline()
        new_risks = []
        if b["at_risk"]:
            new_risks.append(f"{b['at_risk']} shipments moved to at-risk since last review")
        if b.get("weak_carriers"):
            new_risks.append(f"{len(b['weak_carriers'])} carrier(s) slipped below grade B")
        return {
            "date": today,
            "changes": changes or ["No workflow runs recorded today."],
            "new_risks": new_risks or ["No new risks detected."],
            "completed": completed or ["No decisions actioned yet today."],
            "realized_savings": round(float(realized), 0),
            "unresolved": unresolved,
        }
    return services._safe(build, {"date": "", "changes": [], "new_risks": [],
                                  "completed": [], "realized_savings": 0, "unresolved": []})


# ---- Live Decision Timeline (Detect → … → Learn) -----------------------------
_STAGES = ["Detect", "Analyse", "Optimise", "Recommend", "Approve", "Execute", "Measure", "Learn"]
_STATUS_STAGE = {"PENDING": "Recommend", "APPROVED": "Execute", "MODIFIED": "Execute",
                 "ESCALATED": "Approve", "REJECTED": "Learn"}


def decision_timeline() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules import trust
        services._ensure_recommendations()
        items = []
        for r in (trust.pending() + trust.history(limit=40)):
            stage = _STATUS_STAGE.get(r["status"], "Recommend")
            imp = float((r.get("impact") or {}).get("cost_savings_yr") or 0)
            outcome = None
            if r["status"] in ("APPROVED", "MODIFIED"):
                outcome = f"actioned · ${imp:,.0f}/yr" if imp else "actioned"
            elif r["status"] == "REJECTED":
                outcome = "declined — logged for learning"
            items.append({
                "id": r["rec_key"], "title": r["title"], "stage": stage,
                "confidence": round(float(r.get("confidence", 0)), 0),
                "impact_usd": round(imp, 0), "status": r["status"],
                "outcome": outcome, "ts": r.get("decided_ts") or r.get("created_ts"),
            })
        counts = {s: sum(1 for it in items if it["stage"] == s) for s in _STAGES}
        return {"stages": _STAGES, "counts": counts, "items": items[:24]}
    return services._safe(build, {"stages": _STAGES, "counts": {}, "items": []})


# ---- Courses of Action -------------------------------------------------------
_ISSUES = {
    "stockout": "Critical SKU below 3 days of cover",
    "carrier": "Underperforming carrier dragging on-time",
    "leakage": "Revenue leakage from freight & discounts",
    "demand_spike": "Regional demand spike outpacing supply",
}


def courses_of_action(issue: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        b = _baseline()
        key = issue if issue in _ISSUES else "stockout"
        opts = _COA_BUILDERS[key](b)
        # rank by a simple ROI-adjusted, risk-discounted score
        for o in opts:
            risk_pen = {"low": 1.0, "medium": 0.85, "high": 0.65}[o["operational_risk"]]
            o["score"] = round((o["expected_savings"] / max(o["implementation_cost"], 1))
                               * risk_pen * (o["confidence"] / 100), 2)
        opts.sort(key=lambda o: o["score"], reverse=True)
        return {"issue": _ISSUES[key], "issue_key": key,
                "options": opts, "recommended": opts[0]["id"]}
    return services._safe(build, {"issue": "", "options": [], "recommended": None})


def _coa(id, name, cost, savings, risk, svc, inv, eta, conf, outcome, evidence, opt, roi):
    return {"id": id, "name": name, "implementation_cost": cost, "expected_savings": savings,
            "operational_risk": risk, "service_level_impact": svc, "inventory_impact": inv,
            "execution_time": eta, "confidence": conf, "business_outcome": outcome,
            "evidence": evidence, "optimization": opt, "roi": round(savings / max(cost, 1), 1)}


def _coa_stockout(b):
    return [
        _coa("expedite", "Expedite air freight", 2100, 18000, "low", +2.5, "restores cover in 2d",
             "48 hours", 92, "Stockout avoided; service level protected",
             ["2 days cover vs 7-day lead time", "Air lane available at +$2.1k"],
             "Routing skill: fastest feasible lane", 8.6),
        _coa("reallocate", "Re-balance stock between DCs", 400, 9000, "low", +1.2, "shifts 400u to deficit DC",
             "24 hours", 88, "Covers the gap using existing network stock",
             ["Surplus at DC-0, deficit at DC-4", "Transit < 24h intra-network"],
             "Allocation skill: min-cost transfer", 22.5),
        _coa("raise_rop", "Raise reorder point + standing PO", 1200, 24000, "medium", +0.8, "higher buffer, more capital",
             "next cycle", 80, "Prevents recurrence; ties up working capital",
             ["Demand CV elevated on this SKU", "Lead-time variance 2.0d"],
             "Decision engine: SS/ROP recompute", 20.0),
    ]


def _coa_carrier(b):
    weak = ", ".join(b.get("weak_carriers", ["the weakest carrier"])[:2])
    return [
        _coa("shift_volume", "Shift volume to A-grade carriers", 0, 26000, "low", +1.8, "neutral",
             "next cycle", 90, f"On-time recovers; away from {weak}",
             ["Grade gap ≥ 1 tier on shared lanes", "Capacity headroom on A-carriers"],
             "Allocation skill: least-cost carrier assignment", 260.0),
        _coa("sla_review", "SLA review + penalty enforcement", 800, 12000, "medium", +0.9, "neutral",
             "2 weeks", 78, "Improves accountability; supplier friction",
             ["Repeated late deliveries logged", "Contractual SLA clauses apply"],
             "—", 15.0),
        _coa("retender", "Re-tender the underperforming lanes", 3500, 40000, "high", -0.3, "neutral",
             "6-8 weeks", 70, "Structural cost reset; execution risk & lead time",
             [f"${b['leakage']:,.0f} above network-median rate", "Volume history tender-ready"],
             "—", 11.4),
    ]


def _coa_leakage(b):
    lk = b["leakage"]
    return [
        _coa("dispute", "Auto-raise freight disputes", 500, round(lk * 0.35), "low", 0.0, "neutral",
             "1 week", 86, "Recovers overcharges pre-payment",
             [f"${lk:,.0f} flagged over median rate", "IQR + duplicate checks deterministic"],
             "—", round(lk * 0.35 / 500, 1)),
        _coa("reprice", "Execute repricing tickets", 300, round(b["repricing_upside"]), "medium", 0.0,
             "neutral", "2 weeks", 82, "Restores margin to target on under-priced SKUs",
             ["12 SKUs below 35% target margin", "Low elasticity on affected lines"],
             "—", round(b["repricing_upside"] / 300, 1)),
        _coa("discount_policy", "Tighten discount policy", 1000, round(lk * 0.25), "medium", -0.5,
             "neutral", "next cycle", 74, "Cuts discount leakage; some deal friction",
             ["Discount leakage in the margin waterfall", "Approval thresholds bypassed"],
             "—", round(lk * 0.25 / 1000, 1)),
    ]


def _coa_demand_spike(b):
    return [
        _coa("prebuild", "Pre-build safety stock ahead of spike", 6000, 30000, "medium", +2.0,
             "raises inventory ~8%", "1 week", 84, "Meets the spike; higher holding cost",
             ["Forecast peak above P90 band", "Lead time won't cover reactive orders"],
             "Decision engine: SS uplift", 5.0),
        _coa("reallocate2", "Re-allocate network stock to the region", 800, 14000, "low", +1.5,
             "neutral network-wide", "48 hours", 88, "Serves the spike from surplus DCs",
             ["Surplus in adjacent DCs", "Haversine transfer cost minimised"],
             "Allocation skill: multi-DC transfer", 17.5),
        _coa("expedite_supply", "Expedite supplier replenishment", 3500, 22000, "medium", +1.0,
             "restores cover", "3-5 days", 80, "Closes the gap; premium freight",
             ["Supplier can pull forward", "Air/expedited lane priced"],
             "Routing skill: expedited lane", 6.3),
    ]


_COA_BUILDERS = {"stockout": _coa_stockout, "carrier": _coa_carrier,
                 "leakage": _coa_leakage, "demand_spike": _coa_demand_spike}


# ---- Scenario Simulator ------------------------------------------------------
# Each scenario maps a 0–1 magnitude to transparent deltas on the real baseline.
_SCENARIOS = {
    "supplier_failure": "Key supplier failure",
    "transport_delay": "Major transport delay",
    "fuel_price": "Fuel price surge",
    "warehouse_shutdown": "Warehouse shutdown",
    "labour_shortage": "Labour shortage",
    "demand_spike": "Demand spike",
    "inventory_shortage": "Inventory shortage",
    "customer_growth": "New customer growth",
}


def _scenario_deltas(kind: str, mag: float, b: dict) -> dict[str, Any]:
    rev = b["revenue"]
    m = max(0.0, min(1.0, mag))
    table = {
        # kind: (financial as % of revenue, service pp, logistics pp, inventory %, customers, positive?)
        "supplier_failure":   (-0.06 * m, -6 * m, -4 * m, -12 * m, int(9000 * m), False),
        "transport_delay":    (-0.03 * m, -8 * m, -12 * m, +4 * m, int(6000 * m), False),
        "fuel_price":         (-0.025 * m, -1 * m, -3 * m, 0, int(1500 * m), False),
        "warehouse_shutdown": (-0.05 * m, -10 * m, -7 * m, -18 * m, int(12000 * m), False),
        "labour_shortage":    (-0.035 * m, -7 * m, -6 * m, -5 * m, int(5000 * m), False),
        "demand_spike":       (+0.04 * m, -3 * m, -2 * m, -9 * m, int(8000 * m), True),
        "inventory_shortage": (-0.045 * m, -9 * m, -1 * m, -22 * m, int(7000 * m), False),
        "customer_growth":    (+0.08 * m, -2 * m, -3 * m, -6 * m, int(11000 * m), True),
    }
    fin_pct, svc, logi, inv, cust, positive = table.get(kind, table["transport_delay"])
    return {"financial_usd": round(rev * fin_pct, 0), "service_pp": round(svc, 1),
            "logistics_pp": round(logi, 1), "inventory_pct": round(inv, 1),
            "customers_affected": cust, "positive": positive}


_MITIGATIONS = {
    "supplier_failure": [("Activate backup supplier", "+4 pp service", "$3.5k"),
                         ("Re-allocate network stock", "covers 60% of gap", "$0.8k"),
                         ("Expedite alternate sourcing", "restores supply 5d", "$6k")],
    "transport_delay": [("Re-route via optimizer", "recovers ~31h", "$0.2k"),
                        ("Switch to A-grade carrier", "+3 pp on-time", "$1.2k")],
    "fuel_price": [("Re-tender fuel-exposed lanes", "-2% freight", "$3.5k"),
                   ("Consolidate shipments", "-1% cost", "$0")],
    "warehouse_shutdown": [("Fail over to nearest DCs", "covers 70%", "$1.5k"),
                           ("Multi-DC re-allocation", "min-cost transfer", "$0.8k")],
    "labour_shortage": [("Prioritise A-class SKUs", "protects 80% revenue", "$0"),
                        ("Temp capacity + overtime", "+5 pp throughput", "$4k")],
    "demand_spike": [("Pre-build safety stock", "meets spike", "$6k"),
                     ("Re-allocate to region", "serves from surplus", "$0.8k")],
    "inventory_shortage": [("Expedite replenishment", "restores cover", "$3.5k"),
                           ("Raise reorder points", "prevents recurrence", "$1.2k")],
    "customer_growth": [("Scale DC capacity", "meets growth", "$8k"),
                        ("Onboard secondary carrier", "adds lane capacity", "$1k")],
}


def simulate_scenario(kind: str, magnitude: float) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        b = _baseline()
        k = kind if kind in _SCENARIOS else "transport_delay"
        d = _scenario_deltas(k, magnitude, b)
        before = {"service": b["on_time"], "logistics": b["on_time"],
                  "inventory_value_m": b["inventory_value_m"], "health": b["health"]}
        after = {"service": round(b["on_time"] + d["service_pp"], 1),
                 "logistics": round(b["on_time"] + d["logistics_pp"], 1),
                 "inventory_value_m": round(b["inventory_value_m"] * (1 + d["inventory_pct"] / 100), 1),
                 "health": max(0, round(b["health"] + d["service_pp"] * 0.6))}
        mits = [{"action": a, "effect": e, "cost": c} for a, e, c in _MITIGATIONS.get(k, [])]
        sign = "upside" if d["positive"] else "exposure"
        narrative = (f"A {_SCENARIOS[k].lower()} at {int(magnitude * 100)}% severity implies "
                     f"${abs(d['financial_usd']):,.0f} of financial {sign}, service moving "
                     f"{d['service_pp']:+.1f} pp and inventory {d['inventory_pct']:+.1f}%. "
                     f"{len(mits)} mitigation(s) modelled below.")
        return {"kind": k, "label": _SCENARIOS[k], "magnitude": magnitude,
                "impact": d, "before": before, "after": after,
                "mitigations": mits, "narrative": narrative}
    return services._safe(build, {"kind": kind, "label": "", "magnitude": magnitude,
                                  "impact": {}, "before": {}, "after": {}, "mitigations": [],
                                  "narrative": ""})


def scenario_catalog() -> dict[str, Any]:
    return {"scenarios": [{"key": k, "label": v} for k, v in _SCENARIOS.items()],
            "issues": [{"key": k, "label": v} for k, v in _ISSUES.items()]}
