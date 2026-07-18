"""
modules/health_check.py
Supply Chain Health Check — a scored assessment of the loaded network.

Six dimensions, each 0-100, weighted into an overall score and grade:
  1. Delivery performance  — on-time % vs promised dates (DIFOT-style)
  2. Risk posture          — ML delay risk + share of open shipments at risk
  3. Cost discipline       — flagged audit value as a share of freight spend
  4. Inventory discipline  — unrealised savings vs total inventory cost
  5. Network efficiency    — cluster efficiency scores (Haversine metrics)
  6. Data quality          — completeness of dates / carrier / cost fields

Every score is computed from the loaded data; dimensions without data are
excluded from the weighting rather than guessed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_WEIGHTS = {
    "Delivery performance": 0.28,
    "Risk posture": 0.18,
    "Cost discipline": 0.18,
    "Inventory discipline": 0.12,
    "Network efficiency": 0.10,
    "Data quality": 0.14,
}


def _scale(value: float, worst: float, best: float) -> float:
    """Linear 0-100 scale between a worst and best bound (handles inverted)."""
    if worst == best:
        return 50.0
    t = (value - worst) / (best - worst)
    return float(np.clip(t, 0, 1) * 100)


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 65:
        return "C"
    if score >= 50:
        return "D"
    return "F"


def run_health_check(
    shipments: Optional[pd.DataFrame] = None,
    kpis: Optional[dict] = None,
    audit: Optional[dict] = None,
    decision_outputs=None,
    delay_risk: Optional[float] = None,
    centroid_stats: Optional[pd.DataFrame] = None,
) -> dict:
    """Compute the health check. Returns {score, grade, difot, dimensions, recommendations}."""
    dims: list[dict] = []
    recs: list[str] = []
    difot = None

    # ── 1. Delivery performance ───────────────────────────────────────────────
    if kpis and kpis.get("on_time_pct") is not None and not pd.isna(kpis["on_time_pct"]):
        on_time = kpis["on_time_pct"]
        if shipments is not None and "health" in shipments.columns:
            closed = shipments[shipments["health"].isin(
                ["DELIVERED ON TIME", "DELIVERED LATE", "CANCELLED"])]
            if len(closed):
                difot = float((closed["health"] == "DELIVERED ON TIME").mean() * 100)
        s = _scale(on_time, worst=70, best=98)
        dims.append({"dimension": "Delivery performance", "score": s,
                     "detail": f"{on_time:.1f}% on-time vs promise"
                               + (f" · DIFOT (approx) {difot:.1f}%" if difot is not None else "")})
        if on_time < 92:
            recs.append(f"On-time delivery at {on_time:.1f}% — target ≥95%. Start with the "
                        f"worst-graded carriers on the scorecard.")

    # ── 2. Risk posture ───────────────────────────────────────────────────────
    if delay_risk is not None:
        risk_score = _scale(delay_risk, worst=30, best=5)
        detail = f"ML delay risk {delay_risk:.1f}%"
        if kpis and kpis.get("in_transit"):
            at_risk_share = kpis.get("at_risk", 0) / max(kpis["in_transit"], 1) * 100
            risk_score = 0.7 * risk_score + 0.3 * _scale(at_risk_share, worst=25, best=0)
            detail += f" · {at_risk_share:.0f}% of open shipments at risk"
        dims.append({"dimension": "Risk posture", "score": risk_score, "detail": detail})
        if delay_risk > 15:
            recs.append(f"Delay risk of {delay_risk:.1f}% needs intervention — review the "
                        f"disruption radar's signal-agreement zones.")

    # ── 3. Cost discipline ────────────────────────────────────────────────────
    if audit and audit["kpis"]["total_spend"] > 0:
        k = audit["kpis"]
        leak_pct = k["flagged_value"] / k["total_spend"] * 100
        s = _scale(leak_pct, worst=5, best=0)
        dims.append({"dimension": "Cost discipline", "score": s,
                     "detail": f"{leak_pct:.2f}% of freight spend flagged "
                               f"(${k['flagged_value']:,.0f} of ${k['total_spend']:,.0f})"})
        if k["retender_opportunity"] > 0.05 * k["total_spend"]:
            recs.append(f"Re-tender opportunity is ${k['retender_opportunity']:,.0f} "
                        f"({k['retender_opportunity']/k['total_spend']*100:.0f}% of spend) — "
                        f"run a freight tender.")

    # ── 4. Inventory discipline ───────────────────────────────────────────────
    if decision_outputs is not None and decision_outputs.total_optimized_cost > 0:
        gap_pct = (decision_outputs.savings_vs_current /
                   (decision_outputs.total_optimized_cost + decision_outputs.savings_vs_current) * 100)
        s = _scale(gap_pct, worst=40, best=0)
        dims.append({"dimension": "Inventory discipline", "score": s,
                     "detail": f"${decision_outputs.savings_vs_current:,.0f}/yr unrealised savings "
                               f"({gap_pct:.0f}% above optimal cost)"})
        if gap_pct > 15:
            recs.append(f"Inventory policy is {gap_pct:.0f}% above optimal cost — adopt the "
                        f"EOQ/ROP parameters from the Decision Engine.")

    # ── 5. Network efficiency ─────────────────────────────────────────────────
    if centroid_stats is not None and "efficiency_score" in getattr(centroid_stats, "columns", []):
        eff = float(centroid_stats["efficiency_score"].mean())
        dims.append({"dimension": "Network efficiency", "score": float(np.clip(eff, 0, 100)),
                     "detail": f"avg cluster efficiency {eff:.0f}%"})
        if eff < 60:
            recs.append(f"Network efficiency at {eff:.0f}% — test alternative hub counts "
                        f"with the NETWORK HUBS slider and cuOpt routing.")

    # ── 6. Data quality ───────────────────────────────────────────────────────
    if shipments is not None and len(shipments):
        checks = {
            "promised dates": shipments["promised_date"].notna().mean()
            if "promised_date" in shipments.columns else 0.0,
            "delivery dates": shipments["delivered_date"].notna().mean()
            if "delivered_date" in shipments.columns else 0.0,
            "carrier": shipments["carrier"].notna().mean()
            if "carrier" in shipments.columns else 0.0,
            "freight cost": shipments["freight_cost"].notna().mean()
            if "freight_cost" in shipments.columns else 0.0,
        }
        dq = float(np.mean(list(checks.values())) * 100)
        missing = [name for name, v in checks.items() if v < 0.5]
        dims.append({"dimension": "Data quality", "score": dq,
                     "detail": f"{dq:.0f}% field completeness"
                               + (f" · thin: {', '.join(missing)}" if missing else "")})
        if missing:
            recs.append(f"Data gaps in {', '.join(missing)} — richer delivery files unlock "
                        f"deeper carrier and cost analytics.")

    # ── Weighted overall ──────────────────────────────────────────────────────
    if dims:
        total_w = sum(_WEIGHTS[d["dimension"]] for d in dims)
        overall = sum(d["score"] * _WEIGHTS[d["dimension"]] for d in dims) / total_w
    else:
        overall = 0.0

    for d in dims:
        d["score"] = round(d["score"], 1)
        d["grade"] = _grade(d["score"])

    return {
        "score": round(overall, 1),
        "grade": _grade(overall),
        "difot": round(difot, 1) if difot is not None else None,
        "dimensions": dims,
        "recommendations": recs,
    }


def health_report(hc: dict) -> str:
    """Plain-text report of the health check."""
    lines = [
        "SUPCHAINMATE SUPPLY CHAIN HEALTH CHECK",
        "=" * 38,
        "",
        f"OVERALL: {hc['score']:.1f}/100 — GRADE {hc['grade']}",
    ]
    if hc.get("difot") is not None:
        lines.append(f"DIFOT (approx, delivered-in-full-on-time): {hc['difot']:.1f}%")
    lines += ["", "DIMENSIONS:"]
    for d in hc["dimensions"]:
        lines.append(f"  [{d['grade']}] {d['dimension']}: {d['score']:.0f}/100 — {d['detail']}")
    if hc["recommendations"]:
        lines += ["", "PRIORITY ACTIONS:"]
        lines += [f"  {i+1}. {r}" for i, r in enumerate(hc["recommendations"])]
    return "\n".join(lines)
