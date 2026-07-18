"""
modules/trust.py
Decision trust layer — explainable, scored, human-approved recommendations.

Adapts the design patterns common to enterprise supply chain platforms
(action centers, explanation drill-downs, confidence scoring, approval
workflows, audit trails) into an open implementation:

  - Recommendation: a typed record with WHY drivers (each backed by an
    evidence value), a confidence score with a stated basis, and quantified
    business impact (cost savings, stockout risk, service level).
  - Builders wrap the existing deterministic engines (decision engine, SKU
    engine, carrier scorecard + rate simulator, cost audit) — the trust
    layer never invents numbers, it packages the engines' outputs with
    their reasoning.
  - Human workflow: PENDING → APPROVED / REJECTED / MODIFIED, persisted in
    SQLite with an immutable audit trail (see modules/store.py).

Confidence is a transparent heuristic (data support + signal strength),
NOT a calibrated probability — the basis string states exactly what it is
built from, because trust starts with not overstating certainty.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from modules import store
from modules import tender as tender_mod

_log = config.get_logger(__name__)

VALID_STATUSES = ("PENDING", "APPROVED", "REJECTED", "MODIFIED")
MAX_SKU_RECOMMENDATIONS = 5


@dataclass
class Driver:
    """One 'because' behind a recommendation, backed by an evidence value."""
    reason: str
    evidence: str


@dataclass
class Impact:
    """Quantified business impact. None = not applicable, never guessed."""
    cost_savings_yr: Optional[float] = None
    stockout_risk_pct: Optional[float] = None
    service_level_pct: Optional[float] = None
    other: Optional[str] = None


@dataclass
class Recommendation:
    source: str                 # which worker/engine produced it
    category: str
    title: str
    action: str                 # the concrete thing to do
    drivers: list[Driver] = field(default_factory=list)
    confidence: float = 50.0    # 0-100
    confidence_basis: str = ""
    impact: Impact = field(default_factory=Impact)

    @property
    def key(self) -> str:
        return hashlib.sha1(f"{self.category}|{self.title}".encode()).hexdigest()[:16]

    def to_payload(self) -> dict:
        d = asdict(self)
        return d


def _confidence(support: float, strength: float, basis: str) -> tuple[float, str]:
    """
    Transparent heuristic: 30 base + up to 40 for data support + up to 30
    for signal strength, clamped to [20, 95]. Both inputs are 0-1.
    """
    score = 30.0 + 40.0 * float(np.clip(support, 0, 1)) + 30.0 * float(np.clip(strength, 0, 1))
    return float(np.clip(round(score, 1), 20.0, 95.0)), basis


# ══════════════════════════════════════════════════════════════════════════════
# Builders — one per engine
# ══════════════════════════════════════════════════════════════════════════════

def from_decision_engine(demand_profile, outputs, history_days: int,
                         service_level: float) -> list[Recommendation]:
    """Inventory-policy recommendations with the engine's own math as WHY."""
    if outputs is None or demand_profile is None:
        return []
    cv = (demand_profile.std_daily_demand / demand_profile.avg_daily_demand
          if demand_profile.avg_daily_demand > 0 else 1.0)
    support = min(1.0, history_days / 180.0)
    strength = 1.0 / (1.0 + cv)
    conf, basis = _confidence(
        support, strength,
        f"{history_days} days of demand history; demand CV {cv:.2f}; "
        f"lead-time σ {demand_profile.std_lead_time_days:.1f}d")

    drivers = [
        Driver("Average demand", f"{demand_profile.avg_daily_demand:,.1f} units/day"),
        Driver("Demand variability", f"σ = {demand_profile.std_daily_demand:,.1f} units/day (CV {cv:.2f})"),
        Driver("Lead time", f"{demand_profile.avg_lead_time_days:.0f} days ± {demand_profile.std_lead_time_days:.1f}"),
        Driver("Service level target", f"{service_level*100:.0f}% → Z = {outputs.z_value}"),
        Driver("Formula", "SS = Z × √(μ_LT·σ_d² + μ_d²·σ_LT²); EOQ = √(2DS/H)"),
    ]
    impact = Impact(
        cost_savings_yr=float(outputs.savings_vs_current),
        stockout_risk_pct=round((1 - service_level) * 100, 1),
        service_level_pct=round(service_level * 100, 1),
    )
    return [Recommendation(
        source="Planner", category="INVENTORY POLICY",
        title=f"Adopt SS {outputs.safety_stock:,.0f} / ROP {outputs.reorder_point:,.0f} / EOQ {outputs.eoq:,.0f}",
        action=(f"Set safety stock to {outputs.safety_stock:,.0f} units, trigger replenishment at "
                f"{outputs.reorder_point:,.0f} units, order {outputs.eoq:,.0f} units every "
                f"{outputs.order_frequency_days:.0f} days."),
        drivers=drivers, confidence=conf, confidence_basis=basis, impact=impact)]


def from_sku_engine(sku_plan: Optional[pd.DataFrame],
                    avg_lead_time_days: float) -> list[Recommendation]:
    """Reorder recommendations for SKUs at/below their reorder point."""
    if sku_plan is None or "Status" not in getattr(sku_plan, "columns", []):
        return []
    urgent = sku_plan[sku_plan["Status"].str.contains("ORDER NOW", na=False)]
    recs: list[Recommendation] = []
    for _, r in urgent.head(MAX_SKU_RECOMMENDATIONS).iterrows():
        stock = float(r.get("Current Stock", 0) or 0)
        avg_daily = max(float(r.get("Avg Daily", 0) or 0), 0.01)
        cover_days = stock / avg_daily
        stockout_risk = float(np.clip((1 - cover_days / max(avg_lead_time_days, 0.1)) * 100, 0, 100))
        conf, basis = _confidence(
            min(1.0, float(r.get("Avg Daily", 0)) * 30 / 100),
            stockout_risk / 100,
            f"{cover_days:.1f} days of cover vs {avg_lead_time_days:.0f}-day lead time; "
            f"ABC class {r.get('ABC', '?')}")
        recs.append(Recommendation(
            source="Planner", category="SKU REORDER",
            title=f"Reorder {r['SKU']} — {int(r['EOQ']):,} units",
            action=(f"Place a purchase order for {int(r['EOQ']):,} units of {r['SKU']} now "
                    f"(stock {stock:,.0f} vs reorder point {int(r['Reorder Point']):,})."),
            drivers=[
                Driver("Stock position", f"{stock:,.0f} units on hand ≤ ROP {int(r['Reorder Point']):,}"),
                Driver("Days of cover", f"{cover_days:.1f} days vs {avg_lead_time_days:.0f}-day lead time"),
                Driver("Demand rate", f"{r['Avg Daily']} units/day"),
                Driver("ABC class", f"{r.get('ABC', '?')} — service level {r.get('Svc Level', '?')}"),
            ],
            confidence=conf, confidence_basis=basis,
            impact=Impact(cost_savings_yr=float(r.get("Est. Savings/yr ($)", 0) or 0),
                          stockout_risk_pct=round(stockout_risk, 1)),
        ))
    return recs


def from_carrier_scorecard(scorecard: Optional[pd.DataFrame],
                           by_carrier: Optional[pd.DataFrame],
                           shift_pct: int = 25) -> list[Recommendation]:
    """Volume-shift recommendation when the on-time gap is material."""
    if scorecard is None or len(scorecard) < 2:
        return []
    ranked = scorecard.dropna(subset=["On-Time %"])
    if len(ranked) < 2:
        return []
    best, worst = ranked.iloc[0], ranked.iloc[-1]
    gap = float(best["On-Time %"] - worst["On-Time %"])
    if gap < 3:
        return []

    saving = None
    if by_carrier is not None:
        sim = tender_mod.simulate_rate_shift(by_carrier, str(worst["Carrier"]),
                                             str(best["Carrier"]), shift_pct)
        if sim and sim["cost_delta"] < 0:
            saving = float(-sim["cost_delta"])

    n_worst = int(worst["Shipments"])
    conf, basis = _confidence(
        min(1.0, n_worst / 2000.0), min(1.0, gap / 15.0),
        f"{n_worst:,} shipments observed for {worst['Carrier']}; "
        f"on-time gap {gap:.1f} pts vs {best['Carrier']}")
    return [Recommendation(
        source="Carrier Manager", category="CARRIER MIX",
        title=f"Shift {shift_pct}% of {worst['Carrier']} volume to {best['Carrier']}",
        action=(f"Move {shift_pct}% of {worst['Carrier']}'s shipments to {best['Carrier']} "
                f"and open an SLA review with {worst['Carrier']}."),
        drivers=[
            Driver("Performance gap", f"{best['Carrier']} {best['On-Time %']}% vs "
                                      f"{worst['Carrier']} {worst['On-Time %']}% on-time"),
            Driver("Grades", f"{best['Carrier']}: {best['Grade']} · {worst['Carrier']}: {worst['Grade']}"),
            Driver("Late shipments", f"{int(worst['Late']):,} late from {worst['Carrier']}"),
        ],
        confidence=conf, confidence_basis=basis,
        impact=Impact(cost_savings_yr=saving,
                      service_level_pct=float(best["On-Time %"]),
                      other=f"+{gap:.1f} pts on-time on shifted volume"),
    )]


def from_cost_audit(audit: Optional[dict]) -> list[Recommendation]:
    """Dispute recommendation when the audit finds material overcharges."""
    if not audit:
        return []
    k = audit["kpis"]
    if k["flagged_value"] < 100:
        return []
    leak_ratio = k["flagged_value"] / max(k["total_spend"], 1)
    conf, basis = _confidence(
        min(1.0, k["audited_charges"] / 5000.0), min(1.0, leak_ratio * 20),
        f"{k['audited_charges']:,} charges audited; deterministic IQR/duplicate checks")
    return [Recommendation(
        source="Auditor", category="BILLING DISPUTE",
        title=f"Dispute {k['flagged_count']:,} flagged charges (${k['flagged_value']:,.0f})",
        action=(f"Raise a dispute pack for {k['flagged_count']:,} flagged charges before the "
                f"next payment run; start with the outliers."),
        drivers=[
            Driver("Outliers vs carrier rate bands", f"${k['outlier_overcharge']:,.0f}"),
            Driver("Potential duplicates", f"${k['duplicate_value']:,.0f}"),
            Driver("Late-delivery premiums", f"${k['late_premium_value']:,.0f}"),
        ],
        confidence=conf, confidence_basis=basis,
        impact=Impact(cost_savings_yr=float(k["flagged_value"])),
    )]


def generate_all(ctx: dict) -> list[Recommendation]:
    """Run every builder over the live context. Pure function, no I/O."""
    recs: list[Recommendation] = []
    recs += from_decision_engine(ctx.get("demand_profile"), ctx.get("decision_outputs"),
                                 history_days=int(ctx.get("history_days", 0)),
                                 service_level=float(ctx.get("service_level", 0.95)))
    recs += from_sku_engine(ctx.get("sku_plan"),
                            avg_lead_time_days=float(ctx.get("avg_lead_time", 7.0)))
    recs += from_carrier_scorecard(ctx.get("scorecard"),
                                   (ctx.get("audit") or {}).get("by_carrier"))
    recs += from_cost_audit(ctx.get("audit"))
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# Workflow — persistence + audit around human decisions
# ══════════════════════════════════════════════════════════════════════════════

def sync_recommendations(recs: list[Recommendation]) -> int:
    """Persist newly generated recommendations (deduped by key). Returns #new."""
    created = 0
    for rec in recs:
        if store.save_recommendation(rec.key, rec.to_payload()):
            store.log_event("system", "recommendation_created",
                            details=rec.title, rec_key=rec.key)
            created += 1
    if created:
        _log.info("Created %d new recommendation(s)", created)
    return created


def decide(rec_key: str, status: str, note: str = "", actor: str = "user") -> bool:
    """Apply a human decision (APPROVED / REJECTED / MODIFIED) with audit."""
    if status not in VALID_STATUSES or status == "PENDING":
        raise ValueError(f"Invalid decision status: {status}")
    ok = store.update_recommendation(rec_key, status, decided_by=actor, note=note)
    if ok:
        store.log_event(actor, f"recommendation_{status.lower()}",
                        details=note, rec_key=rec_key)
        _log.info("Recommendation %s -> %s", rec_key, status)
    return ok


def pending() -> list[dict]:
    return store.load_recommendations(status="PENDING")


def history(limit: int = 100) -> list[dict]:
    return [r for r in store.load_recommendations(limit=limit)
            if r["status"] != "PENDING"]


def summary_kpis() -> dict:
    """Headline numbers for the Decision Center strip."""
    all_recs = store.load_recommendations(limit=500)
    pend = [r for r in all_recs if r["status"] == "PENDING"]
    approved = [r for r in all_recs if r["status"] in ("APPROVED", "MODIFIED")]
    approved_savings = sum((r.get("impact") or {}).get("cost_savings_yr") or 0
                           for r in approved)
    return {
        "pending": len(pend),
        "approved": len(approved),
        "rejected": sum(1 for r in all_recs if r["status"] == "REJECTED"),
        "approved_savings": float(approved_savings),
        "avg_confidence": round(float(np.mean([r.get("confidence", 0) for r in pend])), 1)
        if pend else None,
    }
