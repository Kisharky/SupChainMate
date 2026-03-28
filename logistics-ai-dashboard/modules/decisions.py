"""
modules/decisions.py
Supply Chain Decision Engine
────────────────────────────
Pure domain maths for prescriptive supply chain intelligence.

Formulas:
  Safety Stock  : SS  = Z × σ_d × √LT
  Reorder Point : ROP = μ_d × μ_LT + SS
  EOQ           : Q*  = √(2 × D × S / H)
  Lead Time Buf : LTB = Z × μ_d × σ_LT

References: Nahmias, "Production and Operations Analysis" (7th ed.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Service Level → Z-Score Lookup ────────────────────────────────────────────

SERVICE_LEVEL_Z = {
    0.80: 0.842,
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326,
    0.999: 3.090,
}


def z_score(service_level: float) -> float:
    """Return the Z-score for a given service level (0–1).
    Interpolates between known values if needed."""
    if service_level in SERVICE_LEVEL_Z:
        return SERVICE_LEVEL_Z[service_level]
    keys = sorted(SERVICE_LEVEL_Z.keys())
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo < service_level < hi:
            t = (service_level - lo) / (hi - lo)
            return SERVICE_LEVEL_Z[lo] + t * (SERVICE_LEVEL_Z[hi] - SERVICE_LEVEL_Z[lo])
    return SERVICE_LEVEL_Z[min(keys, key=lambda k: abs(k - service_level))]


# ── Core Decision Dataclass ────────────────────────────────────────────────────

@dataclass
class DemandProfile:
    """Summarised statistics derived from a daily demand series."""
    avg_daily_demand:    float   # μ_d
    std_daily_demand:    float   # σ_d
    avg_lead_time_days:  float   # μ_LT
    std_lead_time_days:  float   # σ_LT
    annual_demand:       float   # D  (avg_daily × 365)
    horizon_forecast:    float   # forecasted demand over horizon
    horizon_days:        int


@dataclass
class DecisionOutputs:
    """Full set of prescriptive recommendations."""
    # ── Safety Stock ──────────────────────────────────────────────────────────
    safety_stock:          float
    safety_stock_delta_pct: float   # % change vs current implied stock
    service_level:         float
    z_value:               float

    # ── EOQ ───────────────────────────────────────────────────────────────────
    eoq:                   float   # optimal order quantity (units)
    order_frequency_days:  float   # how often to order (days between orders)

    # ── Reorder Point ─────────────────────────────────────────────────────────
    reorder_point:         float   # trigger replenishment at this stock level

    # ── Lead Time Buffer ──────────────────────────────────────────────────────
    lead_time_buffer_days: float   # extra days to add to LT planning

    # ── Cost Impact ───────────────────────────────────────────────────────────
    holding_cost_current:  float
    holding_cost_optimized: float
    ordering_cost_annual:  float
    total_optimized_cost:  float
    savings_vs_current:    float

    # ── Intelligence narrative ────────────────────────────────────────────────
    recommendations:       list[dict] = field(default_factory=list)


# ── Core Calculation Engine ────────────────────────────────────────────────────

def build_demand_profile(
    daily_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    horizon_days: int = 7,
    avg_lead_time_days: float = 7.0,
    std_lead_time_days: float = 2.0,
) -> DemandProfile:
    """
    Construct a DemandProfile from the Prophet daily demand DataFrame and forecast.
    daily_df must have columns: ds (datetime), y (float).
    forecast_df must have columns: ds (datetime), yhat (float).
    """
    y = daily_df["y"].astype(float)
    avg_d = float(y.mean())
    std_d = float(y.std())

    future_yhat = forecast_df.tail(horizon_days)["yhat"].clip(lower=0)
    horizon_total = float(future_yhat.sum())

    return DemandProfile(
        avg_daily_demand=round(avg_d, 2),
        std_daily_demand=round(std_d, 2),
        avg_lead_time_days=avg_lead_time_days,
        std_lead_time_days=std_lead_time_days,
        annual_demand=round(avg_d * 365, 0),
        horizon_forecast=round(horizon_total, 0),
        horizon_days=horizon_days,
    )


def run_decision_engine(
    profile: DemandProfile,
    service_level: float = 0.95,
    unit_cost: float = 15.0,       # $ per unit
    holding_rate: float = 0.25,    # 25% of unit cost per year
    ordering_cost: float = 200.0,  # $ per purchase order
    current_safety_stock: Optional[float] = None,
) -> DecisionOutputs:
    """
    Run the full Supply Chain Decision Engine.

    Returns a DecisionOutputs with safety stock, EOQ, ROP, LT buffer,
    cost projections, and ranked textual recommendations.
    """
    z  = z_score(service_level)
    μd = profile.avg_daily_demand
    σd = profile.std_daily_demand
    μLT = profile.avg_lead_time_days
    σLT = profile.std_lead_time_days
    D  = profile.annual_demand        # annual demand (units/yr)
    H  = unit_cost * holding_rate     # holding cost ($/unit/yr)
    S  = ordering_cost                # setup/order cost ($/order)

    # ── Safety Stock ──────────────────────────────────────────────────────────
    # SS = Z × √( μLT × σd² + μd² × σLT² )   (combined variance formula)
    ss_combined = z * math.sqrt(μLT * (σd ** 2) + (μd ** 2) * (σLT ** 2))
    ss_combined = max(ss_combined, 0.0)

    # Simpler fallback (σ_demand only) if lead time variance is tiny
    ss_simple = z * σd * math.sqrt(μLT)

    safety_stock = round(max(ss_combined, ss_simple), 1)

    # Delta vs implied current (avg demand × half lead time as naive baseline)
    naive_current = current_safety_stock if current_safety_stock is not None else (μd * μLT * 0.5)
    delta_pct = ((safety_stock - naive_current) / naive_current * 100) if naive_current > 0 else 0.0

    # ── Reorder Point ─────────────────────────────────────────────────────────
    rop = round(μd * μLT + safety_stock, 1)

    # ── EOQ ───────────────────────────────────────────────────────────────────
    eoq = round(math.sqrt(2 * D * S / H), 1) if H > 0 else μd * 14
    order_freq = round(eoq / μd, 1) if μd > 0 else 0.0  # days between orders

    # ── Lead Time Buffer ──────────────────────────────────────────────────────
    lt_buffer = round(z * σLT, 1)

    # ── Cost Projections  ─────────────────────────────────────────────────────
    # Current (naive): order monthly-ish, hold too much stock
    naive_orders_py = max(D / (μd * 30), 1)
    holding_cost_current  = naive_current * unit_cost * holding_rate
    ordering_cost_current = naive_orders_py * ordering_cost

    # Optimised (EOQ-based)
    holding_cost_opt   = (eoq / 2 + safety_stock) * unit_cost * holding_rate
    ordering_cost_opt  = (D / eoq) * ordering_cost if eoq > 0 else ordering_cost
    total_opt          = holding_cost_opt + ordering_cost_opt
    total_current      = holding_cost_current + ordering_cost_current
    savings            = max(total_current - total_opt, 0.0)

    # ── Prescriptive Recommendations ─────────────────────────────────────────
    recs = []

    # Safety stock recommendation
    direction = "Increase" if delta_pct > 0 else "Decrease"
    recs.append({
        "priority": 1,
        "category": "SAFETY STOCK",
        "action": (
            f"{direction} safety stock by {abs(delta_pct):.0f}% "
            f"→ target {safety_stock:,.0f} units "
            f"(service level {service_level*100:.0f}%, Z={z:.2f})."
        ),
        "impact": "HIGH" if abs(delta_pct) > 20 else "MEDIUM",
    })

    # EOQ recommendation
    recs.append({
        "priority": 2,
        "category": "ORDER QUANTITY",
        "action": (
            f"Set EOQ to {eoq:,.0f} units per order — reorder every "
            f"{order_freq:.0f} days. "
            f"Reduces total inventory cost by ${savings:,.0f}/yr."
        ),
        "impact": "HIGH" if savings > 5000 else "MEDIUM",
    })

    # Reorder point
    recs.append({
        "priority": 3,
        "category": "REORDER POINT",
        "action": (
            f"Trigger replenishment at {rop:,.0f} units in stock. "
            f"This accounts for {μLT:.0f}-day lead time + safety buffer."
        ),
        "impact": "MEDIUM",
    })

    # Lead time buffer
    if lt_buffer > 0.5:
        recs.append({
            "priority": 4,
            "category": "LEAD TIME BUFFER",
            "action": (
                f"Add {lt_buffer:.1f} days buffer to all supplier lead times "
                f"(σ_LT={σLT:.1f} days variability detected)."
            ),
            "impact": "MEDIUM" if σLT > 1 else "LOW",
        })

    # Stockout risk warning
    if profile.horizon_forecast > μd * profile.horizon_days * 1.2:
        recs.append({
            "priority": 0,  # urgent
            "category": "STOCKOUT RISK",
            "action": (
                f"⚠ Forecast demand ({profile.horizon_forecast:,.0f} units) "
                f"exceeds normal range by "
                f"{(profile.horizon_forecast/(μd*profile.horizon_days)-1)*100:.0f}%. "
                f"Pre-position inventory NOW."
            ),
            "impact": "CRITICAL",
        })

    recs.sort(key=lambda r: r["priority"])

    return DecisionOutputs(
        safety_stock=safety_stock,
        safety_stock_delta_pct=round(delta_pct, 1),
        service_level=service_level,
        z_value=round(z, 3),
        eoq=eoq,
        order_frequency_days=order_freq,
        reorder_point=rop,
        lead_time_buffer_days=lt_buffer,
        holding_cost_current=round(holding_cost_current, 2),
        holding_cost_optimized=round(holding_cost_opt, 2),
        ordering_cost_annual=round(ordering_cost_opt, 2),
        total_optimized_cost=round(total_opt, 2),
        savings_vs_current=round(savings, 2),
        recommendations=recs,
    )


# ── Execution Plan Export ──────────────────────────────────────────────────────

def build_execution_plan(
    profile: DemandProfile,
    outputs: DecisionOutputs,
    unit_cost: float,
    ordering_cost: float,
) -> pd.DataFrame:
    """
    Return a structured DataFrame suitable for CSV/Excel export.
    This is the 'Execution Plan' the user can download and act on.
    """
    rows = []
    for rec in outputs.recommendations:
        rows.append({
            "Priority":    rec["priority"],
            "Category":    rec["category"],
            "Action":      rec["action"],
            "Impact":      rec["impact"],
            "Owner":       "Procurement / Ops",
            "Target Date": f"Within {rec['priority'] * 3} days",
        })

    rows.append({
        "Priority":    99,
        "Category":    "PARAMETERS",
        "Action": (
            f"Service Level: {outputs.service_level*100:.0f}% | "
            f"Safety Stock: {outputs.safety_stock:,.0f} units | "
            f"EOQ: {outputs.eoq:,.0f} units | "
            f"ROP: {outputs.reorder_point:,.0f} units | "
            f"LT Buffer: +{outputs.lead_time_buffer_days:.1f} days | "
            f"Annual Savings: ${outputs.savings_vs_current:,.0f}"
        ),
        "Impact":      "—",
        "Owner":       "AI Decision Engine",
        "Target Date": "Reference",
    })

    return pd.DataFrame(rows).sort_values("Priority")
