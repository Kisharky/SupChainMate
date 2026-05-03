"""Helpers for Small Retailer mode: tier mapping, inventory status, table rows."""

from __future__ import annotations

import math
from typing import Any

from modules import decisions

# Service level targets matching shopkeeper "safety buffer" choice
SAFETY_TIER_TO_SERVICE_LEVEL = {
    "Low": 0.85,
    "Medium": 0.95,
    "High": 0.99,
}

# Retail-friendly defaults for EOQ / cost math
# $75 per order is appropriate for small retail (phone/email ordering, no formal procurement).
# Enterprise default is $200 (formal PO process, admin overhead).
DEFAULT_ORDERING_COST = 75.0
DEFAULT_HOLDING_RATE = 0.25

STATUS_ORDER_NOW = "ORDER NOW"
STATUS_ORDER_SOON = "ORDER SOON"
STATUS_OK = "OK"


def normalize_tier(tier: str) -> str:
    t = (tier or "Medium").strip()
    return t if t in SAFETY_TIER_TO_SERVICE_LEVEL else "Medium"


def service_level_for_tier(tier: str) -> float:
    return SAFETY_TIER_TO_SERVICE_LEVEL[normalize_tier(tier)]


def compute_inventory_status(current_stock: float, reorder_point: float) -> str:
    rop = float(reorder_point)
    if rop <= 0:
        return STATUS_OK
    cur = float(current_stock)
    if cur <= rop:
        return STATUS_ORDER_NOW
    if cur <= rop * 1.2:
        return STATUS_ORDER_SOON
    return STATUS_OK


def status_display_emoji(status: str) -> str:
    if status == STATUS_ORDER_NOW:
        return "🔴"
    if status == STATUS_ORDER_SOON:
        return "🟡"
    return "🟢"


def run_retail_decisions(
    units_per_week: float,
    lead_time_days: float,
    unit_cost: float,
    safety_tier: str,
    horizon_days: int = 7,
    ordering_cost: float = DEFAULT_ORDERING_COST,
    holding_rate: float = DEFAULT_HOLDING_RATE,
):
    """Profile + DecisionOutputs for one SKU."""
    tier = normalize_tier(safety_tier)
    profile = decisions.build_demand_profile_from_retail_inputs(
        units_per_week=units_per_week,
        avg_lead_time_days=lead_time_days,
        safety_tier=tier,
        horizon_days=horizon_days,
    )
    sl = service_level_for_tier(tier)
    outputs = decisions.run_decision_engine(
        profile,
        service_level=sl,
        unit_cost=float(unit_cost),
        holding_rate=holding_rate,
        ordering_cost=ordering_cost,
    )
    return profile, outputs


def product_dict(
    name: str,
    units_per_week: float,
    lead_time_days: float,
    unit_cost: float,
    safety_tier: str,
    current_stock: float = 0.0,
) -> dict[str, Any]:
    return {
        "name": name.strip(),
        "units_per_week": float(units_per_week),
        "lead_time_days": float(lead_time_days),
        "unit_cost": float(unit_cost),
        "safety_tier": normalize_tier(safety_tier),
        "current_stock": float(current_stock),
    }


def tracker_row(
    p: dict[str, Any],
    horizon_days: int = 7,
    ordering_cost: float = DEFAULT_ORDERING_COST,
    holding_rate: float = DEFAULT_HOLDING_RATE,
) -> dict[str, Any]:
    _, out = run_retail_decisions(
        p["units_per_week"],
        p["lead_time_days"],
        p["unit_cost"],
        p["safety_tier"],
        horizon_days=horizon_days,
        ordering_cost=ordering_cost,
        holding_rate=holding_rate,
    )
    rop = out.reorder_point
    st = compute_inventory_status(p["current_stock"], rop)
    return {
        "Product": p["name"],
        "Reorder when (units left)": int(math.ceil(rop)),
        "Order qty": int(round(out.eoq)),
        "Current stock": p["current_stock"],
        "Est. savings/yr ($)": int(round(out.savings_vs_current)),
        "Status": f"{status_display_emoji(st)} {st}",
    }
