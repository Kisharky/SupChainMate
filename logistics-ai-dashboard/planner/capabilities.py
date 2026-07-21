"""
planner/capabilities.py — the adapters that expose existing systems to the
Planner. These are the ONLY files aware of concrete service shapes; each handler
calls a system that already exists (AI Router, Optimization Engine, Commercial
Intelligence, domain services) and normalises the result. No business logic is
duplicated here — handlers are thin translators.

Registering a new capability here (or from anywhere via
``registry.register(...)``) makes it immediately available to the Planner with no
change to planner core.
"""

from __future__ import annotations

from planner.registry import CapabilityRegistry
from planner.schemas import Capability


# ---- Adapters (system → normalised result) -----------------------------------
def _forecast_demand(ctx: dict) -> dict:
    from api import services
    fc = services.forecast_snapshot()
    ins = fc.get("insights", {})
    return {
        "summary": (f"Next-week demand ≈ {ins.get('next_week_total', 0):,}; stockout risk "
                    f"{ins.get('stockout_risk_short', '—')}; WoW "
                    f"{ins.get('demand_pct_change_vs_prior_week', 0):+.1f}%."),
        "metrics": {"next_week_total": ins.get("next_week_total"),
                    "wow_change_pct": ins.get("demand_pct_change_vs_prior_week"),
                    "stockout_risk": ins.get("stockout_risk_short")},
        "confidence": 0.8,
    }


def _optimize_inventory(ctx: dict) -> dict:
    from api import services
    inv = services.inventory_snapshot()
    rows = inv.get("rows", [])
    savings = sum(float(r.get("savings_yr", 0) or 0) for r in rows)
    alloc = inv.get("allocation") or {}
    kpis = inv.get("kpis", {})
    return {
        "summary": (f"{kpis.get('n_skus', len(rows))} SKUs planned; ${savings:,.0f}/yr from "
                    f"reorder optimisation; multi-DC allocation "
                    f"{alloc.get('improvement_pct', 0):.0f}% below the naive baseline."),
        "metrics": {"n_skus": kpis.get("n_skus", len(rows)),
                    "est_savings_yr": round(savings, 0),
                    "alloc_saving_pct": alloc.get("improvement_pct")},
        "impact_usd": savings, "confidence": 0.85,
    }


def _routing_optimizer(ctx: dict) -> dict:
    from api import services
    r = services.optimize_route()
    return {
        "summary": (f"Network tour optimised to {r.get('objective', 0):,.0f} km "
                    f"({r.get('improvement_pct', 0):.0f}% shorter) via {r.get('solver', '—')}."),
        "metrics": {"objective_km": r.get("objective"), "improvement_pct": r.get("improvement_pct"),
                    "solver": r.get("solver")},
        "confidence": 0.8,
    }


def _warehouse_capacity(ctx: dict) -> dict:
    from api import services
    w = services.warehouse_snapshot()
    return {
        "summary": f"{w.get('hub_count', 0)} hub zones at {w.get('avg_utilization', 0):.0f}% average utilisation.",
        "metrics": {"hub_count": w.get("hub_count"), "avg_utilization": w.get("avg_utilization")},
        "confidence": 0.75,
    }


def _calculate_profitability(ctx: dict) -> dict:
    from api import commercial_intel
    b = commercial_intel.commercial_brief()
    return {
        "summary": (f"True net margin {b.get('net_margin_pct', 0)}% after cost-to-serve "
                    f"(gross {b.get('gross_margin_pct', 0)}%); {b.get('customers_action', 0)} "
                    f"accounts need pricing action; ${b.get('profit_uplift', 0):,.0f}/yr uplift."),
        "metrics": {"net_margin_pct": b.get("net_margin_pct"),
                    "gross_margin_pct": b.get("gross_margin_pct"),
                    "customers_action": b.get("customers_action")},
        "impact_usd": float(b.get("profit_uplift", 0) or 0), "confidence": 0.78,
    }


def _revenue_leakage(ctx: dict) -> dict:
    from api import commercial_intel
    lk = commercial_intel.leakage_center()
    return {
        "summary": (f"${lk.get('annual_leakage', 0):,.0f}/yr revenue leakage; "
                    f"${lk.get('recoverable', 0):,.0f} recoverable across "
                    f"{lk.get('affected_customers', 0)} accounts."),
        "metrics": {"annual_leakage": lk.get("annual_leakage"),
                    "recoverable": lk.get("recoverable"),
                    "affected_customers": lk.get("affected_customers")},
        "impact_usd": float(lk.get("recoverable", 0) or 0), "confidence": 0.8,
    }


def _contract_analysis(ctx: dict) -> dict:
    from api import commercial_intel
    c = commercial_intel.contract_intelligence()
    return {
        "summary": (f"{c.get('unprofitable_count', 0)} unprofitable contract(s); "
                    f"{c.get('renewals_90d', 0)} renew within 90 days."),
        "metrics": {"unprofitable_count": c.get("unprofitable_count"),
                    "renewals_90d": c.get("renewals_90d")},
        "confidence": 0.75,
    }


def _scenario_simulation(ctx: dict) -> dict:
    from api import workspace
    s = workspace.simulate_scenario("inventory_shortage", 0.5)
    return {
        "summary": s.get("narrative", ""),
        "metrics": {"financial_usd": (s.get("impact") or {}).get("financial_usd"),
                    "service_pp": (s.get("impact") or {}).get("service_pp")},
        "confidence": 0.6,
    }


# ---- Registration ------------------------------------------------------------
_DEFAULT = [
    Capability("forecast_demand", "Predict near-term demand and stockout risk.",
               required_inputs=[], outputs=["next_week_total", "stockout_risk"],
               dependencies=[], confidence=0.8, priority=1, handler=_forecast_demand,
               keywords=["forecast", "demand", "seasonal", "spike", "holding", "inventory", "plan"]),
    Capability("warehouse_capacity", "Assess hub zones and utilisation.",
               required_inputs=[], outputs=["hub_count", "avg_utilization"],
               dependencies=[], confidence=0.75, priority=1, handler=_warehouse_capacity,
               keywords=["warehouse", "capacity", "storage", "hub", "utilisation", "utilization", "space"]),
    Capability("optimize_inventory", "Optimise reorder points, EOQ and multi-DC allocation.",
               required_inputs=["demand"], outputs=["est_savings_yr", "alloc_saving_pct"],
               dependencies=["forecast_demand"], confidence=0.85, priority=2, handler=_optimize_inventory,
               keywords=["inventory", "stock", "holding", "reorder", "safety", "eoq", "cost", "reduce", "capital"]),
    Capability("routing_optimizer", "Optimise the network delivery tour (cuOpt/local).",
               required_inputs=["hubs"], outputs=["objective_km", "improvement_pct"],
               dependencies=["warehouse_capacity"], confidence=0.8, priority=2, handler=_routing_optimizer,
               keywords=["route", "routing", "logistics", "transport", "delivery", "freight", "distance", "network"]),
    Capability("revenue_leakage", "Detect unbilled/undercharged revenue leakage.",
               required_inputs=[], outputs=["recoverable"], dependencies=[], confidence=0.8,
               priority=2, handler=_revenue_leakage,
               keywords=["leakage", "billing", "recover", "revenue", "unbilled", "undercharged"]),
    Capability("contract_analysis", "Compare contract terms vs actual cost-to-serve.",
               required_inputs=[], outputs=["unprofitable_count"], dependencies=[], confidence=0.75,
               priority=2, handler=_contract_analysis,
               keywords=["contract", "sla", "pricing", "rate", "renewal", "terms"]),
    Capability("scenario_simulation", "Simulate a supply-chain disruption.",
               required_inputs=[], outputs=["financial_usd"], dependencies=[], confidence=0.6,
               priority=2, handler=_scenario_simulation,
               keywords=["scenario", "disruption", "risk", "simulate", "shortage", "what"]),
    Capability("calculate_profitability", "Quantify true customer profitability & pricing uplift.",
               required_inputs=[], outputs=["net_margin_pct", "profit_uplift"],
               dependencies=["optimize_inventory"], confidence=0.78, priority=3,
               handler=_calculate_profitability,
               keywords=["profit", "profitability", "margin", "cost", "commercial", "revenue", "pricing"]),
]


def register_default_capabilities(registry: CapabilityRegistry) -> CapabilityRegistry:
    for cap in _DEFAULT:
        registry.register(cap)
    return registry
