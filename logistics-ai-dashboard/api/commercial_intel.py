"""
api/commercial_intel.py — the Commercial Intelligence decision centre.

Executive commercial brief, Customer 360, true (activity-based) customer
profitability, revenue-leakage detection, contract intelligence, an AI pricing
optimiser, and customer risk scoring.

Grounding: account revenue and order volume are anchored to the real Olist order
data (aggregated by region into named enterprise accounts). Cost-to-serve,
contracts, payments, and SLAs are **modelled** with transparent, deterministic
per-account factors (seeded, so figures are stable) — labelled everywhere as
modelled. Wire a real cost-to-serve / contract feed and the same engine returns
exact figures.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from api import services

# ---- Modelled economics (labelled) -------------------------------------------
_AOV = 120.0                 # average order value ($)
_COGS_RATE = 0.68            # product cost as a share of revenue → 32% gross
# Cost-to-serve activity rates (share of revenue), summing to ~0.22 at complexity 1.
_ACTIVITY_RATES = {
    "warehouse_labour": 0.035, "picking": 0.020, "putaway": 0.012, "storage": 0.025,
    "transport": 0.050, "inventory_holding": 0.018, "returns": 0.012, "rework": 0.006,
    "urgent_orders": 0.008, "customer_support": 0.010, "overhead": 0.025,
}
_TARGET_NET_MARGIN = 0.12

_REGION_NAMES = {
    "SP": "Paulista Retail Group", "RJ": "Carioca Distribution", "MG": "Minas Supply Co.",
    "RS": "Sul Logística", "PR": "Paraná Trading", "SC": "Catarinense Goods",
    "BA": "Bahia Commerce", "DF": "Capital Federal Retail", "GO": "Goiás Wholesale",
    "ES": "Espírito Santo Foods", "PE": "Pernambuco Imports", "CE": "Ceará Distribution",
    "PA": "Pará Northern Supply", "MT": "Mato Grosso Agri", "MA": "Maranhão Trading",
}


def _seed(name: str) -> float:
    """Deterministic 0–1 value from an account name."""
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    return (h % 1000) / 1000.0


def _account_from(region: str, name: str, orders: int) -> dict[str, Any]:
    s = _seed(name)
    aov = _AOV * (0.8 + s * 0.7)                      # $96–$180
    revenue = orders * aov
    complexity = 0.65 + s * 1.05                      # 0.65–1.7 cost-to-serve multiplier
    cogs = revenue * _COGS_RATE
    activities = {k: round(revenue * r * complexity, 0) for k, r in _ACTIVITY_RATES.items()}
    serve_cost = sum(activities.values())
    true_cost = cogs + serve_cost
    net = revenue - true_cost
    net_pct = net / revenue * 100 if revenue else 0.0

    # Modelled operational profile
    returns_pct = round(2.0 + s * 6.0, 1)
    storage_util = round(55 + ((_seed(name + "u")) * 40), 0)
    dso = int(30 + _seed(name + "d") * 45)            # days sales outstanding
    pay_on_time = round(70 + _seed(name + "p") * 28, 0)
    sla_target = 98.0
    sla_actual = round(88 + _seed(name + "s") * 11, 1)
    renewal_months = int(1 + _seed(name + "r") * 22)
    vol_trend = round((_seed(name + "v") - 0.5) * 30, 1)  # % vs prior period

    # Leakage (modelled unbilled/undercharged activity)
    ub = _seed(name + "leak")
    leakage = {
        "unbilled_storage": round(activities["storage"] * ub * 0.4, 0),
        "undercharged_picks": round(activities["picking"] * ub * 0.5, 0),
        "freight_recovery": round(activities["transport"] * ub * 0.25, 0),
        "missing_invoices": round(revenue * 0.004 * ub, 0),
        "handling_charges": round(activities["warehouse_labour"] * ub * 0.2, 0),
    }
    leak_total = sum(leakage.values())

    # Contract terms (modelled) vs actual cost-to-serve
    contract = {
        "pick_fee": round(1.2 + s * 0.8, 2),
        "actual_pick_cost": round(activities["picking"] / max(orders, 1), 2),
        "storage_rate": round(12 + s * 8, 1),
        "sla_target": sla_target,
        "fuel_surcharge_pct": round(3 + s * 4, 1),
        "penalty_per_breach": round(250 + s * 500, 0),
        "rebate_pct": round(s * 3, 1),
        "renewal_months": renewal_months,
    }
    contract["unprofitable"] = net_pct < 0
    contract["pick_underpriced"] = contract["actual_pick_cost"] > contract["pick_fee"]

    action = net_pct < _TARGET_NET_MARGIN * 100 * 0.5 or leak_total > revenue * 0.01 or sla_actual < 92

    return {
        "id": name.lower().replace(" ", "-"),
        "name": name, "region": region, "orders": int(orders),
        "revenue": round(revenue, 0), "cogs": round(cogs, 0),
        "serve_cost": round(serve_cost, 0), "true_cost": round(true_cost, 0),
        "net_margin": round(net, 0), "net_margin_pct": round(net_pct, 1),
        "gross_margin_pct": round((1 - _COGS_RATE) * 100, 1),
        "activities": activities,
        "freight": activities["transport"], "returns_pct": returns_pct,
        "storage_util": storage_util, "dso": dso, "pay_on_time": pay_on_time,
        "sla_target": sla_target, "sla_actual": sla_actual, "vol_trend": vol_trend,
        "leakage": leakage, "leakage_total": round(leak_total, 0),
        "contract": contract, "action_required": bool(action),
    }


@services._ttl_cache(600)
def _accounts() -> list[dict[str, Any]]:
    """Build named enterprise accounts from real regional order volumes."""
    import pandas as pd
    orders = pd.read_csv("data/olist_orders_dataset.csv", usecols=["order_id", "customer_id"])
    customers = pd.read_csv("data/olist_customers_dataset.csv", usecols=["customer_id", "customer_state"])
    by_state = (orders.merge(customers, on="customer_id", how="left")
                .groupby("customer_state").size().reset_index(name="orders")
                .sort_values("orders", ascending=False))
    accts = []
    for _, r in by_state.iterrows():
        st = str(r["customer_state"])
        name = _REGION_NAMES.get(st)
        if not name:
            continue
        accts.append(_account_from(st, name, int(r["orders"])))
    accts.sort(key=lambda a: a["net_margin"], reverse=True)
    return accts


def _insights(a: dict) -> list[str]:
    out = []
    if a["net_margin_pct"] < 0:
        out.append(f"Loss-making at {a['net_margin_pct']}% net — cost-to-serve exceeds gross margin.")
    elif a["net_margin_pct"] < 6:
        out.append(f"Thin net margin ({a['net_margin_pct']}%) after cost-to-serve.")
    if a["contract"]["pick_underpriced"]:
        out.append(f"Pick fee ${a['contract']['pick_fee']} is below the ${a['contract']['actual_pick_cost']} actual cost.")
    if a["leakage_total"] > a["revenue"] * 0.01:
        out.append(f"${a['leakage_total']:,.0f} of billable activity is leaking (unbilled/undercharged).")
    if a["sla_actual"] < 92:
        out.append(f"SLA at {a['sla_actual']}% vs {a['sla_target']}% target — penalty exposure.")
    if a["dso"] > 55:
        out.append(f"Slow payer — DSO {a['dso']} days.")
    if a["vol_trend"] < -8:
        out.append(f"Volume declining {a['vol_trend']}% vs prior period.")
    return out or ["Healthy account — profitable and on-contract."]


# ---- Executive Commercial Brief ----------------------------------------------
def commercial_brief() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        accts = _accounts()
        revenue = sum(a["revenue"] for a in accts)
        true_cost = sum(a["true_cost"] for a in accts)
        gross = revenue * (1 - _COGS_RATE)
        net = revenue - true_cost
        leakage = sum(a["leakage_total"] for a in accts)
        action = [a for a in accts if a["action_required"]]
        # Uplift = leakage recovery + repricing loss-makers back to target
        reprice_uplift = sum(max(0, revenue_gap(a)) for a in accts)
        uplift = leakage * 0.6 + reprice_uplift
        recs = []
        for a in sorted(action, key=lambda x: x["net_margin"])[:4]:
            recs.append({
                "title": f"Reprice {a['name']} to target margin",
                "detail": f"Net margin {a['net_margin_pct']}%; recover leakage + align fees.",
                "impact_usd": round(max(revenue_gap(a), 0) + a["leakage_total"] * 0.6, 0),
                "confidence": 78 + int(_seed(a["name"]) * 15),
                "account": a["name"],
            })
        return {
            "total_revenue": round(revenue, 0), "true_operating_cost": round(true_cost, 0),
            "gross_margin_pct": round(gross / revenue * 100, 1) if revenue else 0,
            "net_margin": round(net, 0), "net_margin_pct": round(net / revenue * 100, 1) if revenue else 0,
            "revenue_leakage": round(leakage, 0),
            "profit_uplift": round(uplift, 0),
            "customers_action": len(action), "accounts_total": len(accts),
            "recommendations": recs,
            "summary": (
                f"${revenue/1e6:.1f}M revenue across {len(accts)} accounts at "
                f"{net/revenue*100:.1f}% true net margin after cost-to-serve. "
                f"${leakage:,.0f} is leaking and {len(action)} account(s) need immediate "
                f"pricing action — a modelled ${uplift:,.0f}/yr profit uplift is recoverable."),
        }
    return services._safe(build, {})


def revenue_gap(a: dict) -> float:
    """$ needed to bring an account to the target net margin (0 if already above)."""
    target_net = a["revenue"] * _TARGET_NET_MARGIN
    return target_net - a["net_margin"]


# ---- True Customer Profitability ---------------------------------------------
def profitability() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        accts = _accounts()
        ranking = [{
            "id": a["id"], "name": a["name"], "region": a["region"],
            "revenue": a["revenue"], "true_cost": a["true_cost"],
            "net_margin": a["net_margin"], "net_margin_pct": a["net_margin_pct"],
            "action": a["action_required"],
        } for a in accts]
        # Aggregate margin waterfall
        revenue = sum(a["revenue"] for a in accts)
        cogs = sum(a["cogs"] for a in accts)
        agg_act = {k: sum(a["activities"][k] for a in accts) for k in _ACTIVITY_RATES}
        waterfall = [{"label": "Revenue", "value": round(revenue, 0), "kind": "start"},
                     {"label": "COGS", "value": -round(cogs, 0), "kind": "neg"}]
        for k, v in sorted(agg_act.items(), key=lambda x: -x[1]):
            waterfall.append({"label": k.replace("_", " ").title(), "value": -round(v, 0), "kind": "neg"})
        waterfall.append({"label": "Net margin", "value": round(revenue - cogs - sum(agg_act.values()), 0), "kind": "end"})
        # Heatmap: account × cost category (share of revenue)
        cats = list(_ACTIVITY_RATES.keys())
        heatmap = [{"account": a["name"],
                    "cells": [round(a["activities"][k] / a["revenue"] * 100, 1) for k in cats]}
                   for a in accts]
        return {"ranking": ranking, "waterfall": waterfall,
                "heatmap": {"categories": cats, "rows": heatmap}}
    return services._safe(build, {"ranking": [], "waterfall": [], "heatmap": {}})


# ---- Customer 360 ------------------------------------------------------------
def customer_360(account_id: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        a = next((x for x in _accounts() if x["id"] == account_id), None)
        if a is None:
            a = _accounts()[0]
        out = dict(a)
        out["insights"] = _insights(a)
        out["forecast_next_qtr_orders"] = int(a["orders"] * (1 + a["vol_trend"] / 100) * 0.25)
        out["inventory_profile"] = {
            "skus": int(40 + _seed(a["name"] + "sku") * 200),
            "days_cover": int(12 + _seed(a["name"] + "dc") * 30),
            "storage_util": a["storage_util"],
        }
        out["revenue_gap"] = round(max(revenue_gap(a), 0), 0)
        return out
    return services._safe(build, {})


def account_list() -> dict[str, Any]:
    return services._safe(lambda: {"accounts": [
        {"id": a["id"], "name": a["name"], "region": a["region"],
         "net_margin_pct": a["net_margin_pct"], "action": a["action_required"]}
        for a in _accounts()]}, {"accounts": []})


# ---- Revenue Leakage Center --------------------------------------------------
_LEAK_CAUSES = {
    "unbilled_storage": "Storage consumed beyond the contracted allowance was never invoiced.",
    "undercharged_picks": "Pick fee billed below the actual pick cost.",
    "freight_recovery": "Fuel surcharge / accessorials not passed through.",
    "missing_invoices": "Fulfilled orders with no matching invoice.",
    "handling_charges": "Special handling performed but not charged.",
}


def leakage_center() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        accts = _accounts()
        by_cause: dict[str, float] = {k: 0.0 for k in _LEAK_CAUSES}
        items = []
        for a in accts:
            for cause, amt in a["leakage"].items():
                by_cause[cause] += amt
                if amt > 500:
                    items.append({
                        "account": a["name"], "cause": cause,
                        "cause_label": cause.replace("_", " ").title(),
                        "amount": round(amt, 0), "root_cause": _LEAK_CAUSES[cause],
                        "recoverable": round(amt * (0.6 if cause != "missing_invoices" else 0.9), 0),
                    })
        items.sort(key=lambda x: x["amount"], reverse=True)
        total = sum(by_cause.values())
        return {
            "annual_leakage": round(total, 0),
            "recoverable": round(sum(i["recoverable"] for i in items), 0),
            "by_cause": [{"cause": c.replace("_", " ").title(), "amount": round(v, 0),
                          "detail": _LEAK_CAUSES[c]} for c, v in
                         sorted(by_cause.items(), key=lambda x: -x[1])],
            "items": items[:20],
            "affected_customers": len({i["account"] for i in items}),
        }
    return services._safe(build, {"annual_leakage": 0, "recoverable": 0,
                                  "by_cause": [], "items": [], "affected_customers": 0})


def generate_invoice(account_id: str, cause: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules import store
        a = next((x for x in _accounts() if x["id"] == account_id), None)
        if a is None:
            raise KeyError(account_id)
        amt = a["leakage"].get(cause, 0)
        num = f"INV-REC-{abs(hash(account_id + cause)) % 100000:05d}"
        store.log_event("commercial", "recovery_invoice_generated",
                        details=f"{a['name']} · {cause} · ${amt:,.0f}", rec_key=num)
        return {"invoice_no": num, "account": a["name"],
                "line_item": cause.replace("_", " ").title(), "amount": round(amt, 0),
                "detail": _LEAK_CAUSES.get(cause, ""), "status": "draft"}
    return services._safe(build, {"invoice_no": "", "amount": 0, "status": "error"})


# ---- Contract Intelligence ---------------------------------------------------
def contract_intelligence() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        accts = _accounts()
        contracts = []
        for a in accts:
            c = a["contract"]
            contracts.append({
                "account": a["name"], "region": a["region"],
                "terms": {
                    "pick_fee": c["pick_fee"], "storage_rate": c["storage_rate"],
                    "sla_target": c["sla_target"], "fuel_surcharge_pct": c["fuel_surcharge_pct"],
                    "penalty_per_breach": c["penalty_per_breach"], "rebate_pct": c["rebate_pct"],
                    "renewal_months": c["renewal_months"],
                },
                "contractual_pick": c["pick_fee"], "actual_pick": c["actual_pick_cost"],
                "net_margin_pct": a["net_margin_pct"],
                "unprofitable": c["unprofitable"], "pick_underpriced": c["pick_underpriced"],
                "renewal_soon": c["renewal_months"] <= 3,
            })
        contracts.sort(key=lambda x: x["net_margin_pct"])
        return {"contracts": contracts,
                "unprofitable_count": sum(1 for c in contracts if c["unprofitable"]),
                "renewals_90d": sum(1 for c in contracts if c["renewal_soon"])}
    return services._safe(build, {"contracts": [], "unprofitable_count": 0, "renewals_90d": 0})


# ---- AI Pricing Optimizer ----------------------------------------------------
def pricing_optimizer() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        recs = []
        for a in _accounts():
            gap = revenue_gap(a)
            if gap <= 0:
                continue
            uplift_pct = min(0.18, gap / a["revenue"])   # required price uplift, capped 18%
            churn = round(min(35, uplift_pct * 180 + (a["vol_trend"] < 0) * 8), 0)
            recs.append({
                "id": a["id"], "account": a["name"], "region": a["region"],
                "net_margin_pct": a["net_margin_pct"],
                "changes": {
                    "storage_fee": f"+{round(uplift_pct*100*1.2,0)}%",
                    "picking_rate": f"+{round(uplift_pct*100,0)}%",
                    "freight_pass_through": "full",
                    "handling_fee": f"+{round(uplift_pct*100*0.8,0)}%",
                },
                "profit_uplift": round(gap, 0),
                "churn_risk_pct": churn,
                "confidence": 72 + int(_seed(a["name"] + "px") * 20),
                "negotiation": ("Low friction — below-market fees" if uplift_pct < 0.06
                                else "Moderate — phase over 2 cycles" if uplift_pct < 0.12
                                else "High — anchor on cost-to-serve transparency"),
                "evidence": [f"Net margin {a['net_margin_pct']}% vs {int(_TARGET_NET_MARGIN*100)}% target",
                             f"Actual pick ${a['contract']['actual_pick_cost']} > fee ${a['contract']['pick_fee']}"],
            })
        recs.sort(key=lambda x: x["profit_uplift"], reverse=True)
        return {"recommendations": recs[:12],
                "total_uplift": round(sum(r["profit_uplift"] for r in recs), 0)}
    return services._safe(build, {"recommendations": [], "total_uplift": 0})


# ---- Customer Risk Scoring ---------------------------------------------------
def _risk(a: dict, total_rev: float) -> dict[str, Any]:
    def band(v):  # 0–100 → label
        return "high" if v >= 66 else "medium" if v >= 33 else "low"
    profitability = max(0, min(100, 60 - a["net_margin_pct"] * 4))
    payment = max(0, min(100, (a["dso"] - 30) * 2 + (100 - a["pay_on_time"])))
    expiry = max(0, min(100, (4 - a["contract"]["renewal_months"]) * 25)) if a["contract"]["renewal_months"] <= 4 else 10
    service = max(0, min(100, (a["sla_target"] - a["sla_actual"]) * 8))
    volume = max(0, min(100, -a["vol_trend"] * 4)) if a["vol_trend"] < 0 else 5
    concentration = min(100, a["revenue"] / total_rev * 100 * 4)
    scores = {"profitability": profitability, "payment": payment, "contract_expiry": expiry,
              "service": service, "volume_decline": volume, "concentration": concentration}
    overall = round(sum(scores.values()) / len(scores), 0)
    return {"scores": {k: round(v, 0) for k, v in scores.items()},
            "bands": {k: band(v) for k, v in scores.items()},
            "overall": overall, "overall_band": band(overall)}


def risk_scoring() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        accts = _accounts()
        total = sum(a["revenue"] for a in accts) or 1
        rows = [{"id": a["id"], "account": a["name"], "region": a["region"], **_risk(a, total)}
                for a in accts]
        rows.sort(key=lambda r: r["overall"], reverse=True)
        return {"dimensions": ["profitability", "payment", "contract_expiry",
                               "service", "volume_decline", "concentration"],
                "rows": rows}
    return services._safe(build, {"dimensions": [], "rows": []})


# ---- Approve / reject / schedule (audited) -----------------------------------
def decide(item: str, action: str, note: str = "") -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules import store
        act = action.upper()
        if act not in ("APPROVED", "REJECTED", "SCHEDULED"):
            raise ValueError(action)
        store.log_event("commercial", f"commercial_{act.lower()}", details=f"{item}{(' — ' + note) if note else ''}")
        return {"ok": True, "item": item, "status": act}
    return services._safe(build, {"ok": False, "item": item, "status": action})
