"""
modules/sku.py
SKU Intelligence — per-product inventory decisions for Enterprise mode.

Takes order lines with a sku column and produces, for every SKU:
  - demand statistics over its active selling window
  - ABC class (revenue-share Pareto: A ~ top 80%, B next 15%, C last 5%)
  - safety stock, reorder point, EOQ, order cadence, annual savings
    via the shared decision engine (decisions.run_decision_engine)
  - ORDER NOW / ORDER SOON / OK status once current stock is entered

The demo assigns a simulated 12-SKU catalogue over the real Olist order
dates (labelled in the UI); uploads use the detected sku/price columns.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from modules import decisions

MAX_SKUS = 200  # cap the engine at the top-N SKUs by volume

# Fictional demo catalogue: (sku, weight, unit_price)
DEMO_CATALOG = [
    ("SKU-1001 Wireless Earbuds", 0.16, 45.0),
    ("SKU-1002 Phone Case", 0.14, 12.0),
    ("SKU-1003 USB-C Cable", 0.12, 8.0),
    ("SKU-1004 Bluetooth Speaker", 0.10, 65.0),
    ("SKU-1005 Laptop Stand", 0.09, 38.0),
    ("SKU-1006 Desk Lamp", 0.08, 27.0),
    ("SKU-1007 Water Bottle", 0.07, 15.0),
    ("SKU-1008 Backpack", 0.06, 55.0),
    ("SKU-1009 Yoga Mat", 0.05, 22.0),
    ("SKU-1010 Coffee Grinder", 0.05, 78.0),
    ("SKU-1011 Notebook Set", 0.04, 9.0),
    ("SKU-1012 Desk Organiser", 0.04, 18.0),
]


def assign_demo_skus(orders: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Attach a simulated SKU catalogue + prices to demo order lines."""
    df = orders.copy()
    rng = np.random.default_rng(seed)
    names = [c[0] for c in DEMO_CATALOG]
    weights = np.array([c[1] for c in DEMO_CATALOG])
    weights = weights / weights.sum()
    prices = {c[0]: c[2] for c in DEMO_CATALOG}
    df["sku"] = rng.choice(names, size=len(df), p=weights)
    df["unit_price"] = df["sku"].map(prices)
    return df


def sku_demand_profiles(orders: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Per-SKU demand statistics from order lines (order_date, quantity, sku,
    optional unit_price). Statistics are over each SKU's own active window.
    """
    if orders is None or "sku" not in orders.columns:
        return None
    df = orders.dropna(subset=["order_date", "sku"]).copy()
    if not len(df):
        return None
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["quantity"] = pd.to_numeric(df.get("quantity", 1.0), errors="coerce").fillna(1.0)
    has_price = "unit_price" in df.columns and df["unit_price"].notna().any()

    rows = []
    for sku_name, grp in df.groupby("sku"):
        daily = (grp.set_index("order_date")["quantity"]
                 .resample("D").sum())
        # active window: first to last sale (zeros in between count)
        daily = daily.loc[daily.ne(0).idxmax():]
        if not len(daily):
            continue
        avg_d, std_d = float(daily.mean()), float(daily.std() or 0.0)
        price = float(grp["unit_price"].dropna().mean()) if has_price else np.nan
        total_units = float(grp["quantity"].sum())
        rows.append({
            "SKU": str(sku_name),
            "Total Units": total_units,
            "Avg Daily": round(avg_d, 2),
            "Demand Std": round(std_d, 2),
            "Active Days": len(daily),
            "Unit Price": round(price, 2) if not math.isnan(price) else np.nan,
            "Revenue": round(total_units * price, 2) if not math.isnan(price) else np.nan,
        })
    if not rows:
        return None
    out = pd.DataFrame(rows).sort_values("Total Units", ascending=False).head(MAX_SKUS)
    return out.reset_index(drop=True)


def abc_classify(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Pareto ABC classification by revenue share (falls back to unit volume
    when no prices exist): A = top 80% of cumulative value, B = next 15%,
    C = the tail.
    """
    df = profiles.copy()
    basis = "Revenue" if df["Revenue"].notna().all() else "Total Units"
    df = df.sort_values(basis, ascending=False).reset_index(drop=True)
    total = df[basis].sum()
    cum = df[basis].cumsum() / total if total > 0 else pd.Series(1.0, index=df.index)
    df["ABC"] = np.select([cum <= 0.80, cum <= 0.95], ["A", "B"], default="C")
    # the first SKU is always A even if it alone exceeds 80%
    if len(df):
        df.loc[0, "ABC"] = "A"
    df["ABC Basis"] = basis
    return df


def run_sku_engine(
    profiles: pd.DataFrame,
    service_level: float = 0.95,
    avg_lead_time_days: float = 7.0,
    std_lead_time_days: float = 2.0,
    ordering_cost: float = 200.0,
    holding_rate: float = 0.25,
    default_unit_cost: float = 15.0,
) -> pd.DataFrame:
    """
    Run the shared decision engine per SKU. A-class SKUs get the full
    requested service level; B and C step down 3 / 8 points (capped at 85/80)
    — standard differentiated service-level practice.
    """
    out_rows = []
    for _, r in profiles.iterrows():
        svc = service_level
        if r.get("ABC") == "B":
            svc = max(0.85, service_level - 0.03)
        elif r.get("ABC") == "C":
            svc = max(0.80, service_level - 0.08)
        unit_cost = r["Unit Price"] if not pd.isna(r.get("Unit Price", np.nan)) else default_unit_cost

        profile = decisions.DemandProfile(
            avg_daily_demand=max(float(r["Avg Daily"]), 0.01),
            std_daily_demand=max(float(r["Demand Std"]), 0.01),
            avg_lead_time_days=avg_lead_time_days,
            std_lead_time_days=std_lead_time_days,
            annual_demand=round(max(float(r["Avg Daily"]), 0.01) * 365, 0),
            horizon_forecast=round(max(float(r["Avg Daily"]), 0.01) * 7, 0),
            horizon_days=7,
        )
        res = decisions.run_decision_engine(
            profile, service_level=svc, unit_cost=float(unit_cost),
            holding_rate=holding_rate, ordering_cost=ordering_cost,
        )
        out_rows.append({
            "SKU": r["SKU"],
            "ABC": r.get("ABC", "—"),
            "Avg Daily": r["Avg Daily"],
            "Svc Level": f"{svc*100:.0f}%",
            "Safety Stock": int(round(res.safety_stock)),
            "Reorder Point": int(math.ceil(res.reorder_point)),
            "EOQ": int(round(res.eoq)),
            "Order Every (d)": round(res.order_frequency_days, 1),
            "Est. Savings/yr ($)": round(res.savings_vs_current, 0),
            "Current Stock": 0.0,
        })
    return pd.DataFrame(out_rows)


def stock_status(plan: pd.DataFrame) -> pd.DataFrame:
    """ORDER NOW / ORDER SOON / OK from entered current stock vs ROP."""
    df = plan.copy()
    stock = pd.to_numeric(df["Current Stock"], errors="coerce").fillna(0.0)
    rop = df["Reorder Point"].astype(float)
    df["Status"] = np.select(
        [stock <= rop, stock <= rop * 1.25],
        ["🔴 ORDER NOW", "🟡 ORDER SOON"],
        default="🟢 OK",
    )
    return df


def sku_kpis(classified: pd.DataFrame, plan: Optional[pd.DataFrame] = None) -> dict:
    k = {
        "n_skus": len(classified),
        "a_class": int((classified["ABC"] == "A").sum()),
        "basis": classified["ABC Basis"].iloc[0] if len(classified) else "—",
        "total_revenue": float(classified["Revenue"].sum())
        if classified["Revenue"].notna().all() else None,
    }
    if plan is not None and "Status" in plan.columns:
        k["order_now"] = int(plan["Status"].str.contains("ORDER NOW").sum())
    return k
