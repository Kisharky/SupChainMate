"""
modules/control_tower.py
Freight Control Tower — shipment tracking board + carrier scorecards.

Works on the same tracking dataframe the rest of the app uses:
- Olist demo shape  (order_purchase_timestamp, order_delivered_customer_date,
  order_estimated_delivery_date, order_status)
- Uploaded shape    (order_date, delivery_date, estimated_date, status, carrier)

When both promised and actual delivery dates exist, on-time performance is
computed from the real dates. Carriers come from a detected carrier column;
the demo assigns fictional carriers (labelled as simulated in the UI).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from modules import tracking

# Fictional carrier names for the demo dataset (no real companies).
DEMO_CARRIERS = ["NovaCargo", "SwiftLine", "BlueFreight", "RapidHaul", "TransAmerica XP"]
# Late shipments skew toward the weaker carriers so the demo scorecard shows a
# realistic performance spread (assignment is simulated; the dates are real).
_DEMO_WEIGHTS_ON_TIME = [0.36, 0.27, 0.19, 0.12, 0.06]
_DEMO_WEIGHTS_LATE = [0.18, 0.17, 0.15, 0.27, 0.23]
# Per-carrier simulated base freight rate ($/shipment) for the demo cost column.
_DEMO_BASE_RATES = {
    "NovaCargo": 8.5,
    "SwiftLine": 10.0,
    "BlueFreight": 7.0,
    "RapidHaul": 12.5,
    "TransAmerica XP": 15.0,
}

_ORDER_DATE_CANDIDATES = ["order_purchase_timestamp", "order_date", "date", "ds"]
_ACTUAL_DATE_CANDIDATES = ["order_delivered_customer_date", "delivery_date", "delivered_date"]
_PROMISED_DATE_CANDIDATES = ["order_estimated_delivery_date", "estimated_date", "promised_date", "eta"]

# An open shipment is "AT RISK" when its ML delay probability lands in the top
# decile of the fleet (with a 15% floor so a uniformly low-risk fleet flags nothing).
AT_RISK_PERCENTILE = 90
AT_RISK_FLOOR_PCT = 15.0


def _first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def assign_demo_carriers(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Attach simulated carrier names + freight costs to the demo tracking data."""
    df = df.copy()
    rng = np.random.default_rng(seed)

    actual_col = _first_col(df, _ACTUAL_DATE_CANDIDATES)
    promised_col = _first_col(df, _PROMISED_DATE_CANDIDATES)
    if actual_col and promised_col:
        actual = pd.to_datetime(df[actual_col], errors="coerce")
        promised = pd.to_datetime(df[promised_col], errors="coerce")
        late = (actual > promised).fillna(False).to_numpy()
    else:
        late = np.zeros(len(df), dtype=bool)

    carriers = rng.choice(DEMO_CARRIERS, size=len(df), p=_DEMO_WEIGHTS_ON_TIME)
    if late.any():
        carriers[late] = rng.choice(DEMO_CARRIERS, size=int(late.sum()), p=_DEMO_WEIGHTS_LATE)
    df["carrier"] = carriers

    base = df["carrier"].map(_DEMO_BASE_RATES).astype(float)
    costs = base.to_numpy() * rng.uniform(0.7, 1.5, size=len(df))
    # Inject ~0.4% simulated billing errors (2-4x normal rate) so the cost
    # audit's outlier detector has realistic anomalies to surface in the demo.
    n_anom = max(3, int(len(df) * 0.004))
    anom_idx = rng.choice(len(df), size=n_anom, replace=False)
    costs[anom_idx] *= rng.uniform(2.2, 4.0, size=n_anom)
    df["freight_cost"] = np.round(costs, 2)

    # Simulated transport mode per fictional carrier (for the Carbon Lens demo)
    from modules import carbon
    df["transport_mode"] = df["carrier"].map(carbon.DEMO_CARRIER_MODES).fillna("road")
    return df


def prepare_shipments(tracking_df: pd.DataFrame, delay_model=None) -> pd.DataFrame:
    """
    Build the shipment tracking board dataframe with one row per shipment:
        shipment_id, order_date, promised_date, delivered_date, carrier,
        status, delay_days, delay_proba, health
    """
    df = tracking_df.copy()
    out = pd.DataFrame(index=df.index)

    if "order_id" in df.columns:
        out["shipment_id"] = df["order_id"].astype(str).str[:8].str.upper()
    else:
        out["shipment_id"] = [f"SHP-{i:06d}" for i in range(len(df))]

    order_col = _first_col(df, _ORDER_DATE_CANDIDATES)
    actual_col = _first_col(df, _ACTUAL_DATE_CANDIDATES)
    promised_col = _first_col(df, _PROMISED_DATE_CANDIDATES)

    out["order_date"] = pd.to_datetime(df[order_col], errors="coerce") if order_col else pd.NaT
    out["delivered_date"] = pd.to_datetime(df[actual_col], errors="coerce") if actual_col else pd.NaT
    out["promised_date"] = pd.to_datetime(df[promised_col], errors="coerce") if promised_col else pd.NaT

    out["carrier"] = df["carrier"].astype(str) if "carrier" in df.columns else None
    if "freight_cost" in df.columns:
        out["freight_cost"] = pd.to_numeric(df["freight_cost"], errors="coerce")
    if "transport_mode" in df.columns:
        out["transport_mode"] = df["transport_mode"].astype(str)

    # Prefer the real order_status over the simulated status column when present.
    if "order_status" in df.columns:
        raw_status = df["order_status"].astype(str).str.lower()
        out["status"] = np.select(
            [
                raw_status.str.contains("deliver"),
                raw_status.str.contains("ship|transit"),
                raw_status.str.contains("cancel|unavailable"),
            ],
            ["Delivered", "Shipped", "Cancelled"],
            default="Processing",
        )
    elif "status" in df.columns:
        out["status"] = df["status"].astype(str)
    else:
        out["status"] = "Processing"

    # Delay vs promise (real dates when available)
    both = out["delivered_date"].notna() & out["promised_date"].notna()
    out["delay_days"] = np.nan
    out.loc[both, "delay_days"] = (
        (out.loc[both, "delivered_date"] - out.loc[both, "promised_date"]).dt.days
    )

    # ML delay probability for shipments still in flight
    if delay_model is not None:
        try:
            out["delay_proba"] = tracking.predict_delay_risk(delay_model, df) * 100.0
        except Exception:
            out["delay_proba"] = np.nan
    else:
        out["delay_proba"] = np.nan

    out["health"] = _classify_health(out)
    return out


def _classify_health(shipments: pd.DataFrame) -> pd.Series:
    delivered = shipments["status"] == "Delivered"
    cancelled = shipments["status"] == "Cancelled"
    late_flag = shipments["delay_days"] > 0

    proba = shipments["delay_proba"]
    if proba.notna().any():
        threshold = max(float(np.nanpercentile(proba, AT_RISK_PERCENTILE)), AT_RISK_FLOOR_PCT)
        risk_flag = proba >= threshold
    else:
        risk_flag = pd.Series(False, index=shipments.index)

    return pd.Series(
        np.select(
            [
                cancelled,
                delivered & late_flag,
                delivered,
                shipments["status"].eq("Delayed") | (~delivered & late_flag),
                ~delivered & risk_flag,
            ],
            ["CANCELLED", "DELIVERED LATE", "DELIVERED ON TIME", "LATE", "AT RISK"],
            default="ON TRACK",
        ),
        index=shipments.index,
    )


def shipment_kpis(shipments: pd.DataFrame) -> dict:
    """Headline control-tower KPIs computed over the full shipment set."""
    total = len(shipments)
    completed = shipments[shipments["delay_days"].notna()]
    on_time_pct = float((completed["delay_days"] <= 0).mean() * 100) if len(completed) else np.nan
    late_completed = completed[completed["delay_days"] > 0]
    open_mask = ~shipments["health"].isin(["DELIVERED ON TIME", "DELIVERED LATE", "CANCELLED"])
    return {
        "total": total,
        "in_transit": int(open_mask.sum()),
        "on_time_pct": on_time_pct,
        "late": int((shipments["health"].isin(["LATE", "DELIVERED LATE"])).sum()),
        "at_risk": int((shipments["health"] == "AT RISK").sum()),
        "avg_delay_days": float(late_completed["delay_days"].mean()) if len(late_completed) else 0.0,
    }


def _grade(on_time_pct: float) -> str:
    if pd.isna(on_time_pct):
        return "—"
    if on_time_pct >= 95:
        return "A"
    if on_time_pct >= 90:
        return "B"
    if on_time_pct >= 80:
        return "C"
    return "D"


def carrier_scorecard(shipments: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Per-carrier performance table: volume, on-time %, avg delay, risk, cost.
    Returns None when no carrier information is available.
    """
    if "carrier" not in shipments.columns or shipments["carrier"].isna().all():
        return None

    rows = []
    for carrier, grp in shipments.groupby("carrier"):
        completed = grp[grp["delay_days"].notna()]
        on_time = float((completed["delay_days"] <= 0).mean() * 100) if len(completed) else np.nan
        late = completed[completed["delay_days"] > 0]
        row = {
            "Carrier": carrier,
            "Shipments": len(grp),
            "On-Time %": round(on_time, 1) if not pd.isna(on_time) else np.nan,
            "Late": int((grp["health"].isin(["LATE", "DELIVERED LATE"])).sum()),
            "Avg Delay (days)": round(float(late["delay_days"].mean()), 1) if len(late) else 0.0,
            "Avg ML Risk %": round(float(grp["delay_proba"].mean()), 1)
            if grp["delay_proba"].notna().any()
            else np.nan,
            "Grade": _grade(on_time),
        }
        if "freight_cost" in grp.columns and grp["freight_cost"].notna().any():
            row["Avg Cost/Shipment ($)"] = round(float(grp["freight_cost"].mean()), 2)
        rows.append(row)

    score = pd.DataFrame(rows).sort_values(
        ["On-Time %", "Shipments"], ascending=[False, False], na_position="last"
    )
    return score.reset_index(drop=True)


def scorecard_insights(score: pd.DataFrame) -> list[str]:
    """Plain-language takeaways from the scorecard for the alert strip."""
    notes = []
    ranked = score.dropna(subset=["On-Time %"])
    if len(ranked) >= 2:
        best, worst = ranked.iloc[0], ranked.iloc[-1]
        gap = best["On-Time %"] - worst["On-Time %"]
        if gap >= 3:
            notes.append(
                f"{best['Carrier']} outperforms {worst['Carrier']} by "
                f"{gap:.1f} pts on-time ({best['On-Time %']:.1f}% vs {worst['On-Time %']:.1f}%) — "
                f"consider shifting volume."
            )
    poor = score[score["Grade"].isin(["C", "D"])]
    for _, r in poor.iterrows():
        notes.append(
            f"{r['Carrier']} is grading {r['Grade']} "
            f"({r['On-Time %']:.1f}% on-time, {r['Late']:,} late shipments) — review SLA."
        )
    if "Avg Cost/Shipment ($)" in score.columns and len(score.dropna(subset=["Avg Cost/Shipment ($)"])) >= 2:
        by_cost = score.dropna(subset=["Avg Cost/Shipment ($)"]).sort_values("Avg Cost/Shipment ($)")
        cheap, dear = by_cost.iloc[0], by_cost.iloc[-1]
        if dear["Avg Cost/Shipment ($)"] > cheap["Avg Cost/Shipment ($)"] * 1.2:
            notes.append(
                f"{dear['Carrier']} costs ${dear['Avg Cost/Shipment ($)']:.2f}/shipment vs "
                f"${cheap['Avg Cost/Shipment ($)']:.2f} at {cheap['Carrier']} — "
                f"{(dear['Avg Cost/Shipment ($)'] / cheap['Avg Cost/Shipment ($)'] - 1) * 100:.0f}% premium."
            )
    return notes[:4]
