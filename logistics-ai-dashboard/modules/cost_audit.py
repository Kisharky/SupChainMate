"""
modules/cost_audit.py
Freight Cost Audit — billing anomaly detection over the shipment board.

Runs three deterministic checks on shipments that carry a freight_cost:
  1. OUTLIER   — cost above Q3 + 1.5×IQR within the carrier's own profile
  2. DUPLICATE — same shipment billed twice, or identical
                 (carrier, day, cost) charges repeated
  3. LATE-PREMIUM — above-median rate paid for a shipment delivered late

Also quantifies the network re-tender opportunity: spend above the
network-median carrier rate. All figures come straight from the data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def run_audit(shipments: pd.DataFrame) -> Optional[dict]:
    """
    Audit freight charges. Returns None when no cost data is available.

    Returns dict:
        kpis     : headline numbers
        flagged  : DataFrame of individual suspect charges (with reason)
        by_carrier : per-carrier cost profile
        insights : plain-language findings
    """
    if "freight_cost" not in shipments.columns or shipments["freight_cost"].notna().sum() == 0:
        return None

    df = shipments[shipments["freight_cost"].notna()].copy()
    total_spend = float(df["freight_cost"].sum())
    has_carrier = "carrier" in df.columns and df["carrier"].notna().any()
    group_key = "carrier" if has_carrier else None

    flagged_frames = []

    # ── 1. Outliers vs the carrier's own cost profile ─────────────────────────
    if group_key:
        stats = df.groupby(group_key)["freight_cost"].agg(
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        stats["iqr_cap"] = stats["q3"] + 1.5 * (stats["q3"] - stats["q1"])
        df = df.join(stats[["median", "iqr_cap"]], on=group_key)
    else:
        q1, q3 = df["freight_cost"].quantile([0.25, 0.75])
        df["median"] = df["freight_cost"].median()
        df["iqr_cap"] = q3 + 1.5 * (q3 - q1)

    outliers = df[df["freight_cost"] > df["iqr_cap"]].copy()
    if len(outliers):
        outliers["overcharge_est"] = outliers["freight_cost"] - outliers["median"]
        outliers["reason"] = "OUTLIER — above carrier IQR cap"
        flagged_frames.append(outliers)
    est_overcharge = float(outliers["overcharge_est"].sum()) if len(outliers) else 0.0

    # ── 2. Potential duplicate charges ────────────────────────────────────────
    dup_mask = pd.Series(False, index=df.index)
    if "shipment_id" in df.columns:
        dup_mask |= df.duplicated(subset=["shipment_id"], keep="first")
    exact_keys = [k for k in [group_key, "order_date", "freight_cost"] if k]
    if len(exact_keys) >= 2:
        dup_mask |= df.duplicated(subset=exact_keys, keep="first")
    duplicates = df[dup_mask].copy()
    if len(duplicates):
        duplicates["overcharge_est"] = duplicates["freight_cost"]
        duplicates["reason"] = "POTENTIAL DUPLICATE charge"
        flagged_frames.append(duplicates)
    duplicate_value = float(duplicates["freight_cost"].sum()) if len(duplicates) else 0.0

    # ── 3. Premium paid for failure (late + above-median cost) ────────────────
    late_premium_value = 0.0
    if "health" in df.columns:
        late_premium = df[
            df["health"].isin(["LATE", "DELIVERED LATE"])
            & (df["freight_cost"] > df["median"])
            & ~df.index.isin(pd.concat(flagged_frames).index if flagged_frames else [])
        ].copy()
        if len(late_premium):
            late_premium["overcharge_est"] = late_premium["freight_cost"] - late_premium["median"]
            late_premium["reason"] = "LATE-PREMIUM — above-median rate, delivered late"
            flagged_frames.append(late_premium)
            late_premium_value = float(late_premium["overcharge_est"].sum())

    # ── Flagged charges table ─────────────────────────────────────────────────
    if flagged_frames:
        flagged = pd.concat(flagged_frames)
        cols = [c for c in ["shipment_id", "carrier", "order_date", "health",
                            "freight_cost", "median", "overcharge_est", "reason"] if c in flagged.columns]
        flagged = (
            flagged[cols]
            .rename(columns={"median": "carrier_median_cost"})
            .sort_values("overcharge_est", ascending=False)
            .round({"freight_cost": 2, "carrier_median_cost": 2, "overcharge_est": 2})
            .reset_index(drop=True)
        )
    else:
        flagged = pd.DataFrame()

    # ── Per-carrier cost profile + re-tender opportunity ──────────────────────
    by_carrier, retender_opportunity = None, 0.0
    if group_key:
        by_carrier = (
            df.groupby(group_key)["freight_cost"]
            .agg(Shipments="count", Avg_Cost="mean", Median_Cost="median",
                 P95_Cost=lambda s: s.quantile(0.95), Total_Spend="sum")
            .round(2)
            .sort_values("Avg_Cost")
            .reset_index()
            .rename(columns={group_key: "Carrier"})
        )
        network_median_rate = float(df["freight_cost"].median())
        above = by_carrier[by_carrier["Avg_Cost"] > network_median_rate]
        retender_opportunity = float(
            ((above["Avg_Cost"] - network_median_rate) * above["Shipments"]).sum()
        )

    kpis = {
        "total_spend": total_spend,
        "audited_charges": len(df),
        "flagged_count": len(flagged),
        "flagged_value": float(flagged["overcharge_est"].sum()) if len(flagged) else 0.0,
        "outlier_overcharge": est_overcharge,
        "duplicate_value": duplicate_value,
        "late_premium_value": late_premium_value,
        "retender_opportunity": retender_opportunity,
    }

    insights = []
    if len(outliers):
        worst = outliers.sort_values("overcharge_est", ascending=False).iloc[0]
        who = f" ({worst['carrier']})" if has_carrier else ""
        insights.append(
            f"{len(outliers):,} charges exceed their carrier's normal cost band — "
            f"est. ${est_overcharge:,.0f} overcharge. Worst: shipment "
            f"{worst.get('shipment_id', '?')}{who} at ${worst['freight_cost']:.2f} "
            f"vs ${worst['median']:.2f} typical."
        )
    if len(duplicates):
        insights.append(
            f"{len(duplicates):,} potential duplicate charges worth ${duplicate_value:,.0f} — "
            f"verify against carrier invoices before paying."
        )
    if late_premium_value > 0:
        insights.append(
            f"${late_premium_value:,.0f} paid above median rates on shipments that still "
            f"arrived late — cite this in carrier rate negotiations."
        )
    if retender_opportunity > 0 and by_carrier is not None:
        cheapest = by_carrier.iloc[0]
        insights.append(
            f"~${retender_opportunity:,.0f} of spend sits above the network-median rate — "
            f"re-tender ceiling if lanes shifted toward {cheapest['Carrier']}-level pricing "
            f"(${cheapest['Avg_Cost']:.2f}/shipment)."
        )
    if not insights:
        insights.append("No billing anomalies detected — freight charges are within normal bands.")

    return {"kpis": kpis, "flagged": flagged, "by_carrier": by_carrier, "insights": insights}


def audit_digest(audit: Optional[dict]) -> str:
    """Plain-text digest of the audit (for the agent / downloads)."""
    if audit is None:
        return "No freight cost data available — add a cost column to the delivery file."
    k = audit["kpis"]
    lines = [
        "SUPCHAINMATE FREIGHT COST AUDIT",
        "=" * 32,
        "",
        f"Total freight spend audited: ${k['total_spend']:,.0f} "
        f"({k['audited_charges']:,} charges)",
        f"Flagged charges: {k['flagged_count']:,} (est. value ${k['flagged_value']:,.0f})",
        f"  - Outlier overcharges: ${k['outlier_overcharge']:,.0f}",
        f"  - Potential duplicates: ${k['duplicate_value']:,.0f}",
        f"  - Late-delivery premiums: ${k['late_premium_value']:,.0f}",
        f"Re-tender opportunity (above network-median rate): ${k['retender_opportunity']:,.0f}",
        "",
        "FINDINGS:",
    ]
    lines += [f"  - {i}" for i in audit["insights"]]
    return "\n".join(lines)
