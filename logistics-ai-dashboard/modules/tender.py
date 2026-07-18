"""
modules/tender.py
Freight Tender / RFP toolkit + carrier rate-shift simulation.

- build_tender_pack(): lane & volume summary + a ready-to-edit RFP document,
  built entirely from the shipment board (volumes, seasonality, current rates,
  service levels). LLM wording is optional; numbers never come from the LLM.
- simulate_rate_shift(): what-if for moving a % of one carrier's volume to
  another carrier's average rate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def build_tender_pack(shipments: pd.DataFrame, scorecard: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """
    Build the tender data pack. Returns None if there is no usable data.
        lanes    : DataFrame — monthly volume (and spend where available)
        carriers : DataFrame — current-state carrier summary for bidders' context
        rfp_text : str       — RFP document draft
        stats    : dict      — headline numbers used in the RFP
    """
    if shipments is None or not len(shipments):
        return None
    df = shipments.copy()
    if "order_date" not in df.columns or df["order_date"].isna().all():
        return None

    df = df[df["order_date"].notna()]
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    has_cost = "freight_cost" in df.columns and df["freight_cost"].notna().any()
    has_carrier = "carrier" in df.columns and df["carrier"].notna().any()

    agg = {"Shipments": ("shipment_id", "count")}
    if has_cost:
        agg["Spend"] = ("freight_cost", "sum")
    lanes = df.groupby("month").agg(**agg).reset_index().rename(columns={"month": "Month"})
    if has_cost:
        lanes["Spend"] = lanes["Spend"].round(2)

    monthly_avg = float(lanes["Shipments"].mean())
    peak = lanes.loc[lanes["Shipments"].idxmax()]
    total_shipments = int(lanes["Shipments"].sum())
    annual_spend = float(df["freight_cost"].sum()) if has_cost else None

    completed = df[df["delay_days"].notna()] if "delay_days" in df.columns else pd.DataFrame()
    on_time = float((completed["delay_days"] <= 0).mean() * 100) if len(completed) else None

    carriers = None
    if has_carrier:
        c_agg = {"Shipments": ("shipment_id", "count")}
        if has_cost:
            c_agg["Avg_Cost"] = ("freight_cost", "mean")
        carriers = df.groupby("carrier").agg(**c_agg).reset_index().rename(columns={"carrier": "Carrier"})
        if has_cost:
            carriers["Avg_Cost"] = carriers["Avg_Cost"].round(2)
        carriers = carriers.sort_values("Shipments", ascending=False).reset_index(drop=True)

    stats = {
        "total_shipments": total_shipments,
        "monthly_avg": monthly_avg,
        "peak_month": str(peak["Month"]),
        "peak_shipments": int(peak["Shipments"]),
        "annual_spend": annual_spend,
        "on_time": on_time,
        "n_carriers": int(df["carrier"].nunique()) if has_carrier else 0,
        "period": f"{df['order_date'].min():%Y-%m} to {df['order_date'].max():%Y-%m}",
    }

    spend_line = (f"Current freight spend over the data period is ${annual_spend:,.0f}."
                  if annual_spend is not None else
                  "Freight spend data will be shared with shortlisted bidders.")
    otd_line = (f"Current network on-time performance is {on_time:.1f}%; bidders must commit "
                f"to a DIFOT of 95% or better." if on_time is not None else
                "Bidders must commit to a DIFOT of 95% or better.")

    rfp_text = f"""REQUEST FOR PROPOSAL — FREIGHT & DISTRIBUTION SERVICES
[Company name]
Issue date: [date] · Responses due: [date + 3 weeks]

1. INTRODUCTION
[Company name] invites proposals from qualified freight carriers for its
distribution network. This RFP is data-backed: the volumes below are actuals
from our shipment records ({stats['period']}).

2. SCOPE & VOLUMES
- Total shipments over the period: {total_shipments:,}
- Average monthly volume: {monthly_avg:,.0f} shipments
- Peak month: {stats['peak_month']} ({stats['peak_shipments']:,} shipments) — capacity
  commitments must cover peak, not average
- Incumbent carriers: {stats['n_carriers'] or '[number]'}
- {spend_line}
(Full monthly breakdown attached: tender_lane_summary.csv)

3. SERVICE REQUIREMENTS
- {otd_line}
- Proactive exception notification within 4 business hours of a delay event
- Monthly performance reporting: on-time %, damages, invoice accuracy
- EDI/API integration for tracking-event feeds preferred

4. PRICING
- Quote per-shipment rates by lane/service tier, valid 12 months
- State all surcharges explicitly (fuel, residential, re-delivery); unlisted
  surcharges will not be paid
- Volume-tier discounts and peak-season capacity pricing to be itemised

5. EVALUATION CRITERIA
- Price (40%) · Service history & DIFOT commitment (30%)
- Network fit & capacity at peak (20%) · Technology & integration (10%)

6. RESPONSE FORMAT
Submit pricing in the attached lane summary structure. Include two customer
references at comparable volume.

[Contact name] · [email] · [phone]
"""

    return {"lanes": lanes, "carriers": carriers, "rfp_text": rfp_text, "stats": stats}


def simulate_rate_shift(
    by_carrier: pd.DataFrame,
    from_carrier: str,
    to_carrier: str,
    shift_pct: float,
) -> Optional[dict]:
    """
    Estimate annualised cost impact of moving shift_pct% of from_carrier's
    shipments to to_carrier's average rate. Uses the cost-audit per-carrier
    profile (columns: Carrier, Shipments, Avg_Cost, Total_Spend).
    """
    if by_carrier is None or from_carrier == to_carrier:
        return None
    src = by_carrier[by_carrier["Carrier"] == from_carrier]
    dst = by_carrier[by_carrier["Carrier"] == to_carrier]
    if src.empty or dst.empty:
        return None
    src, dst = src.iloc[0], dst.iloc[0]

    moved = float(src["Shipments"]) * shift_pct / 100.0
    delta_per_shipment = float(dst["Avg_Cost"]) - float(src["Avg_Cost"])
    cost_delta = moved * delta_per_shipment  # negative = saving
    return {
        "moved_shipments": moved,
        "from_rate": float(src["Avg_Cost"]),
        "to_rate": float(dst["Avg_Cost"]),
        "cost_delta": cost_delta,
        "saving": max(-cost_delta, 0.0),
        "summary": (
            f"Shifting {shift_pct:.0f}% of {from_carrier}'s volume "
            f"({moved:,.0f} shipments) to {to_carrier} rates "
            f"(${src['Avg_Cost']:.2f} → ${dst['Avg_Cost']:.2f}/shipment) "
            + (f"saves ~${-cost_delta:,.0f} over the data period."
               if cost_delta < 0 else
               f"costs ~${cost_delta:,.0f} more over the data period.")
        ),
    }
