"""
modules/allocation.py
Auto Carrier Allocation — multi-criteria volume allocation across carriers.

Adapts the enterprise "auto carrier allocation" pattern (score carriers on
cost, service, risk, and emissions; allocate volume to the optimum under a
concentration cap) into a deterministic, explainable engine:

  score_c = w_cost·cost_score + w_service·service_score
          + w_emissions·emissions_score + w_reliability·reliability_score

Recommended shares are score-proportional, capped per carrier so the
allocation never recreates single-carrier concentration risk. The blended
cost / on-time / CO2e of the recommended mix is compared against the
current mix, and the proposal routes through the Decision Center.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import config
from modules import trust
from modules.trust import Driver, Impact

_log = config.get_logger(__name__)

MAX_SHARE_PCT = 50.0          # concentration cap per carrier
DEFAULT_WEIGHTS = {"cost": 0.35, "service": 0.35, "emissions": 0.15, "reliability": 0.15}
MIN_SHIFT_PCT = 5.0           # proposals below this total shift are noise


def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalise to 0-100 (higher = better). Constant series → 50."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(50.0, index=series.index)
    scaled = (s - lo) / (hi - lo) * 100
    return (100 - scaled) if invert else scaled


def build_carrier_profiles(scorecard: Optional[pd.DataFrame],
                           carbon_table: Optional[pd.DataFrame] = None
                           ) -> Optional[pd.DataFrame]:
    """One row per carrier: share, cost, on-time, delay risk, CO2e."""
    if scorecard is None or len(scorecard) < 2:
        return None
    prof = scorecard[["Carrier", "Shipments", "On-Time %", "Avg ML Risk %"]].copy()
    if "Avg Cost/Shipment ($)" in scorecard.columns:
        prof["Cost ($)"] = scorecard["Avg Cost/Shipment ($)"]
    prof["Current Share %"] = (prof["Shipments"] / prof["Shipments"].sum() * 100).round(1)
    if carbon_table is not None and "kg CO2e/shipment" in carbon_table.columns:
        prof = prof.merge(carbon_table[["Carrier", "kg CO2e/shipment"]],
                          on="Carrier", how="left")
    return prof


def allocation_scores(profiles: pd.DataFrame,
                      weights: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """
    Score each carrier (0-100) on the weighted criteria and derive the
    recommended share: score-proportional, capped at MAX_SHARE_PCT, with
    the excess redistributed proportionally.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    total_w = sum(w.values()) or 1.0
    w = {k: v / total_w for k, v in w.items()}

    df = profiles.copy()
    df["cost_score"] = (_minmax(df["Cost ($)"], invert=True)
                        if "Cost ($)" in df.columns else 50.0)
    df["service_score"] = _minmax(df["On-Time %"])
    df["reliability_score"] = _minmax(df["Avg ML Risk %"], invert=True)
    df["emissions_score"] = (_minmax(df["kg CO2e/shipment"], invert=True)
                             if "kg CO2e/shipment" in df.columns else 50.0)
    df["Allocation Score"] = (
        w["cost"] * df["cost_score"] + w["service"] * df["service_score"]
        + w["emissions"] * df["emissions_score"]
        + w["reliability"] * df["reliability_score"]).round(1)

    # Score-proportional shares under a per-carrier concentration cap
    raw = df["Allocation Score"].clip(lower=1.0)
    share = raw / raw.sum() * 100
    for _ in range(len(df)):
        over = share > MAX_SHARE_PCT
        if not over.any():
            break
        excess = (share[over] - MAX_SHARE_PCT).sum()
        share[over] = MAX_SHARE_PCT
        under = ~over
        if not under.any() or share[under].sum() == 0:
            break
        share[under] += excess * share[under] / share[under].sum()
    df["Recommended Share %"] = share.round(1)
    df["Shift (pts)"] = (df["Recommended Share %"] - df["Current Share %"]).round(1)
    return df.sort_values("Allocation Score", ascending=False).reset_index(drop=True)


def _blend(df: pd.DataFrame, share_col: str, value_col: str) -> Optional[float]:
    if value_col not in df.columns or df[value_col].isna().all():
        return None
    return float((df[share_col] / 100 * df[value_col]).sum())


def allocation_impact(scored: pd.DataFrame, total_shipments: int) -> dict:
    """Blended metrics of the current vs the recommended mix."""
    out: dict = {"total_shift_pts": float(scored["Shift (pts)"].abs().sum() / 2)}
    for label, col in [("cost", "Cost ($)"), ("on_time", "On-Time %"),
                       ("co2", "kg CO2e/shipment")]:
        cur = _blend(scored, "Current Share %", col)
        rec = _blend(scored, "Recommended Share %", col)
        if cur is not None and rec is not None:
            out[f"{label}_current"], out[f"{label}_recommended"] = cur, rec
    if "cost_current" in out:
        out["savings_total"] = (out["cost_current"] - out["cost_recommended"]) * total_shipments
    return out


def build_recommendation(scored: pd.DataFrame, impact: dict,
                         weights: dict[str, float]) -> Optional[trust.Recommendation]:
    """Package the allocation as a Decision Center recommendation."""
    if impact["total_shift_pts"] < MIN_SHIFT_PCT:
        return None
    gainers = scored[scored["Shift (pts)"] > 1].head(2)
    losers = scored[scored["Shift (pts)"] < -1].tail(2)
    move_txt = ("; ".join(f"{r['Carrier']} {r['Current Share %']:.0f}%→{r['Recommended Share %']:.0f}%"
                          for _, r in pd.concat([gainers, losers]).iterrows()))

    drivers = [Driver("Criteria weights",
                      ", ".join(f"{k} {v:.0%}" for k, v in weights.items())),
               Driver("Concentration cap", f"no carrier above {MAX_SHARE_PCT:.0f}%")]
    if "on_time_current" in impact:
        drivers.append(Driver("Blended on-time",
                              f"{impact['on_time_current']:.1f}% → {impact['on_time_recommended']:.1f}%"))
    if "cost_current" in impact:
        drivers.append(Driver("Blended cost/shipment",
                              f"${impact['cost_current']:.2f} → ${impact['cost_recommended']:.2f}"))
    if "co2_current" in impact:
        drivers.append(Driver("Blended CO2e/shipment",
                              f"{impact['co2_current']:.2f} → {impact['co2_recommended']:.2f} kg"))

    n = int(scored["Shipments"].sum())
    conf, basis = trust._confidence(
        min(1.0, n / 10000), min(1.0, impact["total_shift_pts"] / 40),
        f"{n:,} shipments scored across {len(scored)} carriers; "
        f"{impact['total_shift_pts']:.0f} pts of volume to move")
    return trust.Recommendation(
        source="Carrier Manager", category="CARRIER ALLOCATION",
        title=f"Rebalance carrier mix ({impact['total_shift_pts']:.0f} pts of volume)",
        action=f"Adopt the recommended allocation: {move_txt}.",
        drivers=drivers, confidence=conf, confidence_basis=basis,
        impact=Impact(
            cost_savings_yr=round(impact.get("savings_total", 0), 0) or None,
            service_level_pct=round(impact["on_time_recommended"], 1)
            if "on_time_recommended" in impact else None,
            other=(f"CO2e {impact['co2_current']:.2f}→{impact['co2_recommended']:.2f} kg/shipment"
                   if "co2_current" in impact else None)))
