"""
modules/carbon.py
Carbon Lens — freight CO2e estimates over the network's real route distances.

Method (clearly an estimate, labelled as such in the UI):
    kg CO2e = distance_km x (shipment_weight_kg / 1000) x mode_factor

Mode factors are DEFRA-style road/rail/air/sea averages (kg CO2e per
tonne-km). Distances come from the network's Haversine cluster metrics —
the same numbers the route optimiser uses. No emissions data is invented
per carrier; carriers only differ when their transport mode differs.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# kg CO2e per tonne-km (long-run averages in the style of DEFRA/GLEC factors)
MODE_FACTORS = {
    "road": 0.107,
    "rail": 0.028,
    "air": 1.13,
    "sea": 0.016,
}
DEFAULT_MODE = "road"

# kg CO2e per vehicle-km for an average rigid HGV — used to translate
# route-optimisation km savings into emissions savings.
HGV_KG_PER_KM = 0.85

# Simulated transport modes for the fictional demo carriers (labelled in UI).
DEMO_CARRIER_MODES = {
    "NovaCargo": "road",
    "SwiftLine": "road",
    "BlueFreight": "rail",
    "RapidHaul": "road",
    "TransAmerica XP": "air",
}


def network_avg_distance_km(centroid_stats: Optional[pd.DataFrame]) -> Optional[float]:
    """Customer-weighted average delivery distance from the cluster metrics."""
    if centroid_stats is None:
        return None
    df = centroid_stats.reset_index() if "cluster" not in getattr(centroid_stats, "columns", []) else centroid_stats
    if "avg_dist_km" not in df.columns or not len(df):
        return None
    if "customers" in df.columns and df["customers"].sum() > 0:
        return float((df["avg_dist_km"] * df["customers"]).sum() / df["customers"].sum())
    return float(df["avg_dist_km"].mean())


def shipment_co2_kg(distance_km: float, weight_kg: float, mode: str = DEFAULT_MODE) -> float:
    factor = MODE_FACTORS.get(str(mode).lower(), MODE_FACTORS[DEFAULT_MODE])
    return distance_km * (weight_kg / 1000.0) * factor


def carrier_emissions(
    shipments: pd.DataFrame,
    avg_distance_km: float,
    weight_kg: float = 20.0,
    scorecard: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """
    Per-carrier CO2e estimate. Transport mode comes from a `transport_mode`
    column when present, else road for every carrier (and the table says so).
    Joins avg cost per shipment from the scorecard when available so the UI
    can plot greenest vs cheapest.
    """
    if shipments is None or "carrier" not in shipments.columns or shipments["carrier"].isna().all():
        return None

    df = shipments.copy()
    if "transport_mode" not in df.columns:
        df["transport_mode"] = DEFAULT_MODE

    rows = []
    for (carrier, mode), grp in df.groupby(["carrier", "transport_mode"]):
        per_shipment = shipment_co2_kg(avg_distance_km, weight_kg, mode)
        rows.append({
            "Carrier": carrier,
            "Mode": str(mode).lower(),
            "Shipments": len(grp),
            "kg CO2e/shipment": round(per_shipment, 2),
            "Total tCO2e": round(per_shipment * len(grp) / 1000.0, 1),
        })
    out = pd.DataFrame(rows).sort_values("kg CO2e/shipment").reset_index(drop=True)

    if scorecard is not None and "Avg Cost/Shipment ($)" in scorecard.columns:
        out = out.merge(scorecard[["Carrier", "Avg Cost/Shipment ($)"]], on="Carrier", how="left")
    return out


def zone_emissions(
    centroid_stats: Optional[pd.DataFrame],
    weight_kg: float = 20.0,
    mode: str = DEFAULT_MODE,
) -> Optional[pd.DataFrame]:
    """Per-zone CO2e from each cluster's average delivery distance."""
    if centroid_stats is None:
        return None
    df = centroid_stats.reset_index() if "cluster" not in getattr(centroid_stats, "columns", []) else centroid_stats.copy()
    if "avg_dist_km" not in df.columns or "customers" not in df.columns:
        return None
    out = df[["cluster", "customers", "avg_dist_km"]].copy()
    out["kg CO2e/delivery"] = out["avg_dist_km"].apply(
        lambda d: round(shipment_co2_kg(float(d), weight_kg, mode), 2))
    out["Total tCO2e"] = (out["kg CO2e/delivery"] * out["customers"] / 1000.0).round(2)
    out.columns = ["Zone", "Customers", "Avg Dist (km)", "kg CO2e/delivery", "Total tCO2e"]
    return out.sort_values("Total tCO2e", ascending=False).reset_index(drop=True)


def route_savings_co2(savings_km: float) -> float:
    """Tonnes CO2e avoided by the route-optimisation km savings (HGV factor)."""
    return savings_km * HGV_KG_PER_KM / 1000.0


def carbon_insights(carrier_df: Optional[pd.DataFrame]) -> list[str]:
    """Plain-language takeaways, only where the data actually differs."""
    notes = []
    if carrier_df is None or len(carrier_df) < 2:
        return notes
    if carrier_df["Mode"].nunique() > 1:
        green, dirty = carrier_df.iloc[0], carrier_df.iloc[-1]
        ratio = (dirty["kg CO2e/shipment"] / green["kg CO2e/shipment"]
                 if green["kg CO2e/shipment"] > 0 else 0)
        notes.append(
            f"{dirty['Carrier']} ({dirty['Mode']}) emits ~{ratio:.0f}x more CO2e per shipment "
            f"than {green['Carrier']} ({green['Mode']}) — "
            f"{dirty['kg CO2e/shipment']:.2f} vs {green['kg CO2e/shipment']:.2f} kg."
        )
        if "Avg Cost/Shipment ($)" in carrier_df.columns:
            g_cost = carrier_df.iloc[0].get("Avg Cost/Shipment ($)")
            d_cost = carrier_df.iloc[-1].get("Avg Cost/Shipment ($)")
            if pd.notna(g_cost) and pd.notna(d_cost) and g_cost < d_cost:
                notes.append(
                    f"{carrier_df.iloc[0]['Carrier']} is both the greenest AND the cheapest "
                    f"(${g_cost:.2f} vs ${d_cost:.2f}/shipment) — shifting volume wins twice."
                )
    else:
        notes.append(
            "All carriers are modelled as the same transport mode, so per-shipment emissions "
            "are equal — add a transport_mode column to your delivery file to differentiate."
        )
    return notes
