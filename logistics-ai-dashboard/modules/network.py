"""
modules/network.py
Real-world customer geography via Olist geolocation dataset.
Includes:
  - Actual lat/lon join from olist_geolocation_dataset.csv (median per zip prefix)
  - KMeans network clustering with dynamic n_clusters
  - Haversine centroid distance metric per cluster
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────────────────────────
GEOLOCATION_PATH = "data/olist_geolocation_dataset.csv"
FALLBACK_GEO_URL = (
    "https://raw.githubusercontent.com/olist/work-at-olist-data/master/"
    "datasets/olist_geolocation_dataset.csv"
)


# ── Haversine Distance ─────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points (decimal degrees)."""
    R = 6371.0  # Earth radius km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def cluster_centroid_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster, compute:
      - centroid_lat, centroid_lon  — the geographic centre of the cluster
      - avg_dist_km                 — average Haversine distance from centroid (km)
      - max_dist_km                 — max distance (service radius proxy)

    Returns a DataFrame indexed by cluster label.
    """
    records = []
    for cluster_id, group in df.groupby("cluster"):
        c_lat = group["lat"].mean()
        c_lon = group["lon"].mean()
        dists = group.apply(
            lambda r: haversine_km(c_lat, c_lon, r["lat"], r["lon"]), axis=1
        )
        records.append({
            "cluster":        cluster_id,
            "centroid_lat":   round(c_lat, 4),
            "centroid_lon":   round(c_lon, 4),
            "customers":      len(group),
            "avg_dist_km":    round(dists.mean(), 1),
            "max_dist_km":    round(dists.max(), 1),
            "efficiency_score": round(100 - min(dists.mean() / 10, 100), 1),
        })
    return pd.DataFrame(records).set_index("cluster")


# ── Geolocation Lookup Table ───────────────────────────────────────────────────

def _load_geo_lookup() -> Optional[pd.DataFrame]:
    """
    Load or download the Olist geolocation CSV.
    Returns a DataFrame with columns [zip_prefix, lat, lon]
    aggregated to median lat/lon per 5-digit zip prefix.
    Returns None if unavailable.
    """
    path = GEOLOCATION_PATH
    if not os.path.exists(path):
        try:
            import urllib.request
            print(f"[network] Downloading geolocation dataset from GitHub…")
            os.makedirs("data", exist_ok=True)
            urllib.request.urlretrieve(FALLBACK_GEO_URL, path)
        except Exception as e:
            print(f"[network] Could not download geolocation dataset: {e}")
            return None

    geo = pd.read_csv(path, usecols=[
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
    ])
    geo.columns = ["zip_prefix", "lat", "lon"]
    # Clip to Brazil bounding box to remove outliers
    geo = geo[
        (geo["lat"].between(-34, 6)) &
        (geo["lon"].between(-74, -28))
    ]
    lookup = (
        geo.groupby("zip_prefix")[["lat", "lon"]]
        .median()
        .reset_index()
    )
    return lookup


# ── Public API ─────────────────────────────────────────────────────────────────

def prepare_customer_data(
    customers_df: pd.DataFrame,
    geo_lookup: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Join customer zip prefixes to real lat/lon from the Olist geolocation dataset.
    Falls back to synthetic coordinates if geolocation data is unavailable.
    """
    df = customers_df.dropna().copy()
    z = pd.to_numeric(df["customer_zip_code_prefix"], errors="coerce")
    df = df.loc[z.notna()].copy()
    df["zip_prefix"] = z.loc[z.notna()].astype(int)

    # Attempt real geolocation join
    if geo_lookup is None:
        geo_lookup = _load_geo_lookup()

    if geo_lookup is not None:
        merged = df.merge(geo_lookup, on="zip_prefix", how="left")
        valid = merged.dropna(subset=["lat", "lon"])
        if len(valid) > 100:
            return valid[["lat", "lon"]].reset_index(drop=True)

    # Fallback: reproducible synthetic coordinates in Brazil bounds
    np.random.seed(42)
    n = len(df)
    return pd.DataFrame({
        "lat": np.random.uniform(-33.0, 5.0, n),
        "lon": np.random.uniform(-73.0, -35.0, n),
    })


def run_clustering(
    df: pd.DataFrame,
    n_clusters: int = 5,
) -> pd.DataFrame:
    """
    Run KMeans on lat/lon. Adds 'cluster' column.
    Automatically caps n_clusters to the number of unique points.
    """
    out = df.copy()
    n   = min(n_clusters, len(out))
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    out["cluster"] = kmeans.fit_predict(out[["lat", "lon"]])
    return out


def isolation_forest_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use Isolation Forest to score each geographic node for anomalousness
    based on its spatial density relative to all other nodes.

    Anomalous = isolated = higher risk (fewer neighbouring delivery nodes).

    Adds columns:
      - risk_score  : 0–100, higher = more anomalous / isolated
      - risk_level  : 'Safe' | 'Warning' | 'Critical'
      - if_score    : raw Isolation Forest decision score (negative = more anomalous)

    The Isolation Forest is trained on [lat, lon] coordinates.
    Points in dense urban clusters score low (Safe).
    Points in sparse, isolated zones score high (Critical).
    """
    out = df.copy()

    # Scale coordinates so lat and lon contribute equally
    scaler = StandardScaler()
    coords = scaler.fit_transform(out[["lat", "lon"]])

    # contamination = expected fraction of anomalous / isolated nodes
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.10,   # ~10% of nodes flagged as high-risk
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(coords)

    # decision_function: more negative = more anomalous
    raw_scores = iso.decision_function(coords)

    # Invert and normalise to 0–100 (high = risky)
    inverted = -raw_scores
    lo, hi   = inverted.min(), inverted.max()
    if hi > lo:
        normalised = (inverted - lo) / (hi - lo) * 100
    else:
        normalised = np.full(len(out), 50.0)

    out["if_score"]   = raw_scores
    out["risk_score"] = normalised.round(1)
    out["risk_level"] = pd.cut(
        out["risk_score"],
        bins=[-1, 70, 90, 100],
        labels=["Safe", "Warning", "Critical"],
    )
    return out


def get_geo_lookup() -> Optional[pd.DataFrame]:
    """Public accessor so app.py can pass the lookup to prepare_customer_data."""
    return _load_geo_lookup()
