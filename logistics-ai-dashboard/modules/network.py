"""Customer geo proxy + delivery-zone-style clustering (demo coordinates from ZIP prefix)."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans


def prepare_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()
    z = pd.to_numeric(df["customer_zip_code_prefix"], errors="coerce")
    df = df.loc[z.notna()]
    z = z.loc[z.notna()].astype(int)

    # Fake coordinates (for demo) — not true lat/long
    df["lat"] = (z % 90).astype(float)
    df["lon"] = (z % 180).astype(float)
    return df[["lat", "lon"]]


def run_clustering(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    out = df.copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    out["cluster"] = kmeans.fit_predict(out[["lat", "lon"]])
    return out
