"""Network / delivery-style metrics and light segmentation (same orders dataset)."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _with_lead_times(orders_df: pd.DataFrame) -> pd.DataFrame:
    df = orders_df.copy()
    df["purchase"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["delivered"] = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    df["estimated"] = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")
    df = df.dropna(subset=["delivered"])
    if df.empty:
        return df
    df["lead_days"] = (df["delivered"] - df["purchase"]).dt.total_seconds() / 86400.0
    df["lead_days"] = df["lead_days"].clip(lower=0.0, upper=90.0)
    df["on_time"] = (df["delivered"] <= df["estimated"]).astype(int)
    return df


def network_summary(orders_df: pd.DataFrame) -> dict | None:
    feat = _with_lead_times(orders_df)
    if feat.empty:
        return None
    return {
        "avg_lead_days": float(feat["lead_days"].mean()),
        "median_lead_days": float(feat["lead_days"].median()),
        "on_time_pct": float(feat["on_time"].mean() * 100.0),
        "n_delivered_observed": int(len(feat)),
    }


def daily_network_metrics(orders_df: pd.DataFrame) -> pd.DataFrame:
    feat = _with_lead_times(orders_df)
    if feat.empty:
        return pd.DataFrame(columns=["day", "orders", "median_lead", "on_time_rate"])
    feat["day"] = feat["purchase"].dt.normalize()
    agg = feat.groupby("day", as_index=False).agg(
        orders=("order_id", "count"),
        median_lead=("lead_days", "median"),
        on_time_rate=("on_time", "mean"),
    )
    agg["on_time_rate"] = agg["on_time_rate"] * 100.0
    return agg.sort_values("day")


def cluster_operating_days(daily_metrics: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    if daily_metrics.empty or len(daily_metrics) < k:
        out = daily_metrics.copy()
        out["segment"] = "—"
        return out
    X = daily_metrics[["orders", "median_lead"]].values
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)
    out = daily_metrics.copy()
    out["cluster"] = labels
    volume_rank = out.groupby("cluster")["orders"].mean().sort_values(ascending=False)
    names = ["High load", "Balanced", "Low load"]
    label_by_rank = {
        cl: names[i] if i < len(names) else f"Cluster {cl}"
        for i, cl in enumerate(volume_rank.index)
    }
    out["segment"] = out["cluster"].map(label_by_rank).fillna("—")
    return out
