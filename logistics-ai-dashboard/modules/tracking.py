"""
modules/tracking.py
Operational flow tracking + delay risk model.

Upgrade: LightGBM classifier with engineered features for delay prediction.
Falls back to RandomForest if LightGBM is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    _HAS_LGBM = False


def simulate_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate operational order statuses for demo purposes."""
    df = df.copy()
    statuses = ["Processing", "Shipped", "Delivered", "Delayed"]
    df["status"] = np.random.choice(statuses, size=len(df), p=[0.2, 0.4, 0.3, 0.1])
    return df


def get_status_counts(df: pd.DataFrame) -> pd.Series:
    return df["status"].value_counts()


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer richer features for delay prediction from any order dataframe.
    Handles both Olist-style (order_purchase_timestamp) and generic date columns.
    """
    out = pd.DataFrame(index=df.index)

    # ── Timestamp features ───────────────────────────────────────────────────
    date_col = None
    for col in ["order_purchase_timestamp", "order_date", "date", "ds"]:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        out["hour"]          = dt.dt.hour.fillna(12)
        out["day_of_week"]   = dt.dt.dayofweek.fillna(2)      # 0=Mon, 6=Sun
        out["day_of_month"]  = dt.dt.day.fillna(15)
        out["month"]         = dt.dt.month.fillna(6)
        out["is_weekend"]    = (out["day_of_week"] >= 5).astype(int)
        out["is_month_end"]  = (out["day_of_month"] >= 28).astype(int)
    else:
        out["hour"]         = 12
        out["day_of_week"]  = 2
        out["day_of_month"] = 15
        out["month"]        = 6
        out["is_weekend"]   = 0
        out["is_month_end"] = 0

    # ── Lead time features ───────────────────────────────────────────────────
    if "lead_days" in df.columns:
        out["lead_days"] = pd.to_numeric(df["lead_days"], errors="coerce").fillna(7)
    else:
        np.random.seed(42)
        out["lead_days"] = np.random.uniform(1, 20, size=len(df))

    out["lead_days_sq"]  = out["lead_days"] ** 2          # non-linear signal
    out["long_lead"]     = (out["lead_days"] > 14).astype(int)

    return out.astype(float)


def train_delay_model(df: pd.DataFrame):
    """
    Train a LightGBM (or RandomForest fallback) classifier to predict delay risk.
    Returns (model, X_test, y_test).
    """
    df = df.copy()
    df["is_delayed"] = (df["status"] == "Delayed").astype(int)

    X = _engineer_features(df)
    y = df["is_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if _HAS_LGBM:
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,          # suppress LightGBM output
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    return model, X_test, y_test


def predict_delay_risk(model, df: pd.DataFrame) -> np.ndarray:
    """
    Run delay risk prediction on an arbitrary subset of tracking data.
    Returns predicted probabilities of delay (0–1).
    """
    X = _engineer_features(df)
    if _HAS_LGBM and hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def model_backend() -> str:
    """Return a string identifying the active backend."""
    return "LightGBM" if _HAS_LGBM else "RandomForest (LightGBM not installed)"
