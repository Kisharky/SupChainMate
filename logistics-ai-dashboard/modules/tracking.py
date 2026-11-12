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


# Date columns, Olist names first, then the normalised upload names. Shared
# with the control tower so both read the same dates.
ORDER_DATE_COLS = ["order_purchase_timestamp", "order_date", "date", "ds"]
ACTUAL_DATE_COLS = ["order_delivered_customer_date", "delivery_date", "delivered_date"]
PROMISED_DATE_COLS = ["order_estimated_delivery_date", "estimated_date", "promised_date", "eta"]

# Minimum labelled deliveries needed to train on the real outcome.
_MIN_REAL_LABELS = 50


def _first_col(df: pd.DataFrame, candidates: list[str]):
    return next((c for c in candidates if c in df.columns), None)


def delay_labels(df: pd.DataFrame) -> pd.Series | None:
    """Real delivery outcome: 1 if delivered after the promised date, else 0.

    NaN where either date is missing (still in flight, cancelled). Returns
    None when the data has no actual/promised date columns at all.
    """
    actual_col = _first_col(df, ACTUAL_DATE_COLS)
    promised_col = _first_col(df, PROMISED_DATE_COLS)
    if not (actual_col and promised_col):
        return None
    actual = pd.to_datetime(df[actual_col], errors="coerce")
    promised = pd.to_datetime(df[promised_col], errors="coerce")
    labels = (actual > promised).astype(float)
    labels[actual.isna() | promised.isna()] = np.nan
    return labels


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
    date_col = _first_col(df, ORDER_DATE_COLS)

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
    # Lead time: an explicit column if present, else the lead time promised
    # at purchase (promised date − order date), else a constant. Never random.
    promised_col = _first_col(df, PROMISED_DATE_COLS)
    if "lead_days" in df.columns:
        out["lead_days"] = pd.to_numeric(df["lead_days"], errors="coerce").fillna(7)
    elif date_col and promised_col:
        promised = pd.to_datetime(df[promised_col], errors="coerce")
        out["lead_days"] = (promised - pd.to_datetime(df[date_col], errors="coerce")).dt.days
        out["lead_days"] = out["lead_days"].fillna(out["lead_days"].median()).fillna(7)
    else:
        out["lead_days"] = 7.0

    out["lead_days_sq"]  = out["lead_days"] ** 2          # non-linear signal
    out["long_lead"]     = (out["lead_days"] > 14).astype(int)

    return out.astype(float)


def train_delay_model(df: pd.DataFrame):
    """
    Train a LightGBM (or RandomForest fallback) classifier to predict delay risk.
    Returns (model, X_test, y_test).

    The label is the real outcome — delivered after the promised date — on
    every shipment that has both dates. Only when the data carries no such
    dates does it fall back to ``status == "Delayed"``. The source is recorded
    on the model as ``label_source_``.
    """
    labels = delay_labels(df)
    if (labels is not None and labels.notna().sum() >= _MIN_REAL_LABELS
            and labels.dropna().nunique() == 2):
        known = labels.notna()
        X = _engineer_features(df[known])
        y = labels[known].astype(int)
        label_source = "delivered_vs_promised"
    else:
        if "status" not in df.columns:
            raise ValueError("No delivery dates or status column to learn delay risk from.")
        X = _engineer_features(df)
        y = (df["status"] == "Delayed").astype(int)
        label_source = "status"

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
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
    model.label_source_ = label_source
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
