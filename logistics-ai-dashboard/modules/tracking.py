"""Operational flow tracking + delay risk model (demo simulation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def simulate_tracking(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    statuses = ["Processing", "Shipped", "Delivered", "Delayed"]

    df["status"] = np.random.choice(statuses, size=len(df), p=[0.2, 0.4, 0.3, 0.1])

    return df


def get_status_counts(df: pd.DataFrame) -> pd.Series:
    return df["status"].value_counts()


def train_delay_model(df: pd.DataFrame):
    df = df.copy()

    # Create target
    df["is_delayed"] = (df["status"] == "Delayed").astype(int)

    # Simple features (expand later)
    df["order_hour"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.hour
    df["order_day"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.day

    X = df[["order_hour", "order_day"]]
    y = df["is_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    return model, X_test, y_test
