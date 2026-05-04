"""Demand forecasting from Olist order timestamps (Prophet)."""

from __future__ import annotations

import pandas as pd
from prophet import Prophet

DEFAULT_DATA_PATH = "data/olist_orders.csv"


def load_orders(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df


def daily_demand(orders_df: pd.DataFrame) -> pd.DataFrame:
    g = (
        orders_df.groupby(orders_df["order_purchase_timestamp"].dt.normalize())
        .size()
        .reset_index()
    )
    g.columns = ["ds", "y"]
    g["ds"] = pd.to_datetime(g["ds"])
    # Keep demand as float because external signal scaling applies non-integer multipliers.
    g["y"] = pd.to_numeric(g["y"], errors="coerce").astype("float64")
    
    # Add synthetic external signal (e.g., Marketing Event / Holiday)
    import numpy as np
    np.random.seed(42)
    g['external_signal'] = np.random.choice([0, 1], size=len(g), p=[0.95, 0.05])
    # Amplify historical demand when signal is present so the model learns it.
    # Use vectorized arithmetic instead of masked assignment to avoid dtype setitem issues on pandas 3.x.
    g["y"] = g["y"] * (1.0 + 0.5 * g["external_signal"].astype(float))
    
    return g.sort_values("ds").reset_index(drop=True)


def fit_prophet_model(daily_df: pd.DataFrame) -> Prophet:
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.add_regressor('external_signal')
    model.fit(daily_df)
    return model


def run_forecast(daily_df: pd.DataFrame, horizon_days: int = 7) -> tuple[Prophet, pd.DataFrame]:
    model = fit_prophet_model(daily_df)
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)
    return model, forecast


def forecast_insights(
    forecast: pd.DataFrame,
    daily_df: pd.DataFrame,
    horizon_days: int = 7,
) -> dict:
    future_only = forecast.tail(horizon_days)
    next_week_total = int(future_only["yhat"].sum())
    hist = daily_df["y"].astype(float)
    p90 = float(hist.quantile(0.9)) if len(hist) else 0.0

    next_5 = forecast["yhat"].tail(horizon_days).head(min(5, horizon_days))
    peak_next_5 = float(next_5.max()) if len(next_5) else 0.0

    if p90 <= 0:
        risk_label = "Unknown — insufficient history"
        risk_short = "—"
    elif peak_next_5 > p90 * 1.15:
        risk_label = "Elevated — forecast peak is above typical high-volume days"
        risk_short = "Elevated in 5d"
    elif peak_next_5 > p90:
        risk_label = "Moderate — demand approaching historical upper band"
        risk_short = "Moderate in 5d"
    else:
        risk_label = "Low — forecast within normal daily range"
        risk_short = "Low"

    y = daily_df.sort_values("ds")
    last7 = float(y["y"].tail(7).mean()) if len(y) else 0.0
    prev7 = float(y["y"].iloc[-14:-7].mean()) if len(y) >= 14 else None
    if prev7 is not None and prev7 > 0:
        pct_change = (last7 - prev7) / prev7 * 100.0
    else:
        pct_change = 0.0

    return {
        "next_week_total": next_week_total,
        "stockout_risk_detail": risk_label,
        "stockout_risk_short": risk_short,
        "demand_pct_change_vs_prior_week": pct_change,
        "historical_p90_daily": p90,
    }
