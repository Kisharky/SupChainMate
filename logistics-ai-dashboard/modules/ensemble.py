"""
modules/ensemble.py
Ensemble demand forecasting — a model tournament alongside Prophet.

Candidates (all CPU, from existing dependencies):
  - LightGBM regressor (falls back out if lightgbm is missing)
  - Random Forest
  - Gradient Boosting
  - Ridge regression
  - Ensemble mean of the above
  - Prophet (scored on the same holdout when its forecast is supplied)

Method: the last `holdout_days` of the daily series are held out; every
model trains on the rest and is scored on the holdout (MAPE + RMSE). The
champion (lowest MAPE) then produces a recursive multi-step forecast for
the requested horizon. Honest by construction: scores come from a real
backtest, not in-sample fit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

_LAGS = [1, 7, 14]
_ROLLS = [7, 28]
_MIN_HISTORY = 90  # days needed before the tournament makes sense


def _make_features(y: pd.Series, ds: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=y.index)
    for lag in _LAGS:
        X[f"lag_{lag}"] = y.shift(lag)
    for w in _ROLLS:
        X[f"roll_{w}"] = y.shift(1).rolling(w).mean()
    dt = pd.to_datetime(ds)
    X["dow"] = dt.dt.dayofweek.values
    X["month"] = dt.dt.month.values
    X["day"] = dt.dt.day.values
    return X


def _models() -> dict:
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8,
                                               random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                       learning_rate=0.05, random_state=42),
        "Ridge": Ridge(alpha=1.0),
    }
    if _HAS_LGBM:
        models["LightGBM"] = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                               max_depth=5, random_state=42, verbose=-1)
    return models


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def run_tournament(
    daily_df: pd.DataFrame,
    prophet_forecast: Optional[pd.DataFrame] = None,
    holdout_days: int = 28,
    horizon_days: int = 7,
) -> Optional[dict]:
    """
    Backtest all candidates on the last `holdout_days`, crown a champion,
    and produce its recursive forecast for `horizon_days`.

    Returns {leaderboard, champion, champion_mape, prophet_mape, forecast,
    holdout (DataFrame with per-model holdout predictions)} or None when
    the series is too short.
    """
    df = daily_df[["ds", "y"]].dropna().sort_values("ds").reset_index(drop=True)

    # Trim the trailing tail-off: datasets often end with a few partial days
    # (e.g. the Olist extract). Backtesting against near-zero actuals makes
    # MAPE meaningless, so drop trailing days below 10% of the median.
    med = float(df["y"].median())
    if med > 0:
        healthy = df["y"] >= 0.1 * med
        if healthy.any():
            df = df.iloc[: healthy[healthy].index[-1] + 1].reset_index(drop=True)

    if len(df) < _MIN_HISTORY + holdout_days:
        return None

    y = df["y"].astype(float)
    X = _make_features(y, df["ds"])
    valid = X.notna().all(axis=1)
    X, y_v, ds_v = X[valid], y[valid], df["ds"][valid]
    if len(X) < holdout_days + 30:
        return None

    X_train, X_test = X.iloc[:-holdout_days], X.iloc[-holdout_days:]
    y_train, y_test = y_v.iloc[:-holdout_days], y_v.iloc[-holdout_days:]
    ds_test = ds_v.iloc[-holdout_days:]

    rows, fitted, preds = [], {}, {}
    for name, model in _models().items():
        model.fit(X_train, y_train)
        p = np.clip(model.predict(X_test), 0, None)
        fitted[name], preds[name] = model, p
        rows.append({"Model": name, "MAPE %": round(_mape(y_test.values, p), 2),
                     "RMSE": round(_rmse(y_test.values, p), 1)})

    # Ensemble mean of the ML candidates
    ens = np.mean(list(preds.values()), axis=0)
    preds["Ensemble (mean)"] = ens
    rows.append({"Model": "Ensemble (mean)", "MAPE %": round(_mape(y_test.values, ens), 2),
                 "RMSE": round(_rmse(y_test.values, ens), 1)})

    # Prophet scored on the same holdout window when supplied
    prophet_mape = None
    if prophet_forecast is not None:
        pf = prophet_forecast[["ds", "yhat"]].merge(
            pd.DataFrame({"ds": ds_test.values, "y": y_test.values}), on="ds", how="inner")
        if len(pf) >= holdout_days // 2:
            pp = np.clip(pf["yhat"].values, 0, None)
            prophet_mape = round(_mape(pf["y"].values, pp), 2)
            preds["Prophet"] = pp
            rows.append({"Model": "Prophet", "MAPE %": prophet_mape,
                         "RMSE": round(_rmse(pf["y"].values, pp), 1)})

    leaderboard = pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True)
    champion = str(leaderboard.iloc[0]["Model"])
    champion_mape = float(leaderboard.iloc[0]["MAPE %"])

    # ── Recursive future forecast with the champion ───────────────────────────
    # Prophet's own future forecast already exists in the app; the tournament
    # produces one for the champion ML model (or the ensemble).
    def _recursive(models_to_use: list) -> pd.DataFrame:
        hist = y_v.copy().reset_index(drop=True)
        dates = list(pd.to_datetime(df["ds"][valid]).reset_index(drop=True))
        out = []
        for step in range(horizon_days):
            next_date = dates[-1] + pd.Timedelta(days=1)
            feat = {}
            for lag in _LAGS:
                feat[f"lag_{lag}"] = hist.iloc[-lag]
            for w in _ROLLS:
                feat[f"roll_{w}"] = hist.iloc[-w:].mean()
            feat["dow"], feat["month"], feat["day"] = next_date.dayofweek, next_date.month, next_date.day
            row = pd.DataFrame([feat])[X_train.columns]
            val = float(np.mean([np.clip(m.predict(row)[0], 0, None) for m in models_to_use]))
            out.append({"ds": next_date, "yhat": round(val, 1)})
            hist = pd.concat([hist, pd.Series([val])], ignore_index=True)
            dates.append(next_date)
        return pd.DataFrame(out)

    if champion == "Ensemble (mean)":
        future = _recursive(list(fitted.values()))
    elif champion in fitted:
        future = _recursive([fitted[champion]])
    else:  # Prophet won — its future forecast lives in the main app already
        future = None

    holdout = pd.DataFrame({"ds": ds_test.values, "actual": y_test.values})
    for name, p in preds.items():
        if len(p) == len(holdout):
            holdout[name] = np.round(p, 1)

    return {
        "leaderboard": leaderboard,
        "champion": champion,
        "champion_mape": champion_mape,
        "prophet_mape": prophet_mape,
        "forecast": future,
        "holdout": holdout,
        "holdout_days": holdout_days,
    }
