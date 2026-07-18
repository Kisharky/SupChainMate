"""
modules/factors.py
External Factor Engine — market and calendar signals for factor-aware
forecasting (Bloomberg-terminal-style inputs, PostHog-style analytics import).

Factor sources (all keyless; every one optional and skipped on failure):
  - Calendar   : holidays (offline via `holidays`), weekend, month-end, payday
  - FX         : frankfurter.app historical rates (ECB data)
  - Oil        : stooq.com Brent crude daily closes (CSV)
  - Weather    : open-meteo.com historical archive (temp, precipitation)
  - Analytics  : uploaded PostHog / GA daily-events export (date,count) —
                 a leading indicator of demand

The factor frame merges into the ensemble tournament's feature set, so each
factor's value is measured the honest way: did it reduce MAPE on a real
holdout backtest?
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
import requests

_TIMEOUT = 15
FACTOR_COLUMNS = ["is_holiday", "is_weekend", "is_month_end", "is_payday",
                  "fx_rate", "oil_usd", "temp_c", "precip_mm", "web_events"]


# ── Offline calendar factors ───────────────────────────────────────────────────

def calendar_factors(ds: pd.Series, country: str = "BR") -> pd.DataFrame:
    """Holiday / weekend / month-end / payday flags. Fully offline."""
    dt = pd.to_datetime(ds)
    out = pd.DataFrame({"ds": dt})
    try:
        import holidays as _hol
        years = sorted(dt.dt.year.unique().tolist())
        cal = _hol.country_holidays(country, years=years)
        out["is_holiday"] = dt.dt.date.map(lambda d: 1 if d in cal else 0).astype(int)
    except Exception:
        out["is_holiday"] = 0
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    out["is_month_end"] = (dt.dt.day >= 28).astype(int)
    out["is_payday"] = dt.dt.day.isin([1, 15]).astype(int)
    return out


# ── Online market factors (keyless, graceful) ──────────────────────────────────

def fx_factor(start: str, end: str, base: str = "USD",
              symbol: str = "BRL") -> tuple[Optional[pd.DataFrame], str]:
    """Daily FX rate from frankfurter.app (ECB). Weekends forward-filled."""
    try:
        r = requests.get(f"https://api.frankfurter.app/{start}..{end}",
                         params={"from": base, "to": symbol}, timeout=_TIMEOUT)
        r.raise_for_status()
        rates = r.json().get("rates", {})
        if not rates:
            return None, "FX: no data returned"
        df = pd.DataFrame(
            [{"ds": pd.Timestamp(d), "fx_rate": v.get(symbol)} for d, v in rates.items()]
        ).dropna().sort_values("ds")
        return df, f"FX {base}/{symbol}: {len(df)} days (frankfurter.app)"
    except Exception as e:
        return None, f"FX unavailable: {type(e).__name__}"


def oil_factor(start: str, end: str) -> tuple[Optional[pd.DataFrame], str]:
    """Brent crude daily closes from stooq.com public CSV."""
    try:
        r = requests.get("https://stooq.com/q/d/l/",
                         params={"s": "cb.f", "i": "d"}, timeout=_TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "Close" not in df.columns or "Date" not in df.columns:
            return None, "Oil: unexpected data shape"
        df = df.rename(columns={"Date": "ds", "Close": "oil_usd"})[["ds", "oil_usd"]]
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df[(df["ds"] >= start) & (df["ds"] <= end)].dropna()
        if not len(df):
            return None, "Oil: no rows in the data window"
        return df, f"Brent crude: {len(df)} days (stooq.com)"
    except Exception as e:
        return None, f"Oil unavailable: {type(e).__name__}"


def weather_factor(start: str, end: str, lat: float,
                   lon: float) -> tuple[Optional[pd.DataFrame], str]:
    """Daily mean temperature + precipitation from the Open-Meteo archive."""
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": round(lat, 3), "longitude": round(lon, 3),
                    "start_date": start, "end_date": end,
                    "daily": "temperature_2m_mean,precipitation_sum",
                    "timezone": "UTC"},
            timeout=_TIMEOUT)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        if not daily.get("time"):
            return None, "Weather: no data returned"
        df = pd.DataFrame({
            "ds": pd.to_datetime(daily["time"]),
            "temp_c": daily.get("temperature_2m_mean"),
            "precip_mm": daily.get("precipitation_sum"),
        }).dropna(subset=["ds"])
        return df, f"Weather @({lat:.1f},{lon:.1f}): {len(df)} days (open-meteo.com)"
    except Exception as e:
        return None, f"Weather unavailable: {type(e).__name__}"


# ── Analytics import (PostHog / GA style: date,count) ──────────────────────────

def parse_analytics_csv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Normalise a daily-events export (PostHog insights CSV, GA, or any
    date,count file) into (ds, web_events).
    """
    if df is None or not len(df):
        return None
    date_col = count_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if date_col is None and any(k in cl for k in ("date", "day", "time", "week")):
            date_col = c
        elif count_col is None and pd.api.types.is_numeric_dtype(df[c]):
            count_col = c
    if count_col is None:
        for c in df.columns:
            if c != date_col:
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().mean() > 0.8:
                    count_col = c
                    break
    if date_col is None or count_col is None:
        return None
    out = pd.DataFrame({
        "ds": pd.to_datetime(df[date_col], errors="coerce"),
        "web_events": pd.to_numeric(df[count_col], errors="coerce"),
    }).dropna()
    if not len(out):
        return None
    return out.groupby("ds", as_index=False)["web_events"].sum()


# ── Assembly ───────────────────────────────────────────────────────────────────

def build_factor_frame(
    daily_df: pd.DataFrame,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    country: str = "BR",
    fx_symbol: str = "BRL",
    analytics_df: Optional[pd.DataFrame] = None,
    enable_online: bool = True,
) -> dict:
    """
    Build the factor frame covering the demand series' date range.
    Returns {factors (DataFrame ds + factor cols), sources, errors}.
    """
    ds = pd.to_datetime(daily_df["ds"])
    start, end = ds.min().strftime("%Y-%m-%d"), ds.max().strftime("%Y-%m-%d")
    base = pd.DataFrame({"ds": pd.date_range(ds.min(), ds.max(), freq="D")})

    frame = base.merge(calendar_factors(base["ds"], country), on="ds", how="left")
    sources = [f"Calendar ({country}): holidays, weekend, month-end, payday (offline)"]
    errors: list[str] = []

    if enable_online:
        for fetch, args in [(fx_factor, (start, end, "USD", fx_symbol)),
                            (oil_factor, (start, end))]:
            df, msg = fetch(*args)
            if df is not None:
                frame = frame.merge(df, on="ds", how="left")
                sources.append(msg)
            else:
                errors.append(msg)
        if lat is not None and lon is not None:
            df, msg = weather_factor(start, end, lat, lon)
            if df is not None:
                frame = frame.merge(df, on="ds", how="left")
                sources.append(msg)
            else:
                errors.append(msg)

    if analytics_df is not None:
        parsed = parse_analytics_csv(analytics_df)
        if parsed is not None:
            frame = frame.merge(parsed, on="ds", how="left")
            sources.append(f"Web analytics: {len(parsed)} days (uploaded export)")
        else:
            errors.append("Analytics: could not find date + count columns")

    # Market data has weekend/holiday gaps — forward-fill, then back-fill starts
    for col in frame.columns:
        if col != "ds":
            frame[col] = frame[col].ffill().bfill()

    return {"factors": frame, "sources": sources, "errors": errors}


def factor_correlations(daily_df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation of each factor with demand — same-day and with the
    factor leading demand by 7 days (leading-indicator check).
    """
    merged = daily_df[["ds", "y"]].merge(factors, on="ds", how="inner")
    rows = []
    for col in factors.columns:
        if col == "ds" or col not in merged.columns:
            continue
        series = merged[col]
        if series.nunique() < 2:
            continue
        same = float(merged["y"].corr(series))
        lead = float(merged["y"].iloc[7:].reset_index(drop=True).corr(
            series.iloc[:-7].reset_index(drop=True))) if len(merged) > 40 else np.nan
        rows.append({"Factor": col, "Corr (same-day)": round(same, 3),
                     "Corr (leads by 7d)": round(lead, 3) if not np.isnan(lead) else None})
    if not rows:
        return pd.DataFrame(columns=["Factor", "Corr (same-day)", "Corr (leads by 7d)"])
    out = pd.DataFrame(rows)
    return out.reindex(out["Corr (same-day)"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def latest_readings(factors: pd.DataFrame) -> list[dict]:
    """Ticker-strip readings: the latest value of each numeric market factor."""
    if factors is None or not len(factors):
        return []
    last = factors.iloc[-1]
    fmt = {
        "fx_rate": ("FX RATE", "{:.3f}"),
        "oil_usd": ("BRENT $", "{:.1f}"),
        "temp_c": ("TEMP °C", "{:.1f}"),
        "precip_mm": ("PRECIP MM", "{:.1f}"),
        "web_events": ("WEB EVENTS", "{:,.0f}"),
    }
    out = []
    for col, (label, f) in fmt.items():
        if col in factors.columns and pd.notna(last[col]):
            prev = factors[col].iloc[-8] if len(factors) > 8 else last[col]
            delta = float(last[col]) - float(prev)
            out.append({"label": label, "value": f.format(float(last[col])),
                        "delta": delta})
    return out
