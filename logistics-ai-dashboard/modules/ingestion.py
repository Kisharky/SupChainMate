"""
modules/ingestion.py
Auto-detects and normalises user-uploaded supply chain data files.
Supports: CSV, Excel (.xlsx, .xls), with flexible column detection.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import numpy as np
import pandas as pd


# ── Column Detection Patterns ──────────────────────────────────────────────────

_DATE_PATTERNS = re.compile(
    r"(date|time|timestamp|purchased?|ordered?|created?|placed?|period|day|week|month)",
    re.IGNORECASE,
)
_QTY_PATTERNS = re.compile(
    r"(qty|quantity|units?|volume|amount|count|orders?|items?|demand|sales?)",
    re.IGNORECASE,
)
_STATUS_PATTERNS = re.compile(
    r"(status|state|condition|stage|phase|tracking|delivery_status|order_status)",
    re.IGNORECASE,
)
_LAT_PATTERNS = re.compile(r"(lat|latitude|y_coord)", re.IGNORECASE)
_LON_PATTERNS = re.compile(r"(lon|lng|longitude|x_coord)", re.IGNORECASE,)
_CUSTOMER_PATTERNS = re.compile(r"(customer|client|buyer|recipient|zip|postal|postcode)", re.IGNORECASE)
_COST_PATTERNS = re.compile(r"(cost|price|fee|charge|freight|rate|spend|value|revenue)", re.IGNORECASE)
_LEAD_PATTERNS = re.compile(r"(lead|delivery_time|transit|duration|days?_to|eta|tat)", re.IGNORECASE)
_DELIVERY_DATE_PATTERNS = re.compile(
    r"(delivery|delivered?|arrival|received?|dispatch|shipped?|estimated)", re.IGNORECASE
)


def _read_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel from Streamlit UploadedFile object."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".csv"):
        # Try common encodings
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False)
            except Exception:
                continue
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def _find_col(df: pd.DataFrame, pattern: re.Pattern) -> Optional[str]:
    """Return first column name that matches the regex pattern (case-insensitive)."""
    for col in df.columns:
        if pattern.search(str(col)):
            return col
    return None


def _coerce_datetime(series: pd.Series) -> pd.Series:
    """Best-effort parse a series of mixed date strings to datetime."""
    return pd.to_datetime(series, errors="coerce")


# ── Public Normalisation Functions ─────────────────────────────────────────────

def normalise_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with standardised columns:
        order_date (datetime), quantity (float)
    """
    # Date
    date_col = _find_col(df, _DATE_PATTERNS)
    if date_col is None:
        # fall back: first column that looks like a date
        for col in df.columns:
            sample = df[col].dropna().head(5)
            try:
                pd.to_datetime(sample)
                date_col = col
                break
            except Exception:
                continue
    if date_col is None:
        raise ValueError("Could not detect a date column in the orders file.")

    # Quantity
    qty_col = _find_col(df, _QTY_PATTERNS)
    if qty_col is None:
        # fall back: first numeric column that isn't the date
        for col in df.select_dtypes(include="number").columns:
            if col != date_col:
                qty_col = col
                break
    if qty_col is None:
        # If no quantity column, treat every row as 1 order
        df["quantity"] = 1.0
        qty_col = "quantity"

    result = pd.DataFrame()
    result["order_date"] = _coerce_datetime(df[date_col])
    result["quantity"]   = pd.to_numeric(df[qty_col], errors="coerce").fillna(1.0)
    result = result.dropna(subset=["order_date"])
    return result.sort_values("order_date").reset_index(drop=True)


def normalise_delivery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with:
        order_date (datetime), delivery_date (datetime), status (str), lead_days (float)
    """
    order_col    = _find_col(df, _DATE_PATTERNS)
    delivery_col = _find_col(df, _DELIVERY_DATE_PATTERNS)
    status_col   = _find_col(df, _STATUS_PATTERNS)
    lead_col     = _find_col(df, _LEAD_PATTERNS)

    result = pd.DataFrame()

    if order_col:
        result["order_date"] = _coerce_datetime(df[order_col])
    if delivery_col and delivery_col != order_col:
        result["delivery_date"] = _coerce_datetime(df[delivery_col])
    if status_col:
        result["status"] = df[status_col].astype(str).str.strip().str.title()
    if lead_col:
        result["lead_days"] = pd.to_numeric(df[lead_col], errors="coerce")
    elif "order_date" in result and "delivery_date" in result:
        result["lead_days"] = (
            result["delivery_date"] - result["order_date"]
        ).dt.days.clip(lower=0)

    # Normalise status values to our standard labels
    if "status" in result:
        status_map = {
            r"(?i).*deliver.*":     "Delivered",
            r"(?i).*ship.*":        "Shipped",
            r"(?i).*transit.*":     "Shipped",
            r"(?i).*delay.*":       "Delayed",
            r"(?i).*late.*":        "Delayed",
            r"(?i).*process.*":     "Processing",
            r"(?i).*pending.*":     "Processing",
            r"(?i).*cancel.*":      "Cancelled",
        }
        def _norm_status(s: str) -> str:
            for pat, label in status_map.items():
                if re.match(pat, str(s)):
                    return label
            return "Processing"
        result["status"] = result["status"].apply(_norm_status)

    return result.reset_index(drop=True)


def normalise_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with: lat (float), lon (float), label (str)
    """
    lat_col = _find_col(df, _LAT_PATTERNS)
    lon_col = _find_col(df, _LON_PATTERNS)
    cust_col = _find_col(df, _CUSTOMER_PATTERNS)

    result = pd.DataFrame()

    if lat_col and lon_col:
        result["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        result["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        # If no lat/lon, try to derive from zip/postal using first 5 digits as proxy
        zip_col = _find_col(df, re.compile(r"(zip|postal|postcode)", re.IGNORECASE))
        if zip_col is not None:
            # Pseudo-coordinate from zip prefix (for demo purposes)
            np.random.seed(42)
            n = len(df)
            result["lat"] = np.random.uniform(-33, 5, n)    # Brazil-like for demo
            result["lon"] = np.random.uniform(-73, -35, n)
        else:
            raise ValueError("Could not detect latitude/longitude or zip columns.")

    if cust_col:
        result["label"] = df[cust_col].astype(str)
    else:
        result["label"] = "Node"

    result = result.dropna(subset=["lat", "lon"])
    return result.reset_index(drop=True)


def normalise_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with: cost (float), label (str)
    """
    cost_col = _find_col(df, _COST_PATTERNS)
    if cost_col is None:
        raise ValueError("Could not detect a cost/price column.")

    result = pd.DataFrame()
    result["cost"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)

    # Try to find a label column
    label_candidates = [c for c in df.columns if c != cost_col]
    if label_candidates:
        result["label"] = df[label_candidates[0]].astype(str)
    else:
        result["label"] = "Item"

    return result.reset_index(drop=True)


# ── Daily Demand Aggregation (for Prophet) ─────────────────────────────────────

def orders_to_daily_demand(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate normalised orders into a daily demand DataFrame
    compatible with Prophet (columns: ds, y).
    """
    df = orders_df.copy()
    df["day"] = df["order_date"].dt.normalize()
    daily = (
        df.groupby("day")["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"day": "ds", "quantity": "y"})
    )
    daily["ds"] = pd.to_datetime(daily["ds"])
    # Add external signal placeholder
    np.random.seed(42)
    daily["external_signal"] = np.random.choice(
        [0, 1], size=len(daily), p=[0.95, 0.05]
    )
    daily.loc[daily["external_signal"] == 1, "y"] *= 1.5
    return daily.sort_values("ds").reset_index(drop=True)


# ── Delivery → Simulation DataFrame ───────────────────────────────────────────

def delivery_to_tracking(delivery_df: pd.DataFrame, n: int = None) -> pd.DataFrame:
    """Convert normalised delivery data into our standard tracking simulation format."""
    df = delivery_df.copy()
    if "status" not in df.columns:
        np.random.seed(42)
        choices = ["Delivered", "Shipped", "Delayed", "Processing"]
        weights = [0.6, 0.2, 0.12, 0.08]
        df["status"] = np.random.choice(choices, size=len(df), p=weights)
    if "lead_days" not in df.columns:
        np.random.seed(42)
        df["lead_days"] = np.random.uniform(1, 20, size=len(df))
    return df.reset_index(drop=True)


# ── Summary Helper ─────────────────────────────────────────────────────────────

def detected_columns_summary(raw_df: pd.DataFrame, file_type: str) -> dict:
    """Return a dict describing which columns were auto-detected."""
    summary = {"file_type": file_type, "rows": len(raw_df), "columns": list(raw_df.columns)}
    if file_type == "orders":
        summary["date_col"]   = _find_col(raw_df, _DATE_PATTERNS)
        summary["qty_col"]    = _find_col(raw_df, _QTY_PATTERNS)
    elif file_type == "delivery":
        summary["order_date_col"]    = _find_col(raw_df, _DATE_PATTERNS)
        summary["delivery_date_col"] = _find_col(raw_df, _DELIVERY_DATE_PATTERNS)
        summary["status_col"]        = _find_col(raw_df, _STATUS_PATTERNS)
        summary["lead_col"]          = _find_col(raw_df, _LEAD_PATTERNS)
    elif file_type == "location":
        summary["lat_col"]  = _find_col(raw_df, _LAT_PATTERNS)
        summary["lon_col"]  = _find_col(raw_df, _LON_PATTERNS)
    elif file_type == "cost":
        summary["cost_col"] = _find_col(raw_df, _COST_PATTERNS)
    return summary
