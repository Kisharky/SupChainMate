"""
api/data_source.py — the centralized operational data-access layer.

Every analytical module reads customers/orders through here instead of touching
the bundled CSVs directly. When a company has imported its own ERP/CSV data via
the Data Hub, the resolver returns THAT data — normalised through the dataset's
stored column mapping (Data Hub's raw→canonical map) — otherwise it falls back to
the bundled Olist demo files.

Only the *source* of the DataFrame moves; the shapes returned here match what the
existing modules already consume, so all calculations and business logic are
preserved unchanged. With nothing imported, behaviour is byte-for-byte the demo.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from api import data_hub

_OLIST_ORDERS_TS = "data/olist_orders.csv"            # time-series shape (order_purchase_timestamp)
_OLIST_ORDERS = "data/olist_orders_dataset.csv"        # full orders
_OLIST_CUSTOMERS = "data/olist_customers_dataset.csv"

# The resolver keys on which CANONICAL columns a dataset maps to (via the Data
# Hub column mapping) — not the fuzzy detected `type` — so any ERP export that
# carries orders/customers/regions is picked up regardless of how it was labelled.
_ORDER_FIELDSETS = (["order_id"], ["customer"])         # an order-bearing dataset
_CUSTOMER_FIELDSETS = (["customer", "region"], ["region"])  # a region-bearing dataset


def _read_olist(path: str, usecols: Optional[list[str]]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols) if usecols else pd.read_csv(path)


def _imported_records() -> list[dict]:
    """All imported datasets, newest first (metadata only — no file reads)."""
    recs = []
    for pub in data_hub.datasets()["datasets"]:
        if pub["status"] == "imported":
            full = data_hub._load(pub["id"])
            if full:
                recs.append(full)
    recs.sort(key=lambda r: r.get("imported_at") or 0, reverse=True)
    return recs


def _canonical_cols(rec: dict) -> set:
    return {canon for canon in (rec.get("mapping") or {}).values() if canon}


def _pick_record(*fieldsets: list[str]) -> Optional[dict]:
    """Newest imported dataset whose canonical columns include every field of
    ANY of the given field-sets. Metadata only — cheap."""
    for rec in _imported_records():
        cols = _canonical_cols(rec)
        if any(set(fs) <= cols for fs in fieldsets):
            return rec
    return None


def _pick(*fieldsets: list[str]):
    """Like _pick_record, but loads and canonically-renames the DataFrame."""
    rec = _pick_record(*fieldsets)
    if rec is None:
        return None, None
    try:
        df = data_hub.read_dataset(rec)
    except Exception:  # noqa: BLE001 — missing/corrupt file → demo fallback
        return None, None
    mapping = {raw: canon for raw, canon in (rec.get("mapping") or {}).items() if canon}
    return df.rename(columns=mapping), rec


# ── status ────────────────────────────────────────────────────────────────────

def active_source() -> str:
    has_orders = _pick_record(*_ORDER_FIELDSETS) is not None
    has_customers = _pick_record(*_CUSTOMER_FIELDSETS) is not None
    return "imported" if (has_orders or has_customers) else "demo"


def active_summary() -> dict:
    orders = _pick_record(*_ORDER_FIELDSETS)
    custs = _pick_record(*_CUSTOMER_FIELDSETS)
    return {
        "source": "imported" if (orders or custs) else "demo",
        "orders_dataset": orders["name"] if orders else None,
        "customers_dataset": custs["name"] if custs else None,
        "demo_dataset": "Olist (99k orders)",
    }


# ── orders ────────────────────────────────────────────────────────────────────

def _shape_orders(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalise an imported orders frame to the Olist column shape the modules
    consume. Returns None if it can't even identify an order/customer."""
    has_order = "order_id" in df.columns
    has_customer = "customer" in df.columns or "customer_id" in df.columns
    if not (has_order or has_customer):
        return None
    out = pd.DataFrame()
    out["order_id"] = df["order_id"].astype(str) if has_order else [f"ORD-{i}" for i in range(len(df))]
    if "customer" in df.columns:
        out["customer_id"] = df["customer"].astype(str)
    elif "customer_id" in df.columns:
        out["customer_id"] = df["customer_id"].astype(str)
    else:
        out["customer_id"] = out["order_id"]
    out["order_status"] = df["status"].astype(str).str.lower() if "status" in df.columns else "delivered"
    out["order_purchase_timestamp"] = (pd.to_datetime(df["date"], errors="coerce")
                                       if "date" in df.columns else pd.NaT)
    out["order_estimated_delivery_date"] = (pd.to_datetime(df["expected_arrival"], errors="coerce")
                                            if "expected_arrival" in df.columns else out["order_purchase_timestamp"])
    # No separate "actual delivered" column in a generic export → mirror estimate.
    out["order_delivered_customer_date"] = out["order_estimated_delivery_date"]
    # Carry region through if the export is denormalised (drives customer derivation).
    if "region" in df.columns:
        out["customer_state"] = df["region"].astype(str)
    return out


def orders_dataset(usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """Full orders frame (Olist shape) for the control-tower / shipments pipeline."""
    df, _ = _pick(*_ORDER_FIELDSETS)
    if df is not None:
        shaped = _shape_orders(df)
        if shaped is not None:
            cols = [c for c in (usecols or shaped.columns) if c in shaped.columns]
            return shaped[cols] if usecols else shaped
    return _read_olist(_OLIST_ORDERS, usecols)


def orders_min() -> pd.DataFrame:
    """[order_id, customer_id] — used to count order volume per customer/region."""
    df, _ = _pick(*_ORDER_FIELDSETS)
    if df is not None:
        shaped = _shape_orders(df)
        if shaped is not None:
            return shaped[["order_id", "customer_id"]]
    return _read_olist(_OLIST_ORDERS, ["order_id", "customer_id"])


def forecast_orders() -> pd.DataFrame:
    """Orders with an ``order_purchase_timestamp`` column for demand forecasting.
    Only overrides the demo when the imported orders actually carry dates."""
    df, _ = _pick(["date", "order_id"], ["date", "customer"])
    if df is not None:
        shaped = _shape_orders(df)
        if shaped is not None and shaped["order_purchase_timestamp"].notna().any():
            return shaped
    out = _read_olist(_OLIST_ORDERS_TS, None)
    out["order_purchase_timestamp"] = pd.to_datetime(out["order_purchase_timestamp"])
    return out


# ── customers ─────────────────────────────────────────────────────────────────

def customers_states(usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """[customer_id, customer_state] — regionised customers for commercial accounts.
    Uses any imported dataset carrying a region (a customers file, or a
    denormalised orders export with customer + region), else the demo."""
    df, _ = _pick(*_CUSTOMER_FIELDSETS)
    if df is not None and "region" in df.columns:
        if "customer" in df.columns:
            out = (df[["customer", "region"]].dropna(subset=["customer"]).drop_duplicates("customer")
                   .rename(columns={"customer": "customer_id", "region": "customer_state"}))
        else:
            out = pd.DataFrame({"customer_id": [f"C-{i}" for i in range(len(df))],
                                "customer_state": df["region"].astype(str)})
        out["customer_id"] = out["customer_id"].astype(str)
        out["customer_state"] = out["customer_state"].astype(str)
        return out[usecols] if usecols else out
    return _read_olist(_OLIST_CUSTOMERS, usecols)


def customers_geo(usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """Customers with geo attributes (zip/city/state) for hub clustering. Kept on
    the demo geography — arbitrary ERP exports don't carry lat/lon coordinates."""
    return _read_olist(_OLIST_CUSTOMERS, usecols)
