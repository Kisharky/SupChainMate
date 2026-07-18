"""
modules/connect.py
Live store connectors — pull orders straight from Shopify or WooCommerce
instead of uploading a CSV.

- Shopify: Admin REST API with a private-app Admin access token
  (shpat_...). Read-only scope `read_orders` is all that's needed.
- WooCommerce: REST API v3 with a read-only consumer key/secret pair.

Both return a normalised orders DataFrame (order_date, quantity) that the
existing ingestion pipeline consumes unchanged. Credentials are used for
the fetch only and are never persisted.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import requests

SHOPIFY_API_VERSION = "2024-01"
_MAX_PAGES = 20          # 250 orders/page → up to 5,000 orders per sync
_TIMEOUT = 30


def _clean_domain(url: str) -> str:
    """Normalise a pasted store URL to a bare host."""
    url = url.strip().rstrip("/")
    url = re.sub(r"^https?://", "", url)
    return url.split("/")[0]


def fetch_shopify_orders(shop_url: str, access_token: str) -> tuple[Optional[pd.DataFrame], str]:
    """
    Fetch orders from a Shopify store. Returns (orders_df, message);
    orders_df is None on failure.
    """
    host = _clean_domain(shop_url)
    if not host or not access_token.strip():
        return None, "Enter your store URL and Admin API access token."

    url = (f"https://{host}/admin/api/{SHOPIFY_API_VERSION}/orders.json"
           f"?status=any&limit=250&fields=created_at,line_items")
    headers = {"X-Shopify-Access-Token": access_token.strip()}
    rows: list[dict] = []

    try:
        for _ in range(_MAX_PAGES):
            resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
            if resp.status_code == 401:
                return None, "Shopify rejected the token (401) — check the Admin API access token."
            if resp.status_code == 404:
                return None, f"Store not found at {host} — check the URL (e.g. mystore.myshopify.com)."
            resp.raise_for_status()
            orders = resp.json().get("orders", [])
            for o in orders:
                qty = sum(int(li.get("quantity", 0)) for li in o.get("line_items", []))
                rows.append({"order_date": o.get("created_at"), "quantity": max(qty, 1)})
            # Cursor pagination via the Link header
            link = resp.headers.get("Link", "")
            nxt = re.search(r'<([^>]+)>;\s*rel="next"', link)
            if not nxt or not orders:
                break
            url = nxt.group(1)
    except requests.exceptions.RequestException as e:
        return None, f"Shopify connection failed: {e}"

    if not rows:
        return None, "Connected, but the store returned no orders."
    df = pd.DataFrame(rows)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df[df["order_date"].notna()]
    return df, f"Imported {len(df):,} orders from {host}."


def fetch_woocommerce_orders(site_url: str, consumer_key: str,
                             consumer_secret: str) -> tuple[Optional[pd.DataFrame], str]:
    """
    Fetch orders from a WooCommerce store (REST API v3).
    Returns (orders_df, message); orders_df is None on failure.
    """
    host = _clean_domain(site_url)
    if not host or not consumer_key.strip() or not consumer_secret.strip():
        return None, "Enter your site URL, consumer key, and consumer secret."

    base = f"https://{host}/wp-json/wc/v3/orders"
    auth = (consumer_key.strip(), consumer_secret.strip())
    rows: list[dict] = []

    try:
        for page in range(1, _MAX_PAGES + 1):
            resp = requests.get(base, auth=auth,
                                params={"per_page": 100, "page": page},
                                timeout=_TIMEOUT)
            if resp.status_code == 401:
                return None, "WooCommerce rejected the keys (401) — check the consumer key/secret."
            if resp.status_code == 404:
                return None, f"No WooCommerce API at {host} — is the REST API enabled?"
            resp.raise_for_status()
            orders = resp.json()
            if not isinstance(orders, list) or not orders:
                break
            for o in orders:
                qty = sum(int(li.get("quantity", 0)) for li in o.get("line_items", []))
                rows.append({"order_date": o.get("date_created"), "quantity": max(qty, 1)})
            if len(orders) < 100:
                break
    except requests.exceptions.RequestException as e:
        return None, f"WooCommerce connection failed: {e}"
    except ValueError:
        return None, "WooCommerce returned unexpected data — check the site URL."

    if not rows:
        return None, "Connected, but the store returned no orders."
    df = pd.DataFrame(rows)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df[df["order_date"].notna()]
    return df, f"Imported {len(df):,} orders from {host}."
