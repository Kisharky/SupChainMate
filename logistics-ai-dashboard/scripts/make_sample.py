"""Regenerate the in-repo 5,000-order sample from the full Olist dataset.

    python scripts/fetch_olist.py   # full data into data/
    python scripts/make_sample.py   # shrink it back to the committed sample

The sample is referentially consistent: 5,000 random orders (seed 42), exactly
the customers those orders reference, and one geolocation row per customer zip
prefix holding the median lat/lng — the same value modules/network.py computes
from the full 1M-row file, so the map positions are unchanged.
"""
import os

import pandas as pd

N_ORDERS = 5000
SEED = 42
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def p(name: str) -> str:
    return os.path.join(DATA, name)


def main() -> None:
    orders = pd.read_csv(p("olist_orders_dataset.csv"), dtype=str)
    customers = pd.read_csv(p("olist_customers_dataset.csv"), dtype=str)
    geo = pd.read_csv(p("olist_geolocation_dataset.csv"))

    sample = orders.sample(n=min(N_ORDERS, len(orders)), random_state=SEED)
    sample = sample.sort_values("order_purchase_timestamp")
    custs = customers[customers["customer_id"].isin(sample["customer_id"])]

    zips = set(custs["customer_zip_code_prefix"].astype(int))
    geo = geo[geo["geolocation_zip_code_prefix"].isin(zips)]
    # Same Brazil bounding-box clip network.py applies before taking medians.
    geo = geo[geo["geolocation_lat"].between(-34, 6) & geo["geolocation_lng"].between(-74, -28)]
    geo_sample = (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(geolocation_lat=("geolocation_lat", "median"),
             geolocation_lng=("geolocation_lng", "median"),
             geolocation_city=("geolocation_city", "first"),
             geolocation_state=("geolocation_state", "first"))
        .reset_index()
    )

    sample.to_csv(p("olist_orders_dataset.csv"), index=False)
    sample.to_csv(p("olist_orders.csv"), index=False)
    custs.to_csv(p("olist_customers_dataset.csv"), index=False)
    geo_sample.to_csv(p("olist_geolocation_dataset.csv"), index=False)
    print(f"orders={len(sample)} customers={len(custs)} geo_zips={len(geo_sample)}")


if __name__ == "__main__":
    main()
