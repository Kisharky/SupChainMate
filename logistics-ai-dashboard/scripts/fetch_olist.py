"""Download the full Olist Brazilian e-commerce dataset (99k orders) into data/.

The repo ships a 5,000-order sample so it clones fast. Run this once for the
full set:

    python scripts/fetch_olist.py

Source: Olist's own public mirror of the Kaggle dataset
(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — byte-identical
files, no Kaggle account or API token needed. To restore the sample afterwards:
`git checkout -- data/`.
"""
import os
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# (source file, local file) — olist_orders.csv is the same orders file under the
# name the forecasting module reads.
FILES = [
    ("olist_orders_dataset.csv", "olist_orders_dataset.csv"),
    ("olist_orders_dataset.csv", "olist_orders.csv"),
    ("olist_customers_dataset.csv", "olist_customers_dataset.csv"),
    ("olist_geolocation_dataset.csv", "olist_geolocation_dataset.csv"),
]


def main() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)
    for src, dst in FILES:
        path = os.path.join(DATA_DIR, dst)
        print(f"↓ {src} → data/{dst}")
        urllib.request.urlretrieve(BASE + src, path + ".part")
        os.replace(path + ".part", path)
    print("Done — full Olist dataset in data/. Restart the API to pick it up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
