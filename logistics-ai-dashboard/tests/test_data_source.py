"""Tests for the centralized data-source layer (demo fallback ↔ imported switch)."""
from api import data_source, data_hub
from api import commercial_intel as ci


def test_defaults_to_demo_when_nothing_imported():
    assert data_source.active_source() == "demo"
    orders = data_source.orders_min()
    assert {"order_id", "customer_id"} <= set(orders.columns) and len(orders) > 1000
    custs = data_source.customers_states(["customer_id", "customer_state"])
    assert "customer_state" in custs.columns
    # Demo accounts are the named Brazilian-region accounts.
    assert any(a["name"] == "Paulista Retail Group" for a in ci._accounts())


def test_forecast_orders_have_timestamp():
    df = data_source.forecast_orders()
    assert "order_purchase_timestamp" in df.columns
    assert df["order_purchase_timestamp"].notna().any()


def test_imported_orders_switch_the_whole_commercial_pipeline(tmp_path, monkeypatch):
    # Fresh, isolated Data Hub registry for this test.
    monkeypatch.setenv("DATA_HUB_DIR", str(tmp_path))
    rows = [f"SO-{i},Acme Corp,North America" for i in range(5)] + \
           [f"SO-{i},Globex Ltd,EMEA" for i in range(5, 8)]
    csv = ("OrderID,Customer,Region\n" + "\n".join(rows) + "\n").encode()

    up = data_hub.upload("sales_orders.csv", csv)
    did = up["dataset"]["id"]
    # (detected type is fuzzy and irrelevant — the resolver keys on mapped columns)
    mapping = {"OrderID": "order_id", "Customer": "customer", "Region": "region"}
    imp = data_hub.do_import(did, mapping, {"index_docs": False, "learn_suppliers": False, "semantic_search": False})
    assert imp["ok"]

    # The platform now reads the imported data.
    assert data_source.active_source() == "imported"
    accts = ci._accounts()
    names = {a["name"] for a in accts}
    assert "North America" in names and "Emea" in names        # imported regions became accounts
    assert "Paulista Retail Group" not in names                # demo accounts gone
    north = next(a for a in accts if a["name"] == "North America")
    assert north["orders"] == 5                                # real counts from the upload

    # Delete the import → the platform reverts to the demo automatically.
    data_hub.delete(did)
    assert data_source.active_source() == "demo"
    assert any(a["name"] == "Paulista Retail Group" for a in ci._accounts())
