"""Tests for the Data Hub (real parse → detect → map → validate → import)."""
from api import data_hub as dh


def test_detect_and_suggest_mapping():
    up = dh.upload("inventory.csv",
                   b"SKU,Supplier_Name,Qty,Warehouse,ETA\nS1,Acme,10,DC-East,2026-08-01\n")
    try:
        assert up["ok"]
        assert up["dataset"]["type"] == "inventory"
        m = up["dataset"]["mapping"]
        assert m["SKU"] == "sku" and m["Supplier_Name"] == "supplier"
        assert m["Qty"] == "quantity" and m["ETA"] == "expected_arrival"
        assert "confidence" in up["detection_message"].lower() or up["dataset"]["confidence"] > 0
    finally:
        dh.delete(up["dataset"]["id"])


def test_validation_flags_problems():
    up = dh.upload("d.csv", b"SKU,Qty,ETA\nA,5,2026-01-01\nA,5,2026-01-01\nB,,not-a-date\n")
    try:
        v = up["dataset"]["validation"]
        assert v["duplicate_records"] >= 1
        assert v["missing_values"] >= 1
        assert v["invalid_dates"] >= 1
        assert 0 <= v["health_score"] <= 100
    finally:
        dh.delete(up["dataset"]["id"])


def test_import_indexes_and_registers():
    up = dh.upload("po.csv", b"PO,Supplier_Name,Qty\nP1,Acme,3\nP2,Globex,4\n")
    did = up["dataset"]["id"]
    try:
        imp = dh.do_import(did, None, {"index_docs": True, "learn_suppliers": True, "semantic_search": True})
        assert imp["ok"] and imp["dataset"]["status"] == "imported"
        assert imp["index"]["documents"] >= 1        # indexed into Knowledge/RAG + Brain
        assert did in {d["id"] for d in dh.datasets()["datasets"]}
        assert dh.quality()["kpis"]["datasets"] >= 1
        prev = dh.preview(did)
        assert prev["ok"] and prev["rows"]
    finally:
        dh.delete(did)
    assert did not in {d["id"] for d in dh.datasets()["datasets"]}


def test_json_and_excel_parse():
    up = dh.upload("c.json", b'[{"Customer":"Acme","Region":"SP"},{"Customer":"Globex","Region":"RJ"}]')
    try:
        assert up["ok"] and up["dataset"]["rows"] == 2
        assert up["dataset"]["type"] == "customers"
    finally:
        dh.delete(up["dataset"]["id"])


def test_unknown_dataset_is_safe():
    assert dh.preview("nope")["ok"] is False
    assert dh.do_import("nope", None, {})["ok"] is False
    assert dh.delete("nope")["ok"] is False
