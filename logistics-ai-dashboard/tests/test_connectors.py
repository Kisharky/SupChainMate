"""Tests for the Connectors & Integrations catalog (representative, UI-only)."""
from api import connectors


def test_catalog_shape():
    cat = connectors.catalog()
    assert cat["source"] == "representative"
    categories = {c["category"] for c in cat["categories"]}
    # Every category the workspace advertises is present.
    assert {"ERP", "Warehouse", "Transportation", "Cloud Storage",
            "Databases", "Business Intelligence", "APIs", "Files"} <= categories
    # Every connector carries the fields the UI cards need.
    for group in cat["categories"]:
        for c in group["connectors"]:
            assert {"id", "name", "icon", "description", "connected", "category", "auth"} <= c.keys()


def test_summary_counts_match_connected():
    cat = connectors.catalog()
    connected = [c for g in cat["categories"] for c in g["connectors"] if c["connected"]]
    assert cat["summary"]["active_connections"] == len(connected)
    assert cat["summary"]["daily_records"] > 0


def test_pipeline_ends_at_executive_dashboard():
    stages = [s["stage"] for s in connectors.catalog()["pipeline"]]
    assert stages[0] == "Source System"
    assert stages[-1] == "Executive Dashboard"
    assert "Decision Brain" in stages and "Planner" in stages


def test_config_schema_fields_by_category():
    db = connectors.config_schema("postgresql")
    assert db["ok"] and "Password" in db["fields"] and "Database" in db["fields"]
    api_cfg = connectors.config_schema("rest_api")
    assert "Bearer Token" in api_cfg["fields"]
    assert connectors.config_schema("does_not_exist")["ok"] is False


def test_test_connection_is_deterministic():
    a = connectors.test_connection("sap_s4hana")
    b = connectors.test_connection("sap_s4hana")
    assert a["status"] == b["status"] and a["latency_ms"] == b["latency_ms"]
    assert connectors.test_connection("nope")["status"] == "error"
