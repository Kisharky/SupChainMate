"""
api/connectors.py — the Connectors & Integrations catalog.

The enterprise "where does the data come from?" surface: a catalog of
operational systems (ERP, WMS, TMS, cloud storage, databases, BI, APIs, files)
that an enterprise can connect to feed SupChainMate's decision intelligence.

This is the **plug-in seam**. The catalog and connection state here are
representative (labelled everywhere as such) — no system is actually contacted.
A real deployment implements ``_probe(connector_id)`` against a live driver and
persists connection config in a secrets store; every other layer (the API
endpoints, the whole frontend) stays exactly the same.

No business logic lives here, and nothing in ``ai/``, ``planner/``, ``brain/``,
or ``optimize/`` is touched — the pipeline definition below merely *describes*
how ingested data flows into those existing layers.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from api.services import _safe

# Auth methods, per connector category, drive the fields the config panel shows.
_AUTH_BY_CATEGORY = {
    "ERP": "OAuth 2.0",
    "Warehouse": "API Key",
    "Transportation": "API Key",
    "Cloud Storage": "Access Key / Secret",
    "Databases": "Username / Password",
    "Business Intelligence": "OAuth 2.0",
    "APIs": "Bearer Token",
    "Files": "None (upload)",
}

# The connector catalog, grouped by category. ``connected`` seeds a realistic
# demo posture — a few representative systems are "live", the rest are available
# to configure. A real deployment derives this from the connection store.
_CATALOG: dict[str, list[dict[str, Any]]] = {
    "ERP": [
        {"id": "sap_s4hana", "name": "SAP S/4HANA", "icon": "◆",
         "description": "Orders, materials, and financials from SAP's ERP core.", "connected": True},
        {"id": "oracle_scm", "name": "Oracle SCM Cloud", "icon": "◈",
         "description": "Supply chain planning and procurement from Oracle Cloud.", "connected": False},
        {"id": "dynamics_365", "name": "Microsoft Dynamics 365", "icon": "◇",
         "description": "Finance & operations records from Dynamics 365.", "connected": False},
        {"id": "netsuite", "name": "NetSuite", "icon": "▨",
         "description": "Cloud ERP orders, inventory, and GL from NetSuite.", "connected": False},
        {"id": "odoo", "name": "Odoo", "icon": "▧",
         "description": "Open-source ERP modules for SMB operations.", "connected": False},
    ],
    "Warehouse": [
        {"id": "manhattan", "name": "Manhattan Active", "icon": "▦",
         "description": "Warehouse execution and labour from Manhattan Active WM.", "connected": True},
        {"id": "blue_yonder", "name": "Blue Yonder WMS", "icon": "▤",
         "description": "Fulfilment and slotting from Blue Yonder.", "connected": False},
        {"id": "oracle_wms", "name": "Oracle WMS", "icon": "▥",
         "description": "Inbound/outbound and inventory from Oracle WMS Cloud.", "connected": False},
        {"id": "sap_ewm", "name": "SAP EWM", "icon": "▩",
         "description": "Extended warehouse management from SAP EWM.", "connected": False},
    ],
    "Transportation": [
        {"id": "oracle_tms", "name": "Oracle TMS", "icon": "◎",
         "description": "Transportation planning and freight from Oracle TMS.", "connected": False},
        {"id": "sap_tm", "name": "SAP TM", "icon": "◉",
         "description": "Transportation management and settlement from SAP TM.", "connected": False},
        {"id": "project44", "name": "project44", "icon": "◍",
         "description": "Real-time multimodal visibility from project44.", "connected": True},
        {"id": "fourkites", "name": "FourKites", "icon": "◌",
         "description": "Predictive ETAs and tracking from FourKites.", "connected": False},
    ],
    "Cloud Storage": [
        {"id": "aws_s3", "name": "AWS S3", "icon": "☁",
         "description": "Object storage buckets on Amazon S3.", "connected": True},
        {"id": "azure_blob", "name": "Azure Blob Storage", "icon": "☁",
         "description": "Containers and blobs on Microsoft Azure.", "connected": False},
        {"id": "gcs", "name": "Google Cloud Storage", "icon": "☁",
         "description": "Buckets on Google Cloud Storage.", "connected": False},
    ],
    "Databases": [
        {"id": "postgresql", "name": "PostgreSQL", "icon": "▣",
         "description": "Relational tables and views over PostgreSQL.", "connected": True},
        {"id": "sql_server", "name": "SQL Server", "icon": "▢",
         "description": "Microsoft SQL Server databases and views.", "connected": False},
        {"id": "mysql", "name": "MySQL", "icon": "▣",
         "description": "MySQL / MariaDB relational sources.", "connected": False},
        {"id": "snowflake", "name": "Snowflake", "icon": "❄",
         "description": "Warehouses and shares on Snowflake.", "connected": False},
        {"id": "bigquery", "name": "BigQuery", "icon": "▤",
         "description": "Datasets and tables on Google BigQuery.", "connected": False},
    ],
    "Business Intelligence": [
        {"id": "power_bi", "name": "Power BI", "icon": "▥",
         "description": "Datasets and semantic models from Power BI.", "connected": False},
        {"id": "tableau", "name": "Tableau", "icon": "▦",
         "description": "Published data sources from Tableau.", "connected": False},
    ],
    "APIs": [
        {"id": "rest_api", "name": "REST API", "icon": "⇄",
         "description": "Any REST endpoint with token or key auth.", "connected": True},
        {"id": "graphql_api", "name": "GraphQL API", "icon": "⇆",
         "description": "A GraphQL schema queried for records.", "connected": False},
    ],
    "Files": [
        {"id": "csv", "name": "CSV", "icon": "▤",
         "description": "Comma-separated exports, mapped on upload.", "connected": True},
        {"id": "excel", "name": "Excel", "icon": "▦",
         "description": "XLSX workbooks with one sheet per entity.", "connected": True},
        {"id": "json", "name": "JSON", "icon": "❴",
         "description": "Structured JSON documents and arrays.", "connected": False},
        {"id": "xml", "name": "XML", "icon": "❬",
         "description": "XML feeds and EDI-style documents.", "connected": False},
    ],
}

# The pipeline every connected source flows through — a description of the
# existing architecture, not a re-implementation of it.
_PIPELINE = [
    {"stage": "Source System", "detail": "SAP · Oracle · databases · files", "kind": "source"},
    {"stage": "Data Validation", "detail": "Schema checks, types, required fields", "kind": "process"},
    {"stage": "Transformation", "detail": "Map to the canonical supply-chain model", "kind": "process"},
    {"stage": "Decision Brain", "detail": "Ingested as long-term memory & knowledge", "kind": "intelligence"},
    {"stage": "Planner", "detail": "Objectives decomposed over live data", "kind": "intelligence"},
    {"stage": "Executive Dashboard", "detail": "Evidence-backed decisions", "kind": "output"},
]


def _all() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for category, items in _CATALOG.items():
        for c in items:
            out.append({**c, "category": category, "auth": _AUTH_BY_CATEGORY[category]})
    return out


def catalog() -> dict[str, Any]:
    """Full catalog grouped by category, with per-connector auth method."""
    return _safe(lambda: {
        "categories": [
            {"category": cat, "auth": _AUTH_BY_CATEGORY[cat],
             "connectors": [{**c, "category": cat, "auth": _AUTH_BY_CATEGORY[cat]} for c in items]}
            for cat, items in _CATALOG.items()
        ],
        "summary": _summary(),
        "sync": _sync(),
        "pipeline": _PIPELINE,
        "source": "representative",
    }, {"categories": [], "summary": {}, "sync": {}, "pipeline": _PIPELINE, "source": "fallback"})


def _summary() -> dict[str, Any]:
    conns = _all()
    active = [c for c in conns if c["connected"]]
    # Deterministic representative volume derived from the connected set.
    daily = sum(120_000 + (int(hashlib.sha1(c["id"].encode()).hexdigest(), 16) % 380_000)
                for c in active)
    return {
        "active_connections": len(active),
        "connected_systems": len({c["category"] for c in active}),
        "last_sync": "2 min ago",
        "data_health": 98,               # % — representative
        "failed_connections": 1,
        "daily_records": daily,
    }


def _sync() -> dict[str, Any]:
    return {
        "last_sync": "2 minutes ago",
        "next_sync": "in 13 minutes",
        "records_imported": 1_284_930,
        "records_failed": 412,
        "duration_s": 47,
        "status": "healthy",
        "frequency": "Every 15 minutes",
        "progress": 100,
    }


def config_schema(connector_id: str) -> dict[str, Any]:
    """The fields the configuration panel should render for a connector.
    Representative — a real driver returns its own required parameters."""
    match = next((c for c in _all() if c["id"] == connector_id), None)
    if match is None:
        return {"ok": False, "error": "unknown connector", "connector_id": connector_id}
    category = match["category"]
    base = ["Connection Name", "Environment", "Sync Frequency"]
    by_cat = {
        "ERP": ["Host", "Client ID", "OAuth Scope", "Region"],
        "Warehouse": ["Host", "API Key", "Region"],
        "Transportation": ["Host", "API Key"],
        "Cloud Storage": ["Bucket", "Access Key", "Secret Key", "Region"],
        "Databases": ["Host", "Port", "Database", "Username", "Password"],
        "Business Intelligence": ["Workspace", "Client ID", "OAuth Scope"],
        "APIs": ["Base URL", "Bearer Token"],
        "Files": ["Delimiter", "Encoding"],
    }
    return {
        "ok": True, "connector_id": connector_id, "name": match["name"],
        "category": category, "auth": match["auth"], "connected": match["connected"],
        "fields": base[:1] + by_cat.get(category, []) + base[1:],
        "source": "representative",
    }


def test_connection(connector_id: str) -> dict[str, Any]:
    """Representative connection test. Deterministic per id so the demo is
    stable; a real deployment performs an actual handshake here."""
    match = next((c for c in _all() if c["id"] == connector_id), None)
    if match is None:
        return {"ok": False, "connector_id": connector_id, "status": "error",
                "message": "Unknown connector.", "latency_ms": 0}
    # Stable pseudo-latency + result from the id hash.
    h = int(hashlib.sha1(connector_id.encode()).hexdigest(), 16)
    latency = 40 + (h % 220)
    ok = (h % 17) != 0  # ~6% representative failure rate
    return {
        "ok": ok, "connector_id": connector_id, "name": match["name"],
        "status": "success" if ok else "error",
        "message": (f"Reached {match['name']} and authenticated ({match['auth']})."
                    if ok else f"Could not authenticate with {match['name']} — check credentials."),
        "latency_ms": latency, "source": "representative",
    }
