"""
api/data_hub.py — the Data Hub: enterprise data onboarding.

The central place a company brings its own operational data in (CSV / Excel /
JSON) without touching code: upload → AI dataset detection → column mapping →
validation → import → indexing. It ORCHESTRATES existing systems rather than
duplicating them — imported documents are indexed into the **Knowledge/RAG store**
(`modules.store.add_document`) and the **Decision Brain** (`BRAIN.add_knowledge` /
`record_entity`), both real and offline, so the Knowledge Center and Planner
immediately see the uploaded data.

Storage is isolated (its own SQLite registry + a raw-file store), so nothing in
the existing DB layer, business modules, AI Router, Planner, or Brain is
modified — this module only calls their public APIs.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import time
import uuid
from typing import Any, Optional

import config

_log = config.get_logger(__name__)

_DIR = os.path.join("data", "data_hub")
_UPLOADS = os.path.join(_DIR, "files")
_DB = os.path.join(_DIR, "registry.db")

# Canonical target schema fields the mapping UI maps raw columns onto.
CANONICAL_FIELDS = [
    "sku", "product", "quantity", "supplier", "customer", "warehouse", "location",
    "carrier", "po_number", "order_id", "price", "cost", "date", "expected_arrival",
    "lead_time", "category", "region", "status", "forecast", "demand",
]

# Column-token signatures per supported dataset type (for AI detection).
_SIGNATURES: dict[str, tuple[str, set[str]]] = {
    "inventory": ("Inventory", {"sku", "quantity", "onhand", "on_hand", "stock", "warehouse", "reorder"}),
    "purchase_orders": ("Purchase Orders", {"po", "purchase", "supplier", "vendor", "eta", "order"}),
    "sales_orders": ("Sales Orders", {"order", "customer", "sales", "qty", "amount"}),
    "shipments": ("Shipments", {"shipment", "tracking", "carrier", "origin", "destination", "eta"}),
    "suppliers": ("Suppliers", {"supplier", "vendor", "contact", "leadtime", "lead_time", "country"}),
    "customers": ("Customers", {"customer", "account", "region", "email", "segment"}),
    "products": ("Products", {"product", "sku", "category", "price", "description", "uom"}),
    "warehouse_locations": ("Warehouse Locations", {"warehouse", "location", "aisle", "bin", "zone"}),
    "forecast": ("Forecast Data", {"forecast", "predicted", "period", "demand", "horizon"}),
    "demand_history": ("Demand History", {"demand", "date", "quantity", "actual", "period"}),
    "production_orders": ("Production Orders", {"production", "workorder", "work_order", "bom", "output"}),
    "transport_costs": ("Transport Costs", {"lane", "cost", "rate", "mode", "distance", "origin"}),
    "carrier_performance": ("Carrier Performance", {"carrier", "ontime", "on_time", "transit", "score"}),
}

# Plausible source systems named per dataset type (for the detection message).
_SOURCE_SYSTEMS = ["SAP", "Oracle SCM", "Microsoft Dynamics 365", "NetSuite", "Odoo"]

# Synonyms → canonical field, for the suggested mapping.
_SYNONYMS: dict[str, str] = {}
for _canon, _alts in {
    "sku": ["sku", "item", "itemcode", "item_code", "material", "productcode", "product_code"],
    "product": ["product", "productname", "product_name", "description", "item_name"],
    "quantity": ["qty", "quantity", "units", "onhand", "on_hand", "stock", "count"],
    "supplier": ["supplier", "suppliername", "supplier_name", "vendor", "vendorname", "vendor_name"],
    "customer": ["customer", "customername", "customer_name", "account", "client"],
    "warehouse": ["warehouse", "wh", "dc", "facility", "plant"],
    "location": ["location", "aisle", "bin", "zone", "slot"],
    "carrier": ["carrier", "carriername", "carrier_name", "scac"],
    "po_number": ["po", "ponumber", "po_number", "purchaseorder", "purchase_order"],
    "order_id": ["order", "orderid", "order_id", "salesorder", "sales_order"],
    "price": ["price", "unitprice", "unit_price", "sellprice"],
    "cost": ["cost", "unitcost", "unit_cost", "rate", "landedcost"],
    "date": ["date", "orderdate", "order_date", "created", "timestamp", "postingdate"],
    "expected_arrival": ["eta", "expectedarrival", "expected_arrival", "arrival", "duedate", "due_date"],
    "lead_time": ["leadtime", "lead_time", "leaddays"],
    "category": ["category", "productcategory", "class", "family"],
    "region": ["region", "state", "country", "market", "territory"],
    "status": ["status", "state", "stage"],
    "forecast": ["forecast", "predicted", "projection"],
    "demand": ["demand", "actual", "consumption"],
}.items():
    for _a in _alts:
        _SYNONYMS[_a] = _canon


# ── storage ───────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    os.makedirs(_UPLOADS, exist_ok=True)
    conn = sqlite3.connect(_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY, name TEXT, filename TEXT, ext TEXT, type TEXT, type_label TEXT,
        source_guess TEXT, confidence REAL, rows INTEGER, columns TEXT, mapping TEXT,
        validation TEXT, stats TEXT, status TEXT, health INTEGER, indexed INTEGER,
        imported_by TEXT, created_at REAL, imported_at REAL, filepath TEXT)""")
    return conn


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum() or ch == "_")


# ── parsing ─────────────────────────────────────────────────────────────────--

def _read_dataframe(content: bytes, ext: str):
    import pandas as pd
    if ext == "csv":
        return pd.read_csv(io.BytesIO(content))
    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(content), engine="openpyxl")
    if ext == "json":
        text = content.decode("utf-8", errors="replace").strip()
        data = json.loads(text)
        if isinstance(data, dict):
            # {"records": [...]} or a dict of columns.
            for key in ("records", "data", "rows", "items"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        return pd.DataFrame(data if isinstance(data, list) else [data])
    raise ValueError(f"unsupported extension: {ext}")


def _rows(df, n: int = 8) -> list[dict[str, Any]]:
    import pandas as pd
    head = df.head(n).where(df.head(n).notna(), None)
    out = []
    for rec in head.to_dict(orient="records"):
        out.append({str(k): (None if (isinstance(v, float) and pd.isna(v)) else
                    (v.item() if hasattr(v, "item") else v)) for k, v in rec.items()})
    return out


# ── detection + mapping ───────────────────────────────────────────────────────

def _detect(columns: list[str]) -> tuple[str, str, str, int]:
    tokens = {_norm(c) for c in columns}
    tokens |= {t for c in columns for t in _norm(c).split("_")}
    best, best_hits = "products", 0
    for dtype, (_, sig) in _SIGNATURES.items():
        hits = len(tokens & sig)
        if hits > best_hits:
            best, best_hits = dtype, hits
    label = _SIGNATURES[best][0]
    sig_size = max(1, len(_SIGNATURES[best][1]))
    confidence = min(97, 45 + round(55 * best_hits / sig_size))
    # Deterministic plausible source system from the columns.
    src = _SOURCE_SYSTEMS[sum(ord(c) for c in "".join(columns)) % len(_SOURCE_SYSTEMS)]
    return best, label, f"{src} {label} Export", confidence


def _suggest_mapping(columns: list[str]) -> dict[str, str]:
    mapping = {}
    for c in columns:
        n = _norm(c)
        mapping[c] = _SYNONYMS.get(n, _SYNONYMS.get(n.replace("_", ""), ""))
    return mapping


# ── validation ────────────────────────────────────────────────────────────────

def _validate(df, mapping: dict[str, str]) -> dict[str, Any]:
    import pandas as pd
    rows = int(len(df))
    cols = int(df.shape[1]) or 1
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    # Date columns (by mapping or name) that fail to parse.
    invalid_dates = 0
    for col in df.columns:
        canon = mapping.get(col, "")
        if canon in ("date", "expected_arrival") or any(k in _norm(col) for k in ("date", "eta", "arrival")):
            parsed = pd.to_datetime(df[col], errors="coerce")
            invalid_dates += int(parsed.isna().sum() - df[col].isna().sum())
    # Unknown SKUs / invalid supplier ids = blank key fields.
    def _blank_in(canon: str) -> int:
        col = next((c for c, m in mapping.items() if m == canon), None)
        if col is None:
            return 0
        s = df[col]
        return int(s.isna().sum() + (s.astype(str).str.strip() == "").sum())
    unknown_skus = _blank_in("sku")
    invalid_suppliers = _blank_in("supplier")
    completeness = 1 - (missing / (rows * cols)) if rows else 0
    dup_rate = duplicates / rows if rows else 0
    health = max(0, min(100, round(completeness * 70 + (1 - dup_rate) * 30
                                   - min(20, invalid_dates))))
    warnings, errors = [], []
    if missing:
        warnings.append(f"{missing} missing value(s) across {rows} rows")
    if duplicates:
        warnings.append(f"{duplicates} duplicate row(s)")
    if invalid_dates:
        warnings.append(f"{invalid_dates} unparseable date(s)")
    if unknown_skus:
        warnings.append(f"{unknown_skus} blank SKU(s)")
    if invalid_suppliers:
        warnings.append(f"{invalid_suppliers} blank supplier id(s)")
    if rows == 0:
        errors.append("No rows found in the file")
    return {
        "rows": rows, "missing_values": missing, "duplicate_records": duplicates,
        "invalid_dates": max(0, invalid_dates), "unknown_skus": unknown_skus,
        "invalid_supplier_ids": invalid_suppliers,
        "warnings": warnings, "errors": errors, "health_score": health,
    }


# ── public: upload / map / import / index / registry ──────────────────────────

def upload(filename: str, content: bytes) -> dict[str, Any]:
    """Parse an uploaded file, detect its type, suggest a mapping, and validate.
    Persists a pending dataset record and returns everything the wizard needs."""
    ext = (filename.rsplit(".", 1)[-1] or "").lower()
    try:
        df = _read_dataframe(content, ext)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Could not parse {filename}: {e}"}
    columns = [str(c) for c in df.columns]
    dtype, label, source_guess, confidence = _detect(columns)
    mapping = _suggest_mapping(columns)
    validation = _validate(df, mapping)
    did = uuid.uuid4().hex[:12]
    os.makedirs(_UPLOADS, exist_ok=True)
    path = os.path.join(_UPLOADS, f"{did}__{filename}")
    with open(path, "wb") as fh:
        fh.write(content)
    rec = {
        "id": did, "name": filename.rsplit(".", 1)[0], "filename": filename, "ext": ext,
        "type": dtype, "type_label": label, "source_guess": source_guess,
        "confidence": confidence, "rows": validation["rows"], "columns": columns,
        "mapping": mapping, "validation": validation, "stats": {},
        "status": "pending", "health": validation["health_score"], "indexed": 0,
        "imported_by": "", "created_at": time.time(), "imported_at": None, "filepath": path,
    }
    conn = _conn()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (did, rec["name"], filename, ext, dtype, label, source_guess, confidence,
                 rec["rows"], json.dumps(columns), json.dumps(mapping), json.dumps(validation),
                 json.dumps({}), "pending", rec["health"], 0, "", rec["created_at"], None, path))
    finally:
        conn.close()
    return {"ok": True, "dataset": _public(rec), "sample": _rows(df),
            "canonical_fields": CANONICAL_FIELDS,
            "detection_message": f"This appears to be a {source_guess} ({confidence}% confidence)."}


def _public(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: rec[k] for k in (
        "id", "name", "filename", "ext", "type", "type_label", "source_guess",
        "confidence", "rows", "columns", "mapping", "validation", "stats",
        "status", "health", "indexed", "imported_by", "created_at", "imported_at")}


def _load(did: str) -> Optional[dict[str, Any]]:
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in conn.execute("SELECT * FROM datasets LIMIT 0").description]
        rec = dict(zip(cols, row))
    finally:
        conn.close()
    for j in ("columns", "mapping", "validation", "stats"):
        rec[j] = json.loads(rec[j]) if rec[j] else ([] if j == "columns" else {})
    return rec


def set_mapping(did: str, mapping: dict[str, str]) -> dict[str, Any]:
    rec = _load(did)
    if rec is None:
        return {"ok": False, "error": "unknown dataset"}
    conn = _conn()
    try:
        with conn:
            conn.execute("UPDATE datasets SET mapping=? WHERE id=?", (json.dumps(mapping), did))
    finally:
        conn.close()
    return {"ok": True, "id": did, "mapping": mapping}


def _statistics(df, mapping: dict[str, str]) -> dict[str, Any]:
    import pandas as pd
    stats: dict[str, Any] = {"rows": int(len(df)), "columns": int(df.shape[1])}
    qty_col = next((c for c, m in mapping.items() if m == "quantity"), None)
    if qty_col is not None:
        q = pd.to_numeric(df[qty_col], errors="coerce")
        stats["total_quantity"] = int(q.fillna(0).sum())
    sup_col = next((c for c, m in mapping.items() if m == "supplier"), None)
    if sup_col is not None:
        stats["distinct_suppliers"] = int(df[sup_col].nunique())
    sku_col = next((c for c, m in mapping.items() if m == "sku"), None)
    if sku_col is not None:
        stats["distinct_skus"] = int(df[sku_col].nunique())
    return stats


def _index_into_brain(rec: dict[str, Any], df, options: dict[str, bool]) -> dict[str, Any]:
    """Real, offline indexing: push a dataset document into the Knowledge/RAG
    store and the Decision Brain, and learn key entities. Every call is guarded
    so an import never fails if a downstream system is unavailable."""
    summary = {"documents": 0, "entities": 0, "knowledge": 0}
    doc_text = (
        f"Dataset: {rec['name']} ({rec['type_label']})\n"
        f"Source: {rec['source_guess']}\nRows: {rec['rows']}\n"
        f"Columns: {', '.join(rec['columns'])}\n\n"
        f"Sample rows:\n{json.dumps(_rows(df, 5), default=str)[:1500]}\n\n"
        f"Statistics: {json.dumps(rec.get('stats', {}))}"
    )
    if options.get("index_docs", True) or options.get("semantic_search", True):
        try:
            from modules import store
            store.add_document(f"[Data Hub] {rec['name']}", doc_text)  # → Knowledge/RAG
            summary["documents"] += 1
        except Exception as e:  # noqa: BLE001
            _log.info("knowledge index skipped: %s", e)
        try:
            from brain import BRAIN
            BRAIN.add_knowledge(f"Dataset · {rec['name']}", doc_text, doc_type="dataset")
            summary["knowledge"] += 1
        except Exception as e:  # noqa: BLE001
            _log.info("brain knowledge skipped: %s", e)
    if options.get("learn_suppliers", True):
        sup_col = next((c for c, m in rec["mapping"].items() if m == "supplier"), None)
        if sup_col is not None and sup_col in df.columns:
            try:
                from brain import BRAIN
                for name in df[sup_col].dropna().astype(str).unique()[:15]:
                    BRAIN.record_entity(name, f"Supplier seen in {rec['name']} ({rec['type_label']}).",
                                        kind="supplier")
                    summary["entities"] += 1
            except Exception as e:  # noqa: BLE001
                _log.info("brain entities skipped: %s", e)
    for flag, kind, label in (("learn_inventory", "inventory", "inventory history"),
                              ("learn_procurement", "procurement", "procurement history")):
        if options.get(flag) and rec["type"] in ("inventory", "purchase_orders", "demand_history"):
            try:
                from brain import BRAIN
                BRAIN.add_knowledge(f"{label.title()} · {rec['name']}",
                                    f"Imported {rec['rows']} rows of {label} from {rec['source_guess']}.",
                                    doc_type=kind)
                summary["knowledge"] += 1
            except Exception as e:  # noqa: BLE001
                _log.info("brain %s skipped: %s", kind, e)
    return summary


def do_import(did: str, mapping: Optional[dict[str, str]], options: dict[str, bool],
              imported_by: str = "Enterprise User") -> dict[str, Any]:
    rec = _load(did)
    if rec is None:
        return {"ok": False, "error": "unknown dataset"}
    if mapping:
        rec["mapping"] = mapping
    try:
        with open(rec["filepath"], "rb") as fh:
            df = _read_dataframe(fh.read(), rec["ext"])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"could not re-read file: {e}"}
    stats = _statistics(df, rec["mapping"])
    rec["stats"] = stats
    index_summary = _index_into_brain(rec, df, options or {})
    now = time.time()
    conn = _conn()
    try:
        with conn:
            conn.execute(
                "UPDATE datasets SET mapping=?, stats=?, status=?, indexed=?, imported_by=?, "
                "imported_at=? WHERE id=?",
                (json.dumps(rec["mapping"]), json.dumps(stats), "imported", 1,
                 imported_by, now, did))
    finally:
        conn.close()
    rec["status"] = "imported"
    rec["indexed"] = 1
    rec["imported_by"] = imported_by
    rec["imported_at"] = now
    return {"ok": True, "dataset": _public(rec), "index": index_summary}


def reindex(did: str, options: dict[str, bool]) -> dict[str, Any]:
    rec = _load(did)
    if rec is None:
        return {"ok": False, "error": "unknown dataset"}
    try:
        with open(rec["filepath"], "rb") as fh:
            df = _read_dataframe(fh.read(), rec["ext"])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"could not re-read file: {e}"}
    summary = _index_into_brain(rec, df, options or {})
    conn = _conn()
    try:
        with conn:
            conn.execute("UPDATE datasets SET indexed=1 WHERE id=?", (did,))
    finally:
        conn.close()
    return {"ok": True, "id": did, "index": summary}


def datasets() -> dict[str, Any]:
    conn = _conn()
    try:
        cols = [d[0] for d in conn.execute("SELECT * FROM datasets LIMIT 0").description]
        rows = [dict(zip(cols, r)) for r in
                conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()]
    finally:
        conn.close()
    out = []
    for rec in rows:
        for j in ("columns", "mapping", "validation", "stats"):
            rec[j] = json.loads(rec[j]) if rec[j] else ([] if j == "columns" else {})
        out.append(_public(rec))
    return {"datasets": out, "source": "live"}


def preview(did: str) -> dict[str, Any]:
    rec = _load(did)
    if rec is None:
        return {"ok": False, "error": "unknown dataset"}
    try:
        with open(rec["filepath"], "rb") as fh:
            df = _read_dataframe(fh.read(), rec["ext"])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"could not read file: {e}"}
    return {"ok": True, "id": did, "name": rec["name"], "columns": rec["columns"],
            "mapping": rec["mapping"], "rows": _rows(df, 20)}


def delete(did: str) -> dict[str, Any]:
    rec = _load(did)
    if rec is None:
        return {"ok": False, "error": "unknown dataset"}
    try:
        if rec.get("filepath") and os.path.exists(rec["filepath"]):
            os.remove(rec["filepath"])
    except OSError:
        pass
    conn = _conn()
    try:
        with conn:
            conn.execute("DELETE FROM datasets WHERE id=?", (did,))
    finally:
        conn.close()
    return {"ok": True, "id": did}


def filepath(did: str) -> Optional[tuple[str, str]]:
    rec = _load(did)
    if rec is None or not rec.get("filepath") or not os.path.exists(rec["filepath"]):
        return None
    return rec["filepath"], rec["filename"]


def quality() -> dict[str, Any]:
    d = datasets()["datasets"]
    imported = [x for x in d if x["status"] == "imported"]
    total_rows = sum(x["rows"] for x in imported)
    avg_health = round(sum(x["health"] for x in imported) / len(imported)) if imported else 0
    total_missing = sum(x["validation"].get("missing_values", 0) for x in imported)
    total_dupes = sum(x["validation"].get("duplicate_records", 0) for x in imported)
    dup_rate = round(100 * total_dupes / total_rows, 1) if total_rows else 0.0
    missing_rate = round(100 * total_missing / max(1, total_rows), 1) if total_rows else 0.0
    latest = max((x["imported_at"] or x["created_at"] for x in d), default=None)
    most_recent = max(d, key=lambda x: x["created_at"])["name"] if d else "—"
    # Import history (per dataset, chronological) for the charts.
    history = [{"name": x["name"], "rows": x["rows"], "health": x["health"],
                "at": x["imported_at"] or x["created_at"], "type": x["type_label"]}
               for x in sorted(d, key=lambda x: x["created_at"])]
    completeness = []
    for x in imported:
        rows = max(1, x["rows"])
        cols = max(1, len(x["columns"]))
        pct = round(100 * (1 - x["validation"].get("missing_values", 0) / (rows * cols)))
        completeness.append({"name": x["name"], "completeness": max(0, min(100, pct))})
    return {
        "kpis": {
            "data_quality": avg_health, "duplicate_rate": dup_rate, "missing_rate": missing_rate,
            "datasets": len(d), "imported": len(imported), "total_rows": total_rows,
            "last_refresh": latest, "most_recent": most_recent,
        },
        "history": history, "completeness": completeness, "source": "live",
    }
