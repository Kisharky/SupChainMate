"""
modules/store.py
SQLite persistence — retail products and app settings survive restarts.

The DB lives at data/supchainmate.db (gitignored alongside the demo CSVs).
On hosted platforms with ephemeral disks this degrades to session-only
behaviour without erroring.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional

import config

DB_PATH = config.DB_PATH
_log = config.get_logger(__name__)


def _conn() -> Optional[sqlite3.Connection]:
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute("""CREATE TABLE IF NOT EXISTS retail_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS kpi_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            payload TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS recommendations (
            rec_key TEXT PRIMARY KEY,
            created_ts TEXT NOT NULL,
            payload TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            decided_ts TEXT,
            decided_by TEXT,
            note TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            event TEXT NOT NULL,
            rec_key TEXT,
            details TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            workflow TEXT NOT NULL,
            agent TEXT NOT NULL,
            confidence REAL,
            outputs TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            name TEXT NOT NULL,
            content TEXT NOT NULL)""")
        return conn
    except sqlite3.Error as e:
        _log.warning("SQLite unavailable at %s: %s", DB_PATH, e)
        return None


def save_retail_products(products: list[dict]) -> bool:
    conn = _conn()
    if conn is None:
        return False
    try:
        with conn:
            conn.execute("DELETE FROM retail_products")
            conn.executemany(
                "INSERT INTO retail_products (payload) VALUES (?)",
                [(json.dumps(p),) for p in products],
            )
        return True
    except (sqlite3.Error, TypeError):
        return False
    finally:
        conn.close()


def load_retail_products() -> list[dict]:
    conn = _conn()
    if conn is None:
        return []
    try:
        rows = conn.execute("SELECT payload FROM retail_products ORDER BY id").fetchall()
        return [json.loads(r[0]) for r in rows]
    except (sqlite3.Error, json.JSONDecodeError):
        return []
    finally:
        conn.close()


def save_kpi_snapshot(kpis: dict) -> bool:
    """Append a timestamped KPI snapshot (health score, on-time %, etc.)."""
    from datetime import datetime, timezone
    conn = _conn()
    if conn is None:
        return False
    try:
        with conn:
            conn.execute(
                "INSERT INTO kpi_snapshots (ts, payload) VALUES (?, ?)",
                (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), json.dumps(kpis)),
            )
        return True
    except (sqlite3.Error, TypeError):
        return False
    finally:
        conn.close()


def load_kpi_snapshots(limit: int = 100) -> list[dict]:
    """Return snapshots oldest-first, each with its timestamp under 'ts'."""
    conn = _conn()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT ts, payload FROM kpi_snapshots ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for ts, payload in reversed(rows):
            d = json.loads(payload)
            d["ts"] = ts
            out.append(d)
        return out
    except (sqlite3.Error, json.JSONDecodeError):
        return []
    finally:
        conn.close()


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def save_recommendation(rec_key: str, payload: dict) -> bool:
    """Insert a recommendation if that key doesn't already exist (dedupe)."""
    conn = _conn()
    if conn is None:
        return False
    try:
        with conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO recommendations (rec_key, created_ts, payload) "
                "VALUES (?, ?, ?)",
                (rec_key, _utcnow(), json.dumps(payload)),
            )
        return cur.rowcount > 0
    except (sqlite3.Error, TypeError) as e:
        _log.warning("save_recommendation failed: %s", e)
        return False
    finally:
        conn.close()


def load_recommendations(status: str | None = None, limit: int = 200) -> list[dict]:
    """Recommendations newest-first, optionally filtered by status."""
    conn = _conn()
    if conn is None:
        return []
    try:
        q = ("SELECT rec_key, created_ts, payload, status, decided_ts, decided_by, note "
             "FROM recommendations ")
        args: tuple = ()
        if status:
            q += "WHERE status = ? "
            args = (status,)
        q += "ORDER BY created_ts DESC, rec_key LIMIT ?"
        rows = conn.execute(q, args + (limit,)).fetchall()
        out = []
        for rec_key, created_ts, payload, st_, decided_ts, decided_by, note in rows:
            d = json.loads(payload)
            d.update({"rec_key": rec_key, "created_ts": created_ts, "status": st_,
                      "decided_ts": decided_ts, "decided_by": decided_by, "note": note})
            out.append(d)
        return out
    except (sqlite3.Error, json.JSONDecodeError) as e:
        _log.warning("load_recommendations failed: %s", e)
        return []
    finally:
        conn.close()


def update_recommendation(rec_key: str, status: str, decided_by: str = "user",
                          note: str = "") -> bool:
    """Record a human decision on a recommendation."""
    conn = _conn()
    if conn is None:
        return False
    try:
        with conn:
            cur = conn.execute(
                "UPDATE recommendations SET status=?, decided_ts=?, decided_by=?, note=? "
                "WHERE rec_key=?",
                (status, _utcnow(), decided_by, note, rec_key),
            )
        return cur.rowcount > 0
    except sqlite3.Error as e:
        _log.warning("update_recommendation failed: %s", e)
        return False
    finally:
        conn.close()


def log_event(actor: str, event: str, details: str = "", rec_key: str | None = None) -> bool:
    """Append an immutable audit-trail entry."""
    conn = _conn()
    if conn is None:
        return False
    try:
        with conn:
            conn.execute(
                "INSERT INTO audit_log (ts, actor, event, rec_key, details) "
                "VALUES (?, ?, ?, ?, ?)",
                (_utcnow(), actor, event, rec_key, details),
            )
        return True
    except sqlite3.Error as e:
        _log.warning("log_event failed: %s", e)
        return False
    finally:
        conn.close()


def load_audit_log(limit: int = 300) -> list[dict]:
    """Audit entries newest-first."""
    conn = _conn()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT ts, actor, event, rec_key, details FROM audit_log "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "actor": r[1], "event": r[2],
                 "rec_key": r[3], "details": r[4]} for r in rows]
    except sqlite3.Error as e:
        _log.warning("load_audit_log failed: %s", e)
        return []
    finally:
        conn.close()


def save_agent_run(workflow: str, agent: str, confidence: float, outputs: dict) -> bool:
    """Persist one agent's run outputs (the agents' long-term memory)."""
    conn = _conn()
    if conn is None:
        return False
    try:
        with conn:
            conn.execute(
                "INSERT INTO agent_runs (ts, workflow, agent, confidence, outputs) "
                "VALUES (?, ?, ?, ?, ?)",
                (_utcnow(), workflow, agent, float(confidence), json.dumps(outputs, default=str)))
        return True
    except (sqlite3.Error, TypeError) as e:
        _log.warning("save_agent_run failed: %s", e)
        return False
    finally:
        conn.close()


def last_agent_runs(before_latest: bool = False) -> dict[str, dict]:
    """
    Most recent persisted outputs per agent. With before_latest=True, skip
    each agent's newest run and return the one before it (for run-over-run
    comparisons after the current run was saved).
    """
    conn = _conn()
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT agent, ts, confidence, outputs FROM agent_runs "
            "ORDER BY id DESC LIMIT 400").fetchall()
        out: dict[str, dict] = {}
        skipped: set[str] = set()
        for agent, ts, confidence, outputs in rows:
            if before_latest and agent not in skipped:
                skipped.add(agent)
                continue
            if agent not in out:
                out[agent] = {"ts": ts, "confidence": confidence,
                              "outputs": json.loads(outputs)}
        return out
    except (sqlite3.Error, json.JSONDecodeError) as e:
        _log.warning("last_agent_runs failed: %s", e)
        return {}
    finally:
        conn.close()


def add_document(name: str, content: str) -> bool:
    conn = _conn()
    if conn is None:
        return False
    try:
        with conn:
            conn.execute("INSERT INTO documents (ts, name, content) VALUES (?, ?, ?)",
                         (_utcnow(), name, content))
        return True
    except sqlite3.Error as e:
        _log.warning("add_document failed: %s", e)
        return False
    finally:
        conn.close()


def load_documents() -> list[dict]:
    conn = _conn()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT id, ts, name, content FROM documents ORDER BY id").fetchall()
        return [{"id": r[0], "ts": r[1], "name": r[2], "content": r[3]} for r in rows]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def delete_document(doc_id: int) -> bool:
    conn = _conn()
    if conn is None:
        return False
    try:
        with conn:
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def save_setting(key: str, value) -> bool:
    conn = _conn()
    if conn is None:
        return False
    try:
        with conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )
        return True
    except (sqlite3.Error, TypeError):
        return False
    finally:
        conn.close()


def load_setting(key: str, default=None):
    conn = _conn()
    if conn is None:
        return default
    try:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default
    except (sqlite3.Error, json.JSONDecodeError):
        return default
    finally:
        conn.close()
