"""
modules/disputes.py
Dispute Manager — end-to-end lifecycle for billing disputes.

Closes the loop the cost audit opens: flagged charges become dispute
records that move through OPEN → SENT → RESOLVED / WRITTEN_OFF, with
recovered amounts tracked and every transition audit-logged. Adapts the
enterprise dispute-management pattern into an open implementation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Optional

import pandas as pd

import config
from modules import store

_log = config.get_logger(__name__)

STATUSES = ("OPEN", "SENT", "RESOLVED", "WRITTEN_OFF")
_TRANSITIONS = {"OPEN": {"SENT", "WRITTEN_OFF"},
                "SENT": {"RESOLVED", "WRITTEN_OFF"},
                "RESOLVED": set(), "WRITTEN_OFF": set()}


def _conn() -> Optional[sqlite3.Connection]:
    conn = store._conn()
    if conn is None:
        return None
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS disputes (
            dispute_key TEXT PRIMARY KEY,
            created_ts TEXT NOT NULL,
            shipment_id TEXT,
            carrier TEXT,
            reason TEXT,
            amount REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            updated_ts TEXT,
            recovered REAL DEFAULT 0)""")
        return conn
    except sqlite3.Error as e:
        _log.warning("disputes table unavailable: %s", e)
        conn.close()
        return None


def open_disputes_from_audit(flagged: pd.DataFrame, limit: int = 25) -> int:
    """Create dispute records from the audit's flagged charges (deduped)."""
    if flagged is None or not len(flagged):
        return 0
    conn = _conn()
    if conn is None:
        return 0
    created = 0
    try:
        with conn:
            for _, row in flagged.head(limit).iterrows():
                sid = str(row.get("shipment_id", "?"))
                reason = str(row.get("reason", "flagged charge"))
                amount = float(row.get("overcharge_est", 0) or 0)
                if amount <= 0:
                    continue
                key = hashlib.sha1(f"{sid}|{reason}".encode()).hexdigest()[:16]
                cur = conn.execute(
                    "INSERT OR IGNORE INTO disputes "
                    "(dispute_key, created_ts, shipment_id, carrier, reason, amount) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key, store._utcnow(), sid, str(row.get("carrier", "")),
                     reason, round(amount, 2)))
                if cur.rowcount:
                    created += 1
    except sqlite3.Error as e:
        _log.warning("open_disputes_from_audit failed: %s", e)
        return created
    finally:
        conn.close()
    if created:
        store.log_event("auditor", "disputes_opened", details=f"{created} dispute(s)")
        _log.info("Opened %d dispute(s)", created)
    return created


def list_disputes(status: Optional[str] = None) -> list[dict]:
    conn = _conn()
    if conn is None:
        return []
    try:
        q = ("SELECT dispute_key, created_ts, shipment_id, carrier, reason, amount, "
             "status, updated_ts, recovered FROM disputes ")
        args: tuple = ()
        if status:
            q += "WHERE status=? "
            args = (status,)
        q += "ORDER BY amount DESC"
        cols = ["dispute_key", "created_ts", "shipment_id", "carrier", "reason",
                "amount", "status", "updated_ts", "recovered"]
        return [dict(zip(cols, r)) for r in conn.execute(q, args).fetchall()]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def transition(dispute_key: str, new_status: str,
               recovered: float = 0.0, actor: str = "user") -> bool:
    """Move a dispute through its lifecycle; invalid transitions are rejected."""
    if new_status not in STATUSES:
        raise ValueError(f"Unknown dispute status: {new_status}")
    conn = _conn()
    if conn is None:
        return False
    try:
        row = conn.execute("SELECT status FROM disputes WHERE dispute_key=?",
                           (dispute_key,)).fetchone()
        if row is None:
            return False
        current = row[0]
        if new_status not in _TRANSITIONS.get(current, set()):
            raise ValueError(f"Invalid transition {current} → {new_status}")
        with conn:
            conn.execute(
                "UPDATE disputes SET status=?, updated_ts=?, recovered=? "
                "WHERE dispute_key=?",
                (new_status, store._utcnow(),
                 round(float(recovered), 2) if new_status == "RESOLVED" else 0.0,
                 dispute_key))
    except sqlite3.Error as e:
        _log.warning("dispute transition failed: %s", e)
        return False
    finally:
        conn.close()
    store.log_event(actor, f"dispute_{new_status.lower()}",
                    details=f"{dispute_key}"
                            + (f" recovered ${recovered:,.2f}" if new_status == "RESOLVED" else ""))
    return True


def dispute_kpis() -> dict:
    """Headline numbers: open value, in-flight, recovered, recovery rate."""
    rows = list_disputes()
    resolved = [r for r in rows if r["status"] == "RESOLVED"]
    closed = [r for r in rows if r["status"] in ("RESOLVED", "WRITTEN_OFF")]
    disputed_closed = sum(r["amount"] for r in closed)
    recovered = sum(r["recovered"] for r in resolved)
    return {
        "total": len(rows),
        "open_value": sum(r["amount"] for r in rows if r["status"] == "OPEN"),
        "sent_value": sum(r["amount"] for r in rows if r["status"] == "SENT"),
        "recovered": recovered,
        "recovery_rate_pct": round(recovered / disputed_closed * 100, 1)
        if disputed_closed > 0 else None,
    }
