"""
planner/memory.py — records every plan (objective, graph, capabilities, outputs,
recommendation, predicted vs actual outcome, timestamp) to SQLite. This is the
substrate for continuous learning: predicted outcomes are stored now; actual
outcomes can be reconciled later.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

import config

_log = config.get_logger(__name__)
_DB = getattr(config, "DB_PATH", "data/supchainmate.db")


def _conn() -> Optional[sqlite3.Connection]:
    try:
        c = sqlite3.connect(_DB)
        c.execute(
            "CREATE TABLE IF NOT EXISTS planner_runs ("
            "id TEXT PRIMARY KEY, ts TEXT, objective TEXT, graph TEXT, "
            "capabilities TEXT, outputs TEXT, recommendation TEXT, "
            "predicted TEXT, actual TEXT)")
        return c
    except sqlite3.Error as e:  # noqa: BLE001
        _log.warning("planner memory unavailable: %s", e)
        return None


class PlannerMemory:
    def record(self, *, objective: str, graph, capabilities, outputs: dict,
               recommendation: str, predicted: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        conn = _conn()
        if conn is None:
            return rid
        try:
            with conn:
                conn.execute(
                    "INSERT INTO planner_runs (id, ts, objective, graph, capabilities, "
                    "outputs, recommendation, predicted, actual) VALUES (?,?,?,?,?,?,?,?,?)",
                    (rid, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     objective, json.dumps(graph), json.dumps(capabilities),
                     json.dumps(outputs)[:200000], recommendation,
                     json.dumps(predicted), None))
        except sqlite3.Error as e:  # noqa: BLE001
            _log.warning("planner record failed: %s", e)
        finally:
            conn.close()
        return rid

    def recent(self, limit: int = 20) -> list[dict]:
        conn = _conn()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "SELECT id, ts, objective, capabilities, recommendation, predicted "
                "FROM planner_runs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
            return [{"id": r[0], "ts": r[1], "objective": r[2],
                     "capabilities": json.loads(r[3] or "[]"),
                     "recommendation": r[4], "predicted": json.loads(r[5] or "{}")}
                    for r in rows]
        except sqlite3.Error:  # noqa: BLE001
            return []
        finally:
            conn.close()

    def reconcile(self, run_id: str, actual: dict) -> bool:
        """Attach a realised outcome to a past plan (future continuous learning)."""
        conn = _conn()
        if conn is None:
            return False
        try:
            with conn:
                cur = conn.execute("UPDATE planner_runs SET actual=? WHERE id=?",
                                   (json.dumps(actual), run_id))
            return cur.rowcount > 0
        except sqlite3.Error:  # noqa: BLE001
            return False
        finally:
            conn.close()
