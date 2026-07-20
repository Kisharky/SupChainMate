"""
ai/observability.py
Structured observability for every AI request.

Records one row per call — timestamp, capability, model, provider, latency,
tokens, cached/fallback flags, and errors — to SQLite, and exposes recent()
and stats() for the observability panel. This is the router's memory sink;
it never raises (observability must not break a request).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from typing import Optional

import config
from ai.types import AIResponse
from modules import store

_log = config.get_logger(__name__)


@dataclass
class RequestLog:
    ts: str
    task: str
    capability: str
    model: Optional[str]
    provider: Optional[str]
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ok: bool
    cached: bool
    fell_back: bool
    error: Optional[str]


def _conn() -> Optional[sqlite3.Connection]:
    conn = store._conn()
    if conn is None:
        return None
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS ai_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, task TEXT, capability TEXT, model TEXT,
            provider TEXT, latency_ms INTEGER, prompt_tokens INTEGER,
            completion_tokens INTEGER, total_tokens INTEGER,
            ok INTEGER, cached INTEGER, fell_back INTEGER, error TEXT)""")
        return conn
    except sqlite3.Error as e:
        _log.warning("ai_requests table unavailable: %s", e)
        conn.close()
        return None


def record(task: str, resp: AIResponse) -> None:
    """Persist one AI request. Called by the router after every ask()."""
    entry = RequestLog(
        ts=store._utcnow(), task=task, capability=resp.capability,
        model=resp.model_used, provider=resp.provider, latency_ms=resp.latency_ms,
        prompt_tokens=resp.usage.prompt_tokens,
        completion_tokens=resp.usage.completion_tokens,
        total_tokens=resp.usage.total_tokens, ok=resp.ok, cached=resp.cached,
        fell_back=resp.fell_back, error=resp.error)
    _log.info("AI %s · %s · %s · %dms · %dtok%s%s%s", entry.capability,
              entry.model or "offline", "ok" if entry.ok else "FAIL",
              entry.latency_ms, entry.total_tokens,
              " · cached" if entry.cached else "",
              " · fallback" if entry.fell_back else "",
              f" · {entry.error}" if entry.error else "")
    conn = _conn()
    if conn is None:
        return
    try:
        with conn:
            conn.execute(
                "INSERT INTO ai_requests (ts, task, capability, model, provider, "
                "latency_ms, prompt_tokens, completion_tokens, total_tokens, ok, "
                "cached, fell_back, error) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (entry.ts, entry.task, entry.capability, entry.model, entry.provider,
                 entry.latency_ms, entry.prompt_tokens, entry.completion_tokens,
                 entry.total_tokens, int(entry.ok), int(entry.cached),
                 int(entry.fell_back), entry.error))
    except sqlite3.Error as e:
        _log.warning("ai_requests insert failed: %s", e)
    finally:
        conn.close()


def recent(limit: int = 100) -> list[dict]:
    conn = _conn()
    if conn is None:
        return []
    try:
        cols = ["ts", "task", "capability", "model", "provider", "latency_ms",
                "prompt_tokens", "completion_tokens", "total_tokens", "ok",
                "cached", "fell_back", "error"]
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM ai_requests ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(zip(cols, r)) for r in rows]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def stats() -> dict:
    """Aggregate KPIs for the observability panel."""
    conn = _conn()
    if conn is None:
        return {"requests": 0}
    try:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(total_tokens),0), "
            "COALESCE(AVG(latency_ms),0), COALESCE(SUM(ok),0), "
            "COALESCE(SUM(cached),0), COALESCE(SUM(fell_back),0) "
            "FROM ai_requests").fetchone()
        total, tokens, avg_latency, oks, cached, fell = row
        by_cap = conn.execute(
            "SELECT capability, COUNT(*), COALESCE(SUM(total_tokens),0) "
            "FROM ai_requests GROUP BY capability ORDER BY 2 DESC").fetchall()
        return {
            "requests": total,
            "total_tokens": int(tokens),
            "avg_latency_ms": round(float(avg_latency), 1),
            "success_rate": round(oks / total * 100, 1) if total else None,
            "cache_hit_rate": round(cached / total * 100, 1) if total else None,
            "fallback_rate": round(fell / total * 100, 1) if total else None,
            "by_capability": [{"capability": c, "requests": n, "tokens": int(t)}
                              for c, n, t in by_cap],
        }
    except sqlite3.Error:
        return {"requests": 0}
    finally:
        conn.close()
