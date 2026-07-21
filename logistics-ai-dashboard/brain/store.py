"""
brain/store.py — the embedded vector store. A local SQLite table holds every
memory record plus its embedding (float32 blob). No external vector database is
required, so the Brain runs fully offline / on-prem. The store is intentionally
swappable — the retriever only needs ``all(kinds)`` and the embeddings.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np

import config
from brain.embeddings import default_embedder
from brain.schemas import MemoryKind, MemoryRecord

_log = config.get_logger(__name__)
_DB = getattr(config, "DB_PATH", "data/supchainmate.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(kind: str, title: str, content: str) -> str:
    return hashlib.sha1(f"{kind}|{title}|{content}".encode()).hexdigest()[:16]


def _conn() -> Optional[sqlite3.Connection]:
    try:
        c = sqlite3.connect(_DB)
        c.execute(
            "CREATE TABLE IF NOT EXISTS brain_memory ("
            "id TEXT PRIMARY KEY, kind TEXT, title TEXT, content TEXT, "
            "metadata TEXT, ts TEXT, source TEXT, dim INTEGER, embedding BLOB)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_brain_kind ON brain_memory(kind)")
        return c
    except sqlite3.Error as e:  # noqa: BLE001
        _log.warning("brain store unavailable: %s", e)
        return None


class MemoryStore:
    def __init__(self) -> None:
        self.embedder = default_embedder()

    # ---- write ----
    def add(self, kind: MemoryKind, title: str, content: str,
            metadata: Optional[dict] = None, source: str = "") -> str:
        rid = _key(kind.value, title, content)
        conn = _conn()
        if conn is None:
            return rid
        try:
            vec = self.embedder.embed(f"{title}\n{content}")
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO brain_memory "
                    "(id, kind, title, content, metadata, ts, source, dim, embedding) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (rid, kind.value, title, content[:200000],
                     json.dumps(metadata or {}), _now(), source,
                     len(vec), vec.tobytes()))
        except sqlite3.Error as e:  # noqa: BLE001
            _log.warning("brain add failed: %s", e)
        finally:
            conn.close()
        return rid

    # ---- read ----
    def all(self, kinds: Optional[list[MemoryKind]] = None
            ) -> list[tuple[MemoryRecord, np.ndarray]]:
        conn = _conn()
        if conn is None:
            return []
        try:
            q = ("SELECT id, kind, title, content, metadata, ts, source, embedding "
                 "FROM brain_memory")
            args: tuple = ()
            if kinds:
                q += " WHERE kind IN (%s)" % ",".join("?" * len(kinds))
                args = tuple(k.value for k in kinds)
            rows = conn.execute(q, args).fetchall()
            out = []
            for r in rows:
                rec = MemoryRecord(id=r[0], kind=MemoryKind(r[1]), title=r[2],
                                   content=r[3], metadata=json.loads(r[4] or "{}"),
                                   ts=r[5], source=r[6])
                vec = np.frombuffer(r[7], dtype="float32") if r[7] else np.zeros(self.embedder.dim, "float32")
                out.append((rec, vec))
            return out
        except sqlite3.Error:  # noqa: BLE001
            return []
        finally:
            conn.close()

    def recent(self, limit: int = 30, kinds: Optional[list[MemoryKind]] = None) -> list[MemoryRecord]:
        conn = _conn()
        if conn is None:
            return []
        try:
            q = ("SELECT id, kind, title, content, metadata, ts, source FROM brain_memory")
            args: tuple = ()
            if kinds:
                q += " WHERE kind IN (%s)" % ",".join("?" * len(kinds))
                args = tuple(k.value for k in kinds)
            q += " ORDER BY ts DESC LIMIT ?"
            rows = conn.execute(q, args + (limit,)).fetchall()
            return [MemoryRecord(id=r[0], kind=MemoryKind(r[1]), title=r[2], content=r[3],
                                 metadata=json.loads(r[4] or "{}"), ts=r[5], source=r[6])
                    for r in rows]
        except sqlite3.Error:  # noqa: BLE001
            return []
        finally:
            conn.close()

    def stats(self) -> dict:
        conn = _conn()
        if conn is None:
            return {"total": 0, "by_kind": {}}
        try:
            rows = conn.execute("SELECT kind, COUNT(*) FROM brain_memory GROUP BY kind").fetchall()
            by_kind = {r[0]: r[1] for r in rows}
            return {"total": sum(by_kind.values()), "by_kind": by_kind,
                    "embedder": self.embedder.name, "dim": self.embedder.dim}
        except sqlite3.Error:  # noqa: BLE001
            return {"total": 0, "by_kind": {}}
        finally:
            conn.close()

    def has(self, rid: str) -> bool:
        conn = _conn()
        if conn is None:
            return False
        try:
            return conn.execute("SELECT 1 FROM brain_memory WHERE id=?", (rid,)).fetchone() is not None
        finally:
            conn.close()
