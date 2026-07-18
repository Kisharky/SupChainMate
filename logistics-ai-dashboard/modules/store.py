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

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_BASE_DIR, "data", "supchainmate.db")


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
        return conn
    except sqlite3.Error:
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
