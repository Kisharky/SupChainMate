"""
api/db.py — SQLAlchemy engine/session for application (auth/identity) state.

Dual-backend by configuration: defaults to a local SQLite file for offline/demo
mode, and switches to PostgreSQL simply by setting ``DATABASE_URL`` — no code
change. This holds identity state (users, refresh tokens); the analytics/demo
data keeps its existing lightweight SQLite store, so nothing in the domain
modules changes.

    # offline demo (default)
    DATABASE_URL=sqlite:///data/supchainmate_app.db
    # production
    DATABASE_URL=postgresql+psycopg2://user:pass@postgres:5432/supchainmate
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/supchainmate_app.db")

# SQLite needs check_same_thread=False for FastAPI's threadpool; Postgres doesn't.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_session():
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they don't exist (idempotent). Imports models first."""
    from api.auth import models  # noqa: F401 — registers ORM tables on Base
    Base.metadata.create_all(bind=engine)
