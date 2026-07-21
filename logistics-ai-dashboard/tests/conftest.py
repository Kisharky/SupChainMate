"""
Shared test fixtures.

Critically: reset the process-wide AI facade to a provider-less, offline
router before every test so nothing accidentally calls the real NVIDIA
endpoint. Tests that exercise the AI layer inject their own fake providers.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Domain tests exercise business logic, not the HTTP auth gate — disable auth by
# default so endpoint tests run tokenless. tests/test_auth.py opts back in.
os.environ.setdefault("AUTH_ENABLED", "false")

from ai.router import AI, AIRouter
from modules import store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point SQLite at a throwaway file so no test touches the real DB.
    Tests that set their own tmp_db override this (both are autouse; the
    module-level fixture runs after this conftest one)."""
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "conftest.db"))


@pytest.fixture(autouse=True)
def _offline_ai():
    """Every test starts with an AI router that has no live providers, so
    AI.ask/AI.embed degrade deterministically (no network)."""
    AI.configure(AIRouter(providers={}, offline_handler=lambda cap, msgs: None))
    yield
    AI._router = None
