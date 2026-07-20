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

from ai.router import AI, AIRouter


@pytest.fixture(autouse=True)
def _offline_ai():
    """Every test starts with an AI router that has no live providers, so
    AI.ask/AI.embed degrade deterministically (no network)."""
    AI.configure(AIRouter(providers={}, offline_handler=lambda cap, msgs: None))
    yield
    AI._router = None
