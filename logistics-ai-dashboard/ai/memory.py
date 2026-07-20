"""
ai/memory.py
Agent memory service — conversational + operational recall, backed by the
existing SQLite store. Agents remember previous recommendations, decisions,
and run outputs without knowing the storage details (dependency inversion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import config
from modules import store

_log = config.get_logger(__name__)


@dataclass
class MemoryRecord:
    kind: str
    ref: str
    payload: dict[str, Any]
    ts: Optional[str] = None


def remember_run(workflow: str, agent: str, confidence: float,
                 outputs: dict[str, Any]) -> bool:
    """Operational memory: persist an agent's run outputs."""
    return store.save_agent_run(workflow, agent, confidence, outputs)


def recall_last_runs(before_latest: bool = False) -> dict[str, dict]:
    """Most recent outputs per agent (agent memory for run-over-run diffs)."""
    return store.last_agent_runs(before_latest=before_latest)


def recall_decisions(limit: int = 100) -> list[dict]:
    """Decision memory: previous approvals/rejections/modifications."""
    return [r for r in store.load_recommendations(limit=limit)
            if r.get("status") != "PENDING"]


def recall_pending() -> list[dict]:
    return store.load_recommendations(status="PENDING")


def deltas(current_outputs: dict[str, dict]) -> list[str]:
    """Numeric run-over-run movement vs the previous persisted run."""
    prev = recall_last_runs(before_latest=False)
    notes: list[str] = []
    for agent, cur in current_outputs.items():
        before = (prev.get(agent) or {}).get("outputs", {})
        for k, now in (cur or {}).items():
            if isinstance(now, (int, float)) and isinstance(before.get(k), (int, float)):
                if before[k] != now:
                    notes.append(f"{agent}.{k}: {before[k]:,.1f} → {now:,.1f}")
    return notes[:8]
