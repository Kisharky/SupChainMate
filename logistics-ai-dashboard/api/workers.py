"""
api/workers.py — the AI Digital Workers cockpit.

Presents SupChainMate's agentic automation as a roster of "digital workers",
inspired by the freight-tech "digital workers complete the back-office" pattern.
Each worker is a **real registered Planner capability**, discovered dynamically
(no hard-coded list) — so the roster grows automatically as capabilities are
registered. The task queue and productivity metrics are representative
(labelled); exceptions escalate into the existing Decision Center.

No new AI engine is introduced: the workers *are* the capabilities the Planner
already orchestrates. Nothing in planner/, ai/, brain/, or optimize/ is modified.
"""

from __future__ import annotations

import hashlib
from typing import Any

from api.services import _safe

# Map a capability's keywords to a business domain shown on the worker card.
_DOMAIN_KEYWORDS = {
    "Forecasting": ("forecast", "demand", "seasonal", "spike"),
    "Inventory": ("inventory", "stock", "reorder", "eoq", "holding", "allocation"),
    "Procurement": ("procure", "carrier", "supplier", "tender", "rate", "sourcing"),
    "Logistics": ("logistics", "route", "shipment", "delivery", "lane", "transport"),
    "Warehouse": ("warehouse", "capacity", "hub", "utilisation", "utilization", "storage"),
    "Commercial": ("price", "pricing", "margin", "revenue", "leakage", "customer"),
    "Knowledge": ("policy", "sop", "contract", "compliance", "knowledge", "recall"),
    "Risk": ("risk", "supplier", "fraud", "disruption"),
}

# Representative task templates (labelled). A real deployment streams these from
# the agents' live run log.
_TASK_TEMPLATES = [
    ("Reconciled carrier invoice against rate confirmation", "auto_completed"),
    ("Recomputed reorder points for A-class SKUs", "auto_completed"),
    ("Drafted repricing proposal for under-margin lines", "awaiting_approval"),
    ("Re-forecast demand after regional spike", "auto_completed"),
    ("Flagged duplicate freight charge for review", "escalated"),
    ("Optimised multi-DC replenishment allocation", "auto_completed"),
    ("Scored new carrier for on-time reliability", "auto_completed"),
    ("Prepared executive brief for Monday review", "awaiting_approval"),
    ("Detected anomalous discount on high-value account", "escalated"),
    ("Matched proof-of-delivery to shipment record", "auto_completed"),
    ("Rebalanced safety stock for seasonal window", "auto_completed"),
    ("Retrieved governing SLA clause for dispute", "auto_completed"),
    ("Negotiated spot rate within guardrails", "awaiting_approval"),
    ("Verified supplier compliance documents", "running"),
]

_STATUS_META = {
    "auto_completed": {"label": "Auto-completed", "status": "good"},
    "awaiting_approval": {"label": "Awaiting approval", "status": "warning"},
    "escalated": {"label": "Escalated to human", "status": "critical"},
    "running": {"label": "Running", "status": "info"},
}


def _seed(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def _domain_for(cap: dict) -> str:
    hay = " ".join(cap.get("keywords", [])) + " " + cap.get("name", "")
    for domain, kws in _DOMAIN_KEYWORDS.items():
        if any(k in hay for k in kws):
            return domain
    return "Operations"


def _titleise(name: str) -> str:
    return name.replace("_", " ").title()


def _worker(cap: dict) -> dict[str, Any]:
    h = _seed(cap["name"])
    conf = float(cap.get("confidence", 0.7))
    # Zero-touch autonomy scales with the capability's own confidence.
    zero_touch = min(97, max(55, round(conf * 100) - (h % 8)))
    tasks_today = 30 + (h % 190)
    exceptions = h % 5
    return {
        "id": cap["name"],
        "name": f"{_titleise(cap['name'])} Worker",
        "skill": cap.get("description", ""),
        "domain": _domain_for(cap),
        "status": "active" if (h % 6) else "idle",
        "zero_touch_pct": zero_touch,
        "tasks_today": tasks_today,
        "exceptions": exceptions,
        "confidence": round(conf, 2),
        "outputs": cap.get("outputs", []),
    }


def _roster() -> list[dict[str, Any]]:
    from planner import PLANNER
    caps = PLANNER.capabilities()
    workers = [_worker(c) for c in caps]
    workers.sort(key=lambda w: w["tasks_today"], reverse=True)
    return workers


def _queue(workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not workers:
        return []
    items: list[dict[str, Any]] = []
    for i, (task, status) in enumerate(_TASK_TEMPLATES):
        w = workers[i % len(workers)]
        h = _seed(task)
        items.append({
            "id": f"TSK-{4200 + i}",
            "worker": w["name"],
            "worker_id": w["id"],
            "domain": w["domain"],
            "task": task,
            "state": status,
            "state_label": _STATUS_META[status]["label"],
            "state_status": _STATUS_META[status]["status"],
            "confidence": min(98, 60 + h % 38),
            "impact_usd": round(1500 + (h % 42000), -2),
            "minutes_ago": 1 + (h % 180),
        })
    items.sort(key=lambda t: t["minutes_ago"])
    return items


def _summary(workers: list[dict[str, Any]], queue: list[dict[str, Any]]) -> dict[str, Any]:
    active = [w for w in workers if w["status"] == "active"]
    tasks_today = sum(w["tasks_today"] for w in workers)
    # Task-weighted zero-touch rate.
    zt = (round(sum(w["zero_touch_pct"] * w["tasks_today"] for w in workers) / tasks_today)
          if tasks_today else 0)
    return {
        "active_workers": len(active),
        "total_workers": len(workers),
        "tasks_automated_today": tasks_today,
        "zero_touch_pct": zt,
        "hours_saved_week": round(tasks_today * 7 * 0.12),   # ~7 min saved per task
        "awaiting_approval": sum(1 for t in queue if t["state"] == "awaiting_approval"),
        "escalated": sum(1 for t in queue if t["state"] == "escalated"),
    }


def cockpit() -> dict[str, Any]:
    """Full cockpit payload: worker roster, productivity summary, and the live
    task queue (representative). Workers are the Planner's real capabilities."""
    def build() -> dict[str, Any]:
        workers = _roster()
        queue = _queue(workers)
        return {
            "workers": workers,
            "summary": _summary(workers, queue),
            "queue": queue,
            "source": "representative",
        }
    return _safe(build, {"workers": [], "summary": {}, "queue": [], "source": "fallback"})
