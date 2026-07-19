"""
modules/events.py
Event-driven automation — conditions in the data trigger agent workflows
without anyone clicking a button.

Detectors are pure functions over the shared context; when one fires, the
orchestrator runs the mapped workflow and the audit log records the chain
event → workflow → recommendations. Detection runs once per data load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

import config
from modules import store
from modules.agents import Orchestrator

_log = config.get_logger(__name__)

AT_RISK_EVENT_THRESHOLD = 20
DEMAND_SPIKE_PCT = 20.0
WORST_GRADE_TRIGGER = ("D",)


@dataclass
class EventRule:
    name: str
    description: str
    workflow: str                                  # workflow to trigger
    detector: Callable[[dict], Optional[str]]      # returns detail when fired


def _supplier_delay(ctx: dict) -> Optional[str]:
    score = ctx.get("scorecard")
    if score is None or not len(score):
        return None
    worst = score.iloc[-1]
    if str(worst.get("Grade")) in WORST_GRADE_TRIGGER:
        return (f"{worst['Carrier']} is grading {worst['Grade']} "
                f"({worst['On-Time %']}% on-time, {int(worst['Late']):,} late)")
    return None


def _inventory_below_threshold(ctx: dict) -> Optional[str]:
    plan = ctx.get("sku_plan")
    if plan is None or "Status" not in getattr(plan, "columns", []):
        return None
    urgent = int(plan["Status"].str.contains("ORDER NOW", na=False).sum())
    if urgent > 0:
        return f"{urgent} SKU(s) at/below reorder point"
    return None


def _demand_spike(ctx: dict) -> Optional[str]:
    daily, fc = ctx.get("daily_df"), ctx.get("forecast_df")
    days = int(ctx.get("days", 7))
    if daily is None or fc is None or not len(daily):
        return None
    avg = float(daily["y"].mean())
    horizon_daily = float(fc["yhat"].tail(days).clip(lower=0).mean())
    if avg > 0 and (horizon_daily - avg) / avg * 100 >= DEMAND_SPIKE_PCT:
        return f"forecast {((horizon_daily - avg) / avg * 100):+.0f}% vs historical average"
    return None


def _at_risk_surge(ctx: dict) -> Optional[str]:
    kpis = ctx.get("kpis") or {}
    at_risk = kpis.get("at_risk", 0) or 0
    if at_risk >= AT_RISK_EVENT_THRESHOLD:
        return f"{at_risk:,} open shipments flagged at risk by the ML model"
    return None


EVENT_RULES: list[EventRule] = [
    EventRule("supplier_delay_detected",
              "Worst carrier grades D → run the logistics review",
              "logistics_review", _supplier_delay),
    EventRule("inventory_below_threshold",
              "SKUs at/below reorder point → run the planning chain",
              "planning_chain", _inventory_below_threshold),
    EventRule("demand_spike",
              f"Forecast ≥{DEMAND_SPIKE_PCT:.0f}% above average → run the planning chain",
              "planning_chain", _demand_spike),
    EventRule("shipment_risk_surge",
              f"≥{AT_RISK_EVENT_THRESHOLD} at-risk shipments → run the logistics review",
              "logistics_review", _at_risk_surge),
]


def detect(ctx: dict) -> list[dict]:
    """Evaluate every rule. Returns fired events: [{name, detail, workflow}]."""
    fired: list[dict] = []
    for rule in EVENT_RULES:
        try:
            detail = rule.detector(ctx)
        except Exception as e:  # a broken detector must not break the load
            _log.warning("Event detector %s failed: %s", rule.name, e)
            continue
        if detail:
            fired.append({"name": rule.name, "detail": detail,
                          "workflow": rule.workflow, "description": rule.description})
    return fired


def run_triggered(orch: Orchestrator, fired: list[dict],
                  ctx: dict[str, Any]) -> list[dict]:
    """
    Run each fired event's workflow (each distinct workflow once), with
    the full audit chain. Returns [{event, workflow, run}].
    """
    results: list[dict] = []
    seen_workflows: set[str] = set()
    for ev in fired:
        store.log_event("events", "event_detected",
                        details=f"{ev['name']}: {ev['detail']}")
        if ev["workflow"] in seen_workflows:
            continue
        seen_workflows.add(ev["workflow"])
        store.log_event("events", "event_triggered_workflow",
                        details=f"{ev['name']} -> {ev['workflow']}")
        run = orch.run_workflow(ev["workflow"], ctx)
        results.append({"event": ev, "run": run})
        _log.info("Event %s triggered workflow %s (%d recommendations)",
                  ev["name"], ev["workflow"], run.recommendations_created)
    return results
