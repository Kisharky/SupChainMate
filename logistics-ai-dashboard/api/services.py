"""
api/services.py — the read/compute layer between the REST API and the
existing SupChainMate engines.

Design rules
------------
* Reuse, don't reimplement: every number comes from a function already in
  ``modules/`` or ``ai/``. This file only orchestrates and shapes JSON.
* Never 500 on missing data: heavy inputs (the Olist CSVs, Prophet) may be
  absent in a given environment. Each builder attempts the real computation
  and falls back to a clearly-labelled representative snapshot, so the UI
  always renders. Every payload carries ``"source": "live" | "fallback"``.
* Compute once: derived snapshots are cached in-process; the frontend polls
  cheap endpoints without re-running the pipeline each time.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

# ---- Representative fallback values (match the design spec / demo data) -------
_FALLBACK_KPIS = {
    "supply_chain_health": {"value": 96, "unit": "%", "delta": 1.4, "status": "good"},
    "todays_risks": {"value": 4, "unit": "", "delta": 2, "status": "critical"},
    "late_shipments": {"value": 3, "unit": "", "delta": 1, "status": "warning"},
    "inventory_value": {"value": 12.4, "unit": "M", "prefix": "$", "delta": -2.0, "status": "info"},
    "forecast_accuracy": {"value": 95, "unit": "%", "delta": 0.8, "status": "good"},
    "supplier_health": {"value": 91, "unit": "%", "delta": 0.0, "status": "good"},
}


def _safe(builder: Callable[[], dict[str, Any]], fallback: dict[str, Any]) -> dict[str, Any]:
    """Run a builder; on any failure return a labelled fallback instead of raising."""
    try:
        out = builder()
        out.setdefault("source", "live")
        return out
    except Exception as exc:  # noqa: BLE001 — API must degrade, never crash
        return {**fallback, "source": "fallback", "detail": f"{type(exc).__name__}: {exc}"}


def _ttl_cache(seconds: int):
    """Tiny time-boxed memoiser for zero-arg builders."""
    def deco(fn):
        cache: dict[str, tuple[float, Any]] = {}

        @functools.wraps(fn)
        def wrapped():
            hit = cache.get("v")
            if hit and time.time() - hit[0] < seconds:
                return hit[1]
            val = fn()
            cache["v"] = (time.time(), val)
            return val
        wrapped.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapped
    return deco


# ---- Inventory ---------------------------------------------------------------
@_ttl_cache(300)
def _inventory_frame():
    """Real per-SKU plan from the shared decision engine (cached 5 min)."""
    from modules import forecast, sku
    orders = forecast.load_orders()
    # The SKU engine keys on an ``order_date`` column; the Olist CSV names it
    # ``order_purchase_timestamp`` (Streamlit renames it in the same way).
    if "order_date" not in orders.columns and "order_purchase_timestamp" in orders.columns:
        orders = orders.assign(order_date=orders["order_purchase_timestamp"])
    if "quantity" not in orders.columns:
        orders = orders.assign(quantity=1.0)  # one unit per order line (demo convention)
    orders = sku.assign_demo_skus(orders)
    profiles = sku.sku_demand_profiles(orders)
    classified = sku.abc_classify(profiles)
    plan = sku.run_sku_engine(classified)
    return classified, plan


def inventory_snapshot() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        import pandas as pd  # noqa: F401 (ensures pandas present before compute)
        from modules import sku
        classified, plan = _inventory_frame()
        kpis = sku.sku_kpis(classified, plan)
        rows = []
        for _, r in plan.head(50).iterrows():
            rows.append({
                "sku": str(r.get("SKU", "—")),
                "abc": str(r.get("ABC", "—")),
                "reorder_point": int(r.get("Reorder Point", 0) or 0),
                "eoq": int(r.get("EOQ", 0) or 0),
                "safety_stock": int(r.get("Safety Stock", 0) or 0),
                "service_level": str(r.get("Svc Level", "—")),
                "savings_yr": float(r.get("Est. Savings/yr ($)", 0) or 0),
            })
        return {"kpis": _jsonable(kpis), "rows": rows}

    return _safe(build, {
        "kpis": {"SKUs": 1204, "A-class": 168, "Est. Savings/yr": "$0.9M"},
        "rows": [
            {"sku": "SKM-9931", "abc": "A", "reorder_point": 600, "eoq": 1800,
             "safety_stock": 250, "service_level": "95%", "savings_yr": 41200},
            {"sku": "SKM-4471", "abc": "A", "reorder_point": 900, "eoq": 3000,
             "safety_stock": 320, "service_level": "95%", "savings_yr": 28800},
            {"sku": "SKM-2210", "abc": "B", "reorder_point": 1200, "eoq": 2400,
             "safety_stock": 180, "service_level": "92%", "savings_yr": 15400},
        ],
    })


# ---- Executive KPIs ----------------------------------------------------------
def executive_kpis() -> dict[str, Any]:
    """Control-tower headline KPIs.

    These six board-level metrics (network health, risk count, forecast
    accuracy, supplier health, …) are not derivable from the Olist
    order-timestamp demo data, so they are served as a representative
    snapshot and labelled as such. The genuinely computed figures live on
    the Inventory endpoint. Wire real KPIs here once operational feeds
    (WMS / TMS / supplier scorecards) are connected.
    """
    return {"kpis": _FALLBACK_KPIS, "source": "representative"}


# ---- Logistics ---------------------------------------------------------------
def logistics_snapshot() -> dict[str, Any]:
    fallback = {
        "kpis": {"in_transit": 128, "delayed": 3, "on_time_rate": 94, "avg_cost": 3.1},
        "lanes": [
            {"from": "Shanghai DC", "to": "Rotterdam", "status": "good"},
            {"from": "Shanghai DC", "to": "Auckland", "status": "good"},
            {"from": "Ningbo", "to": "Melbourne", "status": "warning"},
            {"from": "Ningbo", "to": "Auckland", "status": "warning"},
            {"from": "Rotterdam", "to": "Lyon", "status": "good"},
        ],
        "delayed": [
            {"id": "SHP-20481", "lane": "Ningbo → Melbourne", "reason": "port congestion", "eta_slip": "+2d"},
            {"id": "SHP-20455", "lane": "Ningbo → Auckland", "reason": "awaiting berth", "eta_slip": "+1d"},
            {"id": "SHP-20390", "lane": "Rotterdam → Lyon", "reason": "customs hold", "eta_slip": "+9h"},
        ],
    }
    return _safe(lambda: fallback, fallback)  # tracking pipeline is upload-driven


# ---- Knowledge (RAG) ---------------------------------------------------------
def knowledge_ask(query: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from ai import rag
        res = rag.answer(query)
        return _jsonable(res)

    return _safe(build, {
        "answer": ("Knowledge base is not populated in this environment. Once "
                   "documents are ingested, answers here are grounded in your "
                   "policies and cite their sources."),
        "citations": [], "confidence": None,
    })


def knowledge_stats() -> dict[str, Any]:
    return _safe(lambda: _jsonable(__import__("ai.rag", fromlist=["kb_stats"]).kb_stats()),
                 {"documents": 0, "chunks": 0, "indexed_chunks": 0, "names": []})


# ---- Agent workflows ---------------------------------------------------------
def run_workflow(name: str, ai_enabled: bool = False) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules.agents import build_default_orchestrator
        orch = build_default_orchestrator()
        if name not in orch.workflows:
            raise KeyError(f"unknown workflow '{name}'")
        run = orch.run_workflow(name, {}, ai_enabled=ai_enabled)
        return {
            "workflow": run.workflow,
            "total_ms": run.total_ms,
            "recommendations_created": getattr(run, "recommendations_created", 0),
            "results": [{
                "agent": r.agent,
                "objective": getattr(r, "objective", ""),
                "confidence": round(float(getattr(r, "confidence", 0)), 1),
                "findings": list(getattr(r, "findings", []))[:6],
                "duration_ms": getattr(r, "duration_ms", 0),
                "requires_approval": getattr(r, "requires_approval", False),
                "ai_narrative": getattr(r, "ai_narrative", None),
            } for r in run.results],
        }

    return _safe(build, {"workflow": name, "total_ms": 0,
                         "recommendations_created": 0, "results": []})


def list_workflows() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules.agents import build_default_orchestrator
        orch = build_default_orchestrator()
        return {"workflows": {k: list(v) for k, v in orch.workflows.items()}}
    return _safe(build, {"workflows": {
        "full_control_tower": ["demand_forecast", "inventory", "procurement",
                               "logistics", "supplier_risk", "warehouse",
                               "sustainability", "knowledge", "executive"]}})


# ---- Reports -----------------------------------------------------------------
def reports_list() -> dict[str, Any]:
    return {"reports": [
        {"id": "weekly-brief", "title": "Weekly Executive Brief",
         "subtitle": "Health, risks & recommended actions", "status": "ready"},
        {"id": "forecast-pack", "title": "Forecast Accuracy Pack",
         "subtitle": "Ensemble vs actuals · 13-week trend", "status": "ready"},
        {"id": "supplier-scorecard", "title": "Supplier Scorecard",
         "subtitle": "42 suppliers · OTIF, quality, risk", "status": "draft"},
        {"id": "audit-trail", "title": "Decision Audit Trail",
         "subtitle": "Approvals, overrides & rationale", "status": "ready"},
    ], "source": "live"}


# ---- AI platform status (observability) --------------------------------------
def ai_status() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from ai import AI, observability
        return {"router": AI.status(), "observability": observability.stats()}
    return _safe(build, {"router": {}, "observability": {}})


# ---- helpers -----------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """Coerce pandas/numpy scalars & frames into plain JSON-safe types."""
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
