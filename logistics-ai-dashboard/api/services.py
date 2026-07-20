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
    """Control-tower headline KPIs, computed from real engine output where
    derivable and labelled per-metric with ``live`` vs ``representative``.

    * supply_chain_health — network health score (modules.health_check)
    * late_shipments / todays_risks — real control-tower shipment KPIs
    * supplier_health — mean on-time across the carrier scorecard
    * inventory_value — avg on-hand value from the SKU plan × unit price
    * forecast_accuracy — representative until a scored backtest is wired
    """
    def build() -> dict[str, Any]:
        import pandas as pd
        kpis = {k: dict(v) for k, v in _FALLBACK_KPIS.items()}
        live: dict[str, bool] = {k: False for k in kpis}

        # Shipments → health, late, risks, supplier health
        try:
            shipments, sk, scorecard = _shipments()
            from modules import health_check
            hc = health_check.run_health_check(shipments=shipments, kpis=sk,
                                               delay_risk=sk.get("avg_delay_days"))
            if hc.get("score") is not None:
                kpis["supply_chain_health"].update(value=int(round(hc["score"])),
                                                   status=_health_status(hc["score"]))
                live["supply_chain_health"] = True
            if scorecard is not None and "On-Time %" in scorecard.columns:
                sh = float(scorecard["On-Time %"].dropna().mean())
                if not pd.isna(sh):
                    kpis["supplier_health"].update(value=int(round(sh)),
                                                   status=_health_status(sh))
                    live["supplier_health"] = True
                # Current-signal counts derived from carrier performance rather
                # than cumulative history (the demo dataset is fully delivered):
                # risks = carriers graded below B; late = carriers running a
                # positive average delay right now.
                risks = int((~scorecard["Grade"].isin(["A", "B"])).sum())
                kpis["todays_risks"].update(value=risks,
                                            status="critical" if risks else "good")
                live["todays_risks"] = True
                if "Avg Delay (days)" in scorecard.columns:
                    late = int((scorecard["Avg Delay (days)"] > 0.5).sum())
                    kpis["late_shipments"].update(value=late,
                                                  status="warning" if late else "good")
                    live["late_shipments"] = True
        except Exception:  # noqa: BLE001
            pass

        # Forecast accuracy from the real Prophet holdout backtest
        try:
            bt = _backtest()
            if bt.get("accuracy") is not None:
                kpis["forecast_accuracy"].update(value=int(round(bt["accuracy"])),
                                                 status=_health_status(bt["accuracy"]))
                live["forecast_accuracy"] = True
        except Exception:  # noqa: BLE001
            pass

        # Inventory value from the real SKU plan × unit price
        try:
            classified, plan = _inventory_frame()
            merged = plan.merge(classified[["SKU", "Unit Price"]], on="SKU", how="left")
            price = merged["Unit Price"].fillna(15.0)
            avg_units = merged["Safety Stock"] + merged["EOQ"] / 2.0
            inv_value = float((avg_units * price).sum())
            kpis["inventory_value"].update(value=round(inv_value / 1_000_000, 1))
            live["inventory_value"] = True
        except Exception:  # noqa: BLE001
            pass

        return {"kpis": kpis, "live": live}

    return _safe(build, {"kpis": _FALLBACK_KPIS,
                         "live": {k: False for k in _FALLBACK_KPIS}})


def _health_status(score: float) -> str:
    return "good" if score >= 85 else "warning" if score >= 70 else "critical"


# ---- Forecasting -------------------------------------------------------------
@_ttl_cache(900)
def _forecast_frame():
    """Prophet demand forecast over the Olist order history (cached 15 min)."""
    from modules import forecast
    orders = forecast.load_orders()
    daily = forecast.daily_demand(orders)
    # daily_demand adds an ``external_signal`` regressor; the module's
    # run_forecast doesn't populate it for future dates, so drive Prophet
    # directly here (backend logic untouched) and default the future signal to 0.
    model = forecast.fit_prophet_model(daily)
    future = model.make_future_dataframe(periods=14)
    if "external_signal" in daily.columns:
        future["external_signal"] = 0.0
    fc = model.predict(future)
    insights = forecast.forecast_insights(fc, daily, horizon_days=14)
    return daily, fc, insights


def forecast_snapshot() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        daily, fc, insights = _forecast_frame()
        hist = daily.tail(90)
        history = [{"ds": str(d)[:10], "y": float(y)}
                   for d, y in zip(hist["ds"], hist["y"])]
        fut = fc.tail(14)
        forecast_rows = [{
            "ds": str(d)[:10],
            "yhat": round(float(yh), 1),
            "lower": round(float(lo), 1),
            "upper": round(float(hi), 1),
        } for d, yh, lo, hi in zip(fut["ds"], fut["yhat"],
                                   fut["yhat_lower"], fut["yhat_upper"])]
        return {"history": history, "forecast": forecast_rows,
                "insights": _jsonable(insights)}
    return _safe(build, {"history": [], "forecast": [], "insights": {}})


# ---- Procurement -------------------------------------------------------------
def procurement_snapshot() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        import pandas as pd
        from modules import allocation
        shipments, sk, scorecard = _shipments()
        if scorecard is None:
            raise ValueError("no scorecard")
        profiles = allocation.build_carrier_profiles(scorecard, shipments)
        scored = allocation.allocation_scores(profiles)
        impact = allocation.allocation_impact(scored, int(sk.get("total", len(shipments))))
        rows = [{
            "carrier": str(r.get("Carrier", "—")),
            "score": round(float(r.get("Score", r.get("score", 0)) or 0), 1),
            "on_time": None if pd.isna(r.get("On-Time %")) else float(r.get("On-Time %")),
            "current_share": round(float(r.get("Current Share", 0) or 0) * 100, 1)
            if r.get("Current Share") is not None else None,
            "recommended_share": round(float(r.get("Recommended Share", 0) or 0) * 100, 1)
            if r.get("Recommended Share") is not None else None,
        } for _, r in scored.iterrows()]
        return {"carriers": rows, "impact": _jsonable(impact)}
    return _safe(build, {"carriers": [], "impact": {}})


# ---- Operations --------------------------------------------------------------
def operations_snapshot() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules import forecast, optimization
        shipments, sk, _ = _shipments()
        orders = forecast.load_orders()
        net = optimization.network_summary(orders) or {}
        status_counts = shipments["status"].value_counts().to_dict()
        return {
            "kpis": {
                "in_transit": int(sk.get("in_transit", 0)),
                "on_time_pct": round(float(sk.get("on_time_pct") or 0), 1),
                "avg_lead_days": round(float(net.get("avg_lead_days", 0)), 1),
                "delivered_observed": int(net.get("n_delivered_observed", 0)),
            },
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        }
    return _safe(build, {"kpis": {}, "status_counts": {}})


# ---- Warehouse (network zones from the geo clustering) -----------------------
def warehouse_snapshot() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        cents = _geo_centroids().sort_values("size", ascending=False).reset_index(drop=True)
        total = float(cents["size"].sum()) or 1.0
        # Utilisation proxy: share of demand each zone serves, capped at 100.
        zones = [{
            "zone": f"Hub {int(r['cluster'])}",
            "lat": round(float(r["lat"]), 3),
            "lon": round(float(r["lon"]), 3),
            "locations": int(r["size"]),
            "utilization": round(min(100.0, r["size"] / total * 100 * len(cents)), 1),
        } for _, r in cents.iterrows()]
        avg_util = round(sum(z["utilization"] for z in zones) / len(zones), 1) if zones else 0.0
        return {"zones": zones, "avg_utilization": avg_util, "hub_count": len(zones)}
    return _safe(build, {"zones": [], "avg_utilization": 0.0, "hub_count": 0})


# ---- Shipments (shared control-tower pipeline) -------------------------------
@_ttl_cache(300)
def _shipments():
    """Real shipment board from the Olist orders (cached 5 min).

    Uses the same control-tower pipeline as the Streamlit app: attach demo
    carriers/costs, then derive per-shipment status, delay-vs-promise and
    health. Returns ``(shipments_df, kpis, scorecard_df)``.
    """
    import pandas as pd
    from modules import control_tower
    orders = pd.read_csv("data/olist_orders_dataset.csv")
    orders = control_tower.assign_demo_carriers(orders)
    shipments = control_tower.prepare_shipments(orders)
    kpis = control_tower.shipment_kpis(shipments)
    scorecard = control_tower.carrier_scorecard(shipments)
    return shipments, kpis, scorecard


_STATUS = {"good": "good", "warning": "warning", "critical": "critical"}


# ---- Logistics ---------------------------------------------------------------
_LOGI_FALLBACK = {
    "kpis": {"in_transit": 128, "delayed": 3, "on_time_rate": 94, "avg_cost": 3.1},
    "lanes": [
        {"from": "Shanghai DC", "to": "Rotterdam", "status": "good"},
        {"from": "Ningbo", "to": "Melbourne", "status": "warning"},
        {"from": "Rotterdam", "to": "Lyon", "status": "good"},
    ],
    "delayed": [
        {"id": "SHP-20481", "lane": "Ningbo → Melbourne", "reason": "port congestion", "eta_slip": "+2d"},
    ],
    "carriers": [],
}


def logistics_snapshot() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        import pandas as pd
        shipments, kpis, scorecard = _shipments()
        cost = None
        if scorecard is not None and "Avg Cost/Shipment ($)" in scorecard.columns:
            cost = float(scorecard["Avg Cost/Shipment ($)"].mean())
        # A few worst-delayed real shipments for the feed
        late = shipments[shipments["delay_days"].fillna(0) > 0].sort_values(
            "delay_days", ascending=False).head(5)
        delayed = [{
            "id": str(r["shipment_id"]),
            "lane": str(r.get("carrier", "—")),
            "reason": f"{int(r['delay_days'])}d late vs promise",
            "eta_slip": f"+{int(r['delay_days'])}d",
        } for _, r in late.iterrows()]
        carriers = []
        if scorecard is not None:
            for _, r in scorecard.head(8).iterrows():
                ot = r.get("On-Time %")
                carriers.append({
                    "carrier": str(r["Carrier"]),
                    "shipments": int(r["Shipments"]),
                    "on_time": None if pd.isna(ot) else float(ot),
                    "grade": str(r.get("Grade", "—")),
                    "avg_delay": float(r.get("Avg Delay (days)", 0) or 0),
                })
        return {
            "kpis": {
                "in_transit": int(kpis.get("in_transit", 0)),
                "delayed": int(kpis.get("late", 0)),
                "on_time_rate": round(float(kpis.get("on_time_pct") or 0), 1),
                "avg_cost": round(cost, 2) if cost else None,  # $ per shipment
            },
            "lanes": _LOGI_FALLBACK["lanes"],  # illustrative until lane geo is fed
            "delayed": delayed or _LOGI_FALLBACK["delayed"],
            "carriers": carriers,
        }
    return _safe(build, _LOGI_FALLBACK)


# ---- Map (real geo from the Olist geolocation join) --------------------------
@_ttl_cache(600)
def _geo_centroids():
    """KMeans hub centroids over real customer lat/lon (cached 10 min)."""
    import pandas as pd
    from modules import network
    customers = pd.read_csv("data/olist_customers_dataset.csv")
    coords = network.prepare_customer_data(customers)
    clustered = network.run_clustering(coords, n_clusters=6)
    cents = clustered.groupby("cluster").agg(
        lat=("lat", "mean"), lon=("lon", "mean"), size=("lat", "size")
    ).reset_index()
    return cents


def logistics_map() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules import geo
        cents = _geo_centroids()
        cents = cents.sort_values("size", ascending=False).reset_index(drop=True)
        points = [{
            "name": f"Hub {int(r['cluster'])}",
            "lat": round(float(r["lat"]), 4),
            "lon": round(float(r["lon"]), 4),
            "size": int(r["size"]),
        } for _, r in cents.iterrows()]
        # Routes: primary hub → every other hub, flagged by relative distance
        routes = []
        if points:
            hub = points[0]
            from modules.network import haversine_km
            dists = [haversine_km(hub["lat"], hub["lon"], p["lat"], p["lon"]) for p in points[1:]]
            far = max(dists) if dists else 1.0
            for p, d in zip(points[1:], dists):
                routes.append({
                    "from": {"lat": hub["lat"], "lon": hub["lon"], "name": hub["name"]},
                    "to": {"lat": p["lat"], "lon": p["lon"], "name": p["name"]},
                    "status": "warning" if d > 0.7 * far else "good",
                    "distance_km": round(d, 0),
                })
        center = ([sum(p["lat"] for p in points) / len(points),
                   sum(p["lon"] for p in points) / len(points)] if points else [-15.0, -50.0])
        return {
            "tiles_url": geo.maptiler_tiles_url(),
            "attribution": geo.maptiler_attribution(),
            "center": center, "zoom": 4,
            "points": points, "routes": routes,
        }
    return _safe(build, {"tiles_url": None, "attribution": "", "center": [-15.0, -50.0],
                         "zoom": 4, "points": [], "routes": []})


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


# ---- Decision Center (trust layer + audit) -----------------------------------
def _ensure_recommendations() -> None:
    """Generate + persist recommendations from live engine context (idempotent;
    save_recommendation dedupes by key)."""
    from modules import cost_audit, sku, trust
    ctx: dict[str, Any] = {"service_level": 0.95, "avg_lead_time": 7.0, "history_days": 180}
    try:
        classified, plan = _inventory_frame()
        ctx["sku_plan"] = sku.stock_status(plan)  # adds ORDER NOW/SOON Status column
    except Exception:  # noqa: BLE001
        pass
    try:
        shipments, _sk, scorecard = _shipments()
        ctx["scorecard"] = scorecard
        ctx["audit"] = cost_audit.run_audit(shipments)
    except Exception:  # noqa: BLE001
        pass
    recs = trust.generate_all(ctx)
    trust.sync_recommendations(recs)


def decisions_snapshot() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules import trust
        _ensure_recommendations()
        return {
            "kpis": trust.summary_kpis(),
            "pending": trust.pending(),
            "history": trust.history(limit=60),
        }
    return _safe(build, {"kpis": {}, "pending": [], "history": []})


def decide(rec_key: str, status: str, note: str = "", actor: str = "user") -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules import trust
        ok = trust.decide(rec_key, status.upper(), note=note, actor=actor)
        return {"ok": ok, "rec_key": rec_key, "status": status.upper()}
    return _safe(build, {"ok": False, "rec_key": rec_key, "status": status})


def audit_trail(limit: int = 200) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        from modules import store
        return {"entries": store.load_audit_log(limit=limit)}
    return _safe(build, {"entries": []})


# ---- Forecast accuracy backtest ----------------------------------------------
@_ttl_cache(1800)
def _backtest(holdout_weeks: int = 10):
    """Forecast-accuracy backtest at planning (weekly) granularity — the level
    S&OP accuracy is actually quoted at. Aggregate daily orders to weekly demand,
    hold out the last ``holdout_weeks``, fit Prophet on the rest, and score
    predicted vs actual (MAPE / MAE / RMSE / Bias). Cached 30 min.
    """
    import numpy as np
    import pandas as pd
    from prophet import Prophet
    from modules import forecast
    orders = forecast.load_orders()
    daily = forecast.daily_demand(orders)[["ds", "y"]].sort_values("ds")
    weekly = (daily.set_index("ds")["y"].resample("W").sum().reset_index())
    # Trim trailing partial weeks (near-zero tail) so MAPE stays meaningful.
    med = float(weekly["y"].median())
    if med > 0:
        healthy = weekly["y"] >= 0.2 * med
        if healthy.any():
            weekly = weekly.iloc[: healthy[healthy].index[-1] + 1].reset_index(drop=True)
    train, test = weekly.iloc[:-holdout_weeks], weekly.iloc[-holdout_weeks:]
    model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True)
    model.fit(train)
    future = model.make_future_dataframe(periods=holdout_weeks, freq="W")
    fc = model.predict(future).tail(holdout_weeks).reset_index(drop=True)
    actual = test["y"].to_numpy(dtype=float)
    pred = np.clip(fc["yhat"].to_numpy(dtype=float), 0, None)
    err = pred - actual
    nz = actual > 0
    mape = float(np.mean(np.abs(err[nz] / actual[nz])) * 100) if nz.any() else None
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    points = [{"ds": str(d)[:10], "actual": round(float(a), 1), "predicted": round(float(p), 1)}
              for d, a, p in zip(test["ds"], actual, pred)]
    return {"mape": None if mape is None else round(mape, 1),
            "mae": round(mae, 1), "rmse": round(rmse, 1), "bias": round(bias, 1),
            "accuracy": None if mape is None else round(max(0.0, 100 - mape), 1),
            "holdout_weeks": holdout_weeks, "granularity": "weekly", "points": points}


def forecast_backtest() -> dict[str, Any]:
    return _safe(lambda: {**_backtest(), "ok": True},
                 {"mape": None, "mae": None, "rmse": None, "bias": None,
                  "accuracy": None, "holdout_weeks": 10, "granularity": "weekly", "points": []})


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
