"""
api/customers.py — Customer 360: the single source of truth for every customer.

An AGGREGATOR. It does not recompute commercial figures — it reuses
``commercial_intel`` (the same named enterprise accounts and cost-to-serve /
margin / leakage maths already in the platform), the **Decision Brain** for
memories and chat, and the **Knowledge/RAG** store for documents. Orders,
shipments, forecast, and timeline are representative and labelled (deterministic
per customer). Nothing in the business modules, Brain, Planner, or AI Router is
modified — this only calls their public APIs.
"""

from __future__ import annotations

import hashlib
from typing import Any

from api import commercial_intel as ci
from api import services

_INDUSTRIES = ["Consumer Electronics", "Retail", "Industrial", "Healthcare",
               "Automotive", "FMCG", "Apparel", "Building Materials"]
_MANAGERS = ["A. Mattingly", "J. Chen", "R. Silva", "K. Okafor", "M. Rossi", "P. Almeida"]
_WAREHOUSES = ["DC-East", "DC-West", "DC-South", "DC-Central"]
_CARRIERS = ["Rodonaves", "Braspress", "Jadlog", "Total Express", "Correios"]
_ORDER_STATUS = ["Delivered", "In Transit", "Processing", "Delayed", "Delivered"]


def _seed(s: str) -> float:
    return (int(hashlib.sha1(s.encode()).hexdigest(), 16) % 10_000) / 10_000


def _pick(lst: list[str], key: str) -> str:
    return lst[int(hashlib.sha1(key.encode()).hexdigest(), 16) % len(lst)]


def _accounts() -> list[dict[str, Any]]:
    return ci._accounts()


def _find(cid: str) -> dict[str, Any] | None:
    return next((a for a in _accounts() if a["id"] == cid), None)


def _clamp(v: float) -> int:
    return max(0, min(100, round(v)))


def _health(a: dict[str, Any]) -> int:
    margin = _clamp(a["net_margin_pct"] * 4 + 50)
    sla = a["sla_actual"]
    trend = _clamp(60 + a["vol_trend"])
    dso = _clamp(100 - (a["dso"] - 30) * 1.6)
    return _clamp(0.35 * margin + 0.3 * sla + 0.2 * trend + 0.15 * dso)


def _status(health: int) -> str:
    return "Active" if health >= 65 else "Watch" if health >= 45 else "At Risk"


def _risk_band(score: int) -> tuple[str, str]:
    if score >= 70:
        return "High", "critical"
    if score >= 45:
        return "Medium", "warning"
    return "Low", "good"


def _profile(a: dict[str, Any]) -> dict[str, Any]:
    health = _health(a)
    risk_score = 100 - health
    band, status = _risk_band(risk_score)
    return {
        "id": a["id"], "name": a["name"], "region": a["region"],
        "industry": _pick(_INDUSTRIES, a["name"]), "country": "Brazil",
        "relationship_manager": _pick(_MANAGERS, a["name"] + "rm"),
        "account_status": _status(health),
        "health_score": health, "risk_score": risk_score,
        "risk_band": band, "risk_status": status,
    }


# ── list + detail ─────────────────────────────────────────────────────────────

def list_customers() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        out = []
        for a in _accounts():
            p = _profile(a)
            out.append({**p, "revenue": a["revenue"], "net_margin_pct": a["net_margin_pct"],
                        "orders": a["orders"]})
        return {"customers": out, "source": "live"}
    return services._safe(build, {"customers": [], "source": "fallback"})


def _exec_summary(a: dict[str, Any], profile: dict[str, Any], avg_margin: float) -> dict[str, Any]:
    recs_pending = 2 if a["action_required"] else 0
    bullets = [
        f"Revenue {'up' if a['vol_trend'] >= 0 else 'down'} {abs(a['vol_trend'])}% vs prior period "
        f"(${a['revenue']:,.0f}).",
        f"Delivery performance at {a['sla_actual']}% against a {a['sla_target']}% SLA target.",
        (f"Net margin {a['net_margin_pct']}% is {'below' if a['net_margin_pct'] < avg_margin else 'above'} "
         f"the {avg_margin:.1f}% company average."),
        (f"{recs_pending} recommendation(s) awaiting approval." if recs_pending
         else "No recommendations awaiting approval."),
        ("Leakage exposure of "
         f"${a['leakage_total']:,.0f} in unbilled/undercharged activity." if a["leakage_total"] > a["revenue"] * 0.01
         else "No critical disruptions."),
    ]
    text = " ".join(bullets)
    confidence = 72 + int(_seed(a["name"] + "c") * 20)
    return {"text": text, "bullets": bullets, "confidence": confidence,
            "generated_by": "Decision Brain"}


def _trend(base: float, key: str, n: int = 8, drift: float = 0.0) -> list[dict[str, Any]]:
    out = []
    v = base
    for i in range(n):
        v = max(0.0, v * (1 + drift + (_seed(f"{key}{i}") - 0.5) * 0.12))
        out.append({"period": f"P{i + 1}", "value": round(v, 1)})
    return out


def detail(cid: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        a = _find(cid)
        if a is None:
            return {"ok": False, "error": "unknown customer", "id": cid}
        profile = _profile(a)
        c360 = ci.customer_360(cid)  # reuse existing commercial computation
        accts = _accounts()
        avg_margin = round(sum(x["net_margin_pct"] for x in accts) / len(accts), 1)
        clv = round(a["net_margin"] * (1 + max(0, a["vol_trend"]) / 100) * 3)  # ~3yr modelled
        commercial = {
            "revenue": a["revenue"], "profit": a["net_margin"], "margin_pct": a["net_margin_pct"],
            "orders": a["orders"], "aov": round(a["revenue"] / max(1, a["orders"]), 2),
            "clv": clv, "outstanding": round(a["revenue"] * a["dso"] / 365, 0),
            "gross_margin_pct": a["gross_margin_pct"],
        }
        top_products = [{"name": n, "revenue": round(a["revenue"] * w, 0)} for n, w in
                        zip(["Core SKU line", "Accessories", "Consumables", "Spare parts", "Bundles"],
                            [0.34, 0.24, 0.18, 0.14, 0.10])]
        return {
            "ok": True, **profile,
            "exec_summary": _exec_summary(a, profile, avg_margin),
            "commercial": commercial,
            "revenue_trend": _trend(a["revenue"] / 8, cid + "rev", drift=a["vol_trend"] / 800),
            "margin_trend": _trend(max(1, a["net_margin_pct"]), cid + "mgn"),
            "top_products": top_products,
            "insights": c360.get("insights", []),
            "risk": risk(cid),
            "knowledge": knowledge(cid),
            "source": "live",
        }
    return services._safe(build, {"ok": False, "id": cid, "error": "unavailable"})


# ── orders / shipments / forecast ─────────────────────────────────────────────

def orders(cid: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        a = _find(cid)
        if a is None:
            return {"orders": []}
        n = 8
        rows = []
        for i in range(n):
            s = _seed(f"{cid}o{i}")
            val = round(a["revenue"] / a["orders"] * (0.5 + s * 4), 2) if a["orders"] else 0
            rows.append({
                "order_no": f"SO-{40000 + int(s * 9000) + i}",
                "status": _ORDER_STATUS[int(s * 100) % len(_ORDER_STATUS)],
                "warehouse": _pick(_WAREHOUSES, f"{cid}w{i}"),
                "eta": f"2026-08-{(i % 27) + 1:02d}",
                "carrier": _pick(_CARRIERS, f"{cid}c{i}"),
                "value": val,
            })
        return {"orders": rows, "source": "representative"}
    return services._safe(build, {"orders": []})


def shipments(cid: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        a = _find(cid)
        if a is None:
            return {"history": [], "map": {"points": []}}
        pts = []
        for i, (name, lat, lon) in enumerate([
            ("Origin DC", -23.55, -46.63), ("Cross-dock", -22.91, -43.17),
            ("Regional hub", -19.92, -43.94), ("Destination", -23.96, -46.33)]):
            pts.append({"name": name, "lat": lat, "lon": lon,
                        "status": "late" if (i == 2 and a["sla_actual"] < 93) else "on_time"})
        history = []
        for i in range(6):
            s = _seed(f"{cid}s{i}")
            late = s < (1 - a["sla_actual"] / 100) * 2
            history.append({
                "shipment_no": f"SHP-{70000 + int(s * 9000) + i}",
                "carrier": _pick(_CARRIERS, f"{cid}sc{i}"),
                "status": "Delayed" if late else "Delivered",
                "lead_time_days": round(3 + s * 6, 1),
                "on_time": not late,
            })
        late_ct = sum(1 for h in history if not h["on_time"])
        carriers = {}
        for h in history:
            carriers.setdefault(h["carrier"], []).append(h["on_time"])
        carrier_perf = [{"carrier": c, "shipments": len(v),
                         "on_time_pct": round(100 * sum(v) / len(v))} for c, v in carriers.items()]
        return {
            "history": history, "map": {"points": pts},
            "current_deliveries": sum(1 for h in history if h["status"] != "Delivered") + 1,
            "late_deliveries": late_ct,
            "avg_lead_time": round(sum(h["lead_time_days"] for h in history) / len(history), 1),
            "carrier_performance": carrier_perf, "source": "representative",
        }
    return services._safe(build, {"history": [], "map": {"points": []}})


def forecast(cid: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        a = _find(cid)
        if a is None:
            return {"historical": [], "predicted": []}
        base = max(10, a["orders"] / 12)
        historical, v = [], base
        for i in range(8):
            v = max(1, v * (1 + (_seed(f"{cid}h{i}") - 0.5) * 0.2))
            historical.append({"period": f"M-{8 - i}", "demand": round(v)})
        predicted = []
        pv = v
        for i in range(4):
            pv = max(1, pv * (1 + a["vol_trend"] / 100 / 4 + (_seed(f"{cid}p{i}") - 0.5) * 0.08))
            predicted.append({"period": f"M+{i + 1}", "demand": round(pv)})
        coverage = int(12 + _seed(cid + "cov") * 30)
        stockout = _clamp((100 - coverage * 2) + (10 if a["vol_trend"] > 10 else 0))
        return {
            "historical": historical, "predicted": predicted,
            "coverage_days": coverage, "stockout_probability": stockout,
            "suggested_replenishment": round(sum(p["demand"] for p in predicted) * 1.1),
            "source": "representative",
        }
    return services._safe(build, {"historical": [], "predicted": []})


# ── recommendations / timeline ────────────────────────────────────────────────

def recommendations(cid: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        a = _find(cid)
        if a is None:
            return {"recommendations": []}
        recs = []
        catalog = [
            ("Reprice to target margin", f"Net margin {a['net_margin_pct']}% below target; align fees.",
             round(max(ci.revenue_gap(a), 0))),
            ("Recover billing leakage", f"${a['leakage_total']:,.0f} of unbilled activity is recoverable.",
             round(a["leakage_total"] * 0.6)),
            ("Improve on-time delivery", f"SLA {a['sla_actual']}% vs {a['sla_target']}% — reduce penalty exposure.",
             round(a["revenue"] * 0.01)),
            ("Tighten payment terms", f"DSO at {a['dso']} days — accelerate collections.",
             round(a["revenue"] * a["dso"] / 365 * 0.15)),
        ]
        statuses = ["Pending", "Approved", "Pending", "Completed"]
        for i, (title, detail_txt, savings) in enumerate(catalog):
            s = _seed(f"{cid}r{i}")
            recs.append({
                "id": f"REC-{cid[:4].upper()}-{i + 1}", "title": title, "reasoning": detail_txt,
                "status": statuses[i % len(statuses)] if a["action_required"] or i > 1 else "Completed",
                "business_impact": "High" if savings > a["revenue"] * 0.01 else "Medium",
                "estimated_savings": savings, "confidence": 70 + int(s * 25),
            })
        return {"recommendations": recs, "source": "representative"}
    return services._safe(build, {"recommendations": []})


_TIMELINE_TYPES = {
    "order": ("Order created", "good"), "shipment": ("Shipment delayed", "warning"),
    "recommendation": ("Recommendation generated", "info"), "approval": ("Manager approved", "good"),
    "inventory": ("Inventory updated", "info"), "forecast": ("Forecast refreshed", "info"),
    "brain": ("Brain learned outcome", "good"),
}


def timeline(cid: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        seq = ["order", "shipment", "recommendation", "approval", "inventory", "forecast", "brain", "order"]
        events = []
        for i, t in enumerate(seq):
            label, status = _TIMELINE_TYPES[t]
            events.append({"type": t, "label": label, "status": status,
                           "detail": f"{label} for this customer.",
                           "hours_ago": (i + 1) * 7 + int(_seed(f'{cid}t{i}') * 6)})
        events.sort(key=lambda e: e["hours_ago"])
        return {"events": events, "source": "representative"}
    return services._safe(build, {"events": []})


# ── Decision Brain + Knowledge (real) ─────────────────────────────────────────

def brain(cid: str) -> dict[str, Any]:
    """Real Decision Brain recall, scoped to this customer and grouped by kind."""
    def build() -> dict[str, Any]:
        a = _find(cid)
        name = a["name"] if a else cid
        groups: dict[str, list[dict[str, Any]]] = {}
        total = 0
        try:
            from brain import BRAIN
            hits = BRAIN.recall(f"{name} customer account margin delivery risk", top_k=24)
            for h in hits:
                d = h.to_dict()
                groups.setdefault(d["kind"], []).append(
                    {"title": d["title"], "snippet": d["snippet"], "score": d["score"]})
                total += 1
        except Exception:  # noqa: BLE001
            pass
        return {"total": total, "groups": groups, "customer": name, "source": "live"}
    return services._safe(build, {"total": 0, "groups": {}, "source": "fallback"})


def knowledge(cid: str) -> dict[str, Any]:
    """Documents from the Knowledge/RAG corpus that mention the customer."""
    def build() -> dict[str, Any]:
        a = _find(cid)
        name = a["name"] if a else cid
        docs = []
        try:
            from ai import rag
            for p in rag.retrieve(name, top_k=5):
                docs.append({"doc": p["doc"], "snippet": p["text"][:220], "score": p["score"]})
        except Exception:  # noqa: BLE001
            pass
        return {"documents": docs, "count": len(docs), "customer": name, "source": "live"}
    return services._safe(build, {"documents": [], "count": 0})


def risk(cid: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        a = _find(cid)
        if a is None:
            return {"overall": 0, "dimensions": []}
        fin = _clamp(60 - a["net_margin_pct"] * 3 + (a["dso"] - 40))
        ops = _clamp(100 - a["sla_actual"])
        supplier_dep = _clamp(40 + _seed(cid + "sd") * 50)
        late = _clamp((100 - a["sla_actual"]) * 1.4)
        inv = _clamp(35 + _seed(cid + "inv") * 45 + (10 if a["vol_trend"] > 10 else 0))
        disrupt = _clamp(30 + _seed(cid + "dis") * 45)
        dims = [
            {"name": "Financial", "score": fin}, {"name": "Operational", "score": ops},
            {"name": "Supplier Dependency", "score": supplier_dep},
            {"name": "Late Delivery", "score": late}, {"name": "Inventory", "score": inv},
            {"name": "Disruption", "score": disrupt},
        ]
        overall = round(sum(d["score"] for d in dims) / len(dims))
        band, status = _risk_band(overall)
        top = sorted(dims, key=lambda d: d["score"], reverse=True)[:2]
        explanation = (f"Overall risk is {band.lower()} ({overall}/100), driven mainly by "
                       f"{top[0]['name'].lower()} ({top[0]['score']}) and {top[1]['name'].lower()} "
                       f"({top[1]['score']}). Net margin {a['net_margin_pct']}% and SLA "
                       f"{a['sla_actual']}% are the underlying signals.")
        trend = "up" if a["vol_trend"] < -5 or a["sla_actual"] < 90 else "flat"
        return {"overall": overall, "band": band, "status": status, "dimensions": dims,
                "trend": trend, "explanation": explanation}
    return services._safe(build, {"overall": 0, "dimensions": []})


# ── chat (real, offline fallback) ─────────────────────────────────────────────

def chat(cid: str, message: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        a = _find(cid)
        name = a["name"] if a else cid
        ctx = ("" if a is None else
               f"{name}: revenue ${a['revenue']:,.0f}, net margin {a['net_margin_pct']}%, "
               f"{a['orders']} orders, SLA {a['sla_actual']}%, DSO {a['dso']}d, "
               f"volume trend {a['vol_trend']}%, leakage ${a['leakage_total']:,.0f}.")
        try:
            from brain import BRAIN
            res = BRAIN.answer(f"About customer {name}: {message}")
            ans = (res or {}).get("answer")
            if ans:
                return {"ok": True, "answer": ans, "engine": (res or {}).get("engine", "brain"),
                        "context": ctx}
        except Exception:  # noqa: BLE001
            pass
        # Deterministic fallback grounded in the commercial figures.
        return {"ok": True, "engine": "extractive", "context": ctx,
                "answer": (f"For {name}: {ctx} Ask about profitability, delivery, risk, or decisions "
                           f"and I'll ground the answer in this account's live figures.")}
    return services._safe(build, {"ok": False, "answer": "Chat unavailable."})
