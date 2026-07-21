"""
api/fraud.py — Fraud & Anomaly Detection.

An enterprise trust surface inspired by the freight-fraud problem space
(double-brokering, duplicate invoices, payment anomalies, carrier/supplier
identity risk). It rides SupChainMate's existing audit + Commercial layers —
detected anomalies escalate into the Decision Center for a human call.

Signals here are representative and labelled (deterministic, so the demo is
stable). The detection *methods* are real patterns a production deployment would
run over live invoice / payment / document feeds; this module is the clean seam
where those detectors plug in. No AI engine or business logic is modified.
"""

from __future__ import annotations

import hashlib
from typing import Any

from api.services import _safe

# Anomaly archetypes → how they present on the alert feed.
_TYPES = {
    "duplicate_invoice": {"label": "Duplicate invoice", "icon": "⎘"},
    "double_brokering": {"label": "Double-brokering", "icon": "⇄"},
    "price_anomaly": {"label": "Price anomaly", "icon": "◮"},
    "payment_pattern": {"label": "Payment pattern", "icon": "⏣"},
    "identity_risk": {"label": "Identity risk", "icon": "⚑"},
    "discount_anomaly": {"label": "Discount anomaly", "icon": "◔"},
}

# Representative alerts (labelled). Each is deterministic via its id.
_ALERTS = [
    ("duplicate_invoice", "high", "Northwind Freight",
     "Invoice INV-8842 duplicates INV-8817 — identical amount, carrier, 2 days apart.",
     "Hold payment and merge to a single payable", 18400),
    ("double_brokering", "high", "Apex Carriers LLC",
     "Carrier name differs across BOL, Rate Confirmation, and POD — classic re-brokering signal.",
     "Freeze load, verify MC authority before release", 26200),
    ("identity_risk", "high", "SwiftHaul (MC-1180294)",
     "Carrier MC# registered 11 days ago; contact domain age 9 days; no insurance history.",
     "Block onboarding pending manual verification", 0),
    ("price_anomaly", "medium", "Lane SP→RJ",
     "Freight rate 3.1σ above the 90-day lane average with no fuel-surcharge basis.",
     "Re-quote lane and audit the rate con", 4300),
    ("payment_pattern", "medium", "Vendor #4471",
     "Payment velocity spike — 4× normal weekly volume to a payee added this month.",
     "Route to AP review before the next run", 31200),
    ("discount_anomaly", "medium", "GlobalMart (key account)",
     "Unapproved 22% discount applied below the contracted floor.",
     "Recover margin; require approval on next order", 9800),
    ("duplicate_invoice", "low", "Cedar Logistics",
     "Near-duplicate invoice: same PO, amount within $2 — likely a re-submission.",
     "Confirm with carrier before paying", 2100),
    ("identity_risk", "low", "Pioneer Transit",
     "Bank account on file changed 3 days before invoicing — verify remittance details.",
     "Confirm banking change out-of-band", 0),
]

# Detection methods that are "running" — the coverage strip.
_CHECKS = [
    {"name": "Document consistency (BOL · POD · Rate Con)", "coverage": 100, "status": "good"},
    {"name": "Duplicate & near-duplicate invoice match", "coverage": 100, "status": "good"},
    {"name": "Payment-pattern anomaly scoring", "coverage": 94, "status": "good"},
    {"name": "Carrier / supplier identity verification", "coverage": 88, "status": "warning"},
    {"name": "Price / rate outlier detection (per lane)", "coverage": 100, "status": "good"},
]

# Representative entity risk register.
_ENTITIES = [
    ("Apex Carriers LLC", "carrier", 88, "Document inconsistency across load docs"),
    ("SwiftHaul", "carrier", 82, "New MC authority; thin insurance history"),
    ("Vendor #4471", "supplier", 71, "Abnormal payment velocity"),
    ("Northwind Freight", "carrier", 63, "Repeated duplicate submissions"),
    ("Pioneer Transit", "carrier", 44, "Recent banking-detail change"),
    ("Cedar Logistics", "carrier", 31, "Occasional re-submissions"),
    ("Blue Ridge Haulage", "carrier", 18, "Clean — monitored"),
    ("Continental Supply Co", "supplier", 12, "Clean — monitored"),
]

_SEVERITY_STATUS = {"high": "critical", "medium": "warning", "low": "info"}


def _seed(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def _tier(score: int) -> str:
    return "Critical" if score >= 80 else "High" if score >= 60 else "Medium" if score >= 35 else "Low"


def _alerts() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (kind, sev, entity, detail, action, at_risk) in enumerate(_ALERTS):
        h = _seed(f"{kind}{entity}{i}")
        status = ("open", "investigating", "open", "open")[h % 4]
        out.append({
            "id": f"ALT-{2100 + i}",
            "type": kind,
            "type_label": _TYPES[kind]["label"],
            "icon": _TYPES[kind]["icon"],
            "severity": sev,
            "severity_status": _SEVERITY_STATUS[sev],
            "entity": entity,
            "detail": detail,
            "recommended_action": action,
            "amount_at_risk": at_risk,
            "confidence": min(98, 66 + h % 32),
            "status": status,
            "hours_ago": 1 + (h % 70),
        })
    out.sort(key=lambda a: ({"high": 0, "medium": 1, "low": 2}[a["severity"]], a["hours_ago"]))
    return out


def _entities() -> list[dict[str, Any]]:
    return [
        {"name": n, "kind": k, "risk_score": s, "tier": _tier(s),
         "tier_status": _SEVERITY_STATUS["high" if s >= 60 else "medium" if s >= 35 else "low"],
         "top_factor": f}
        for n, k, s, f in _ENTITIES
    ]


def _summary(alerts: list[dict[str, Any]], entities: list[dict[str, Any]]) -> dict[str, Any]:
    open_alerts = [a for a in alerts if a["status"] != "resolved"]
    return {
        "open_alerts": len(open_alerts),
        "high_severity": sum(1 for a in open_alerts if a["severity"] == "high"),
        "amount_at_risk": sum(a["amount_at_risk"] for a in open_alerts),
        "entities_flagged": sum(1 for e in entities if e["risk_score"] >= 60),
        "duplicate_invoices": sum(1 for a in alerts if a["type"] == "duplicate_invoice"),
        "detection_accuracy": 96,   # representative model precision (%)
    }


def overview() -> dict[str, Any]:
    """Full fraud console payload: summary, detection coverage, alert feed, and
    the entity risk register. Representative + labelled."""
    def build() -> dict[str, Any]:
        alerts = _alerts()
        entities = _entities()
        return {
            "summary": _summary(alerts, entities),
            "checks": _CHECKS,
            "alerts": alerts,
            "entities": entities,
            "source": "representative",
        }
    return _safe(build, {"summary": {}, "checks": _CHECKS, "alerts": [], "entities": [], "source": "fallback"})
