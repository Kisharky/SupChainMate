"""
api/freight.py — Freight Operations (brokerage back office).

The freight-broker workflows that a "digital worker" runs end-to-end: carrier
vetting & onboarding (authority + insurance + fraud signals), load ↔ carrier
matching, instant spot quoting, inbound email triage, and OS&D claims. This is
the vertical, execution-oriented layer that sits alongside the shipper-side
decision intelligence.

All signals are representative and labelled (deterministic, so demos are
stable). Authority/insurance checks, matching scores, and triage classifications
are the seams where real FMCSA / RMIS / TMS / email connectors plug in — the
API surface and UI stay identical. No AI engine or business logic is modified;
infra-heavy items (voice check-calls, appointment scheduling, factoring, EDI)
are surfaced as roadmap, not faked.
"""

from __future__ import annotations

import hashlib
from typing import Any

from api.services import _safe

_SEVERITY_STATUS = {"high": "critical", "medium": "warning", "low": "info"}

# ---- Carrier vetting & onboarding --------------------------------------------
# (name, mc, dot, authority, authority_age_days, insurance, days_to_expiry, stage, [flags])
_CARRIERS = [
    ("Apex Carriers LLC", "MC-885421", "DOT-2841190", "active", 512, "valid", 210, "monitoring",
     ["vin_overlap", "name_change"]),
    ("SwiftHaul", "MC-1180294", "DOT-3902551", "active", 11, "valid", 350, "verification",
     ["new_authority", "shared_phone"]),
    ("Blue Ridge Haulage", "MC-712004", "DOT-1993220", "active", 1480, "valid", 96, "approved", []),
    ("Cedar Logistics", "MC-640883", "DOT-1770559", "active", 905, "expiring", 12, "monitoring",
     ["insurance_expiring"]),
    ("Northwind Freight", "MC-559120", "DOT-1550430", "active", 380, "valid", 260, "approved",
     ["duplicate_docs"]),
    ("Pioneer Transit", "MC-903471", "DOT-3110882", "active", 240, "valid", 45, "verification",
     ["address_mismatch", "banking_change"]),
    ("Continental Expedited", "MC-410092", "DOT-1220771", "active", 2100, "valid", 300, "approved", []),
    ("GreyLine Transport", "MC-1204553", "DOT-3990221", "pending", 6, "lapsed", -3, "docs",
     ["new_authority", "insurance_lapsed"]),
]

_STAGES = ["docs", "verification", "approved", "monitoring", "rejected"]

# Human-readable fraud/risk flags.
_FLAG_LABELS = {
    "vin_overlap": "Truck VIN shared with another carrier",
    "name_change": "Legal name changed in last 90 days",
    "new_authority": "Operating authority < 30 days old",
    "shared_phone": "Contact phone shared with another MC#",
    "insurance_expiring": "Insurance expires within 30 days",
    "insurance_lapsed": "Insurance certificate lapsed",
    "duplicate_docs": "Onboarding docs match an existing carrier",
    "address_mismatch": "Physical address ≠ FMCSA registration",
    "banking_change": "Remittance bank changed pre-invoicing",
}


def _seed(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def _carrier_risk(authority: str, ins: str, age: int, flags: list[str]) -> int:
    score = 0
    if authority != "active":
        score += 25
    if ins == "lapsed":
        score += 30
    elif ins == "expiring":
        score += 12
    if age < 30:
        score += 22
    score += 12 * len(flags)
    return min(98, score)


def _carriers() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (name, mc, dot, auth, age, ins, exp, stage, flags) in enumerate(_CARRIERS):
        risk = _carrier_risk(auth, ins, age, flags)
        sev = "high" if risk >= 60 else "medium" if risk >= 30 else "low"
        out.append({
            "id": mc,
            "name": name, "mc_number": mc, "dot_number": dot,
            "authority_status": auth, "authority_age_days": age,
            "insurance_status": ins, "insurance_days_to_expiry": exp,
            "stage": stage, "flags": flags, "flag_count": len(flags),
            "risk_score": risk, "risk_severity": sev, "risk_status": _SEVERITY_STATUS[sev],
            "recommendation": ("Block — manual verification required" if risk >= 60
                               else "Review flagged items" if risk >= 30
                               else "Clear to onboard"),
        })
    out.sort(key=lambda c: c["risk_score"], reverse=True)
    return out


def carrier_detail(carrier_id: str) -> dict[str, Any]:
    """Full vetting checklist for one carrier (representative)."""
    def build() -> dict[str, Any]:
        c = next((x for x in _carriers() if x["id"] == carrier_id), None)
        if c is None:
            return {"ok": False, "error": "unknown carrier", "carrier_id": carrier_id}
        checks = [
            {"name": "Operating authority active (FMCSA)", "ok": c["authority_status"] == "active",
             "detail": f"{c['authority_status'].title()} · {c['authority_age_days']} days old"},
            {"name": "Authority age ≥ 30 days", "ok": c["authority_age_days"] >= 30,
             "detail": f"{c['authority_age_days']} days"},
            {"name": "Insurance certificate valid", "ok": c["insurance_status"] == "valid",
             "detail": (f"{c['insurance_status'].title()} · "
                        f"{c['insurance_days_to_expiry']} days to expiry")},
            {"name": "No identity / fraud signals", "ok": c["flag_count"] == 0,
             "detail": f"{c['flag_count']} signal(s)"},
        ]
        return {
            "ok": True, "carrier_id": carrier_id, "name": c["name"],
            "mc_number": c["mc_number"], "dot_number": c["dot_number"],
            "stage": c["stage"], "risk_score": c["risk_score"], "risk_status": c["risk_status"],
            "recommendation": c["recommendation"],
            "checks": checks,
            "flags": [{"code": f, "label": _FLAG_LABELS.get(f, f)} for f in c["flags"]],
            "source": "representative",
        }
    return _safe(build, {"ok": False, "carrier_id": carrier_id, "error": "unavailable"})


# ---- Load ↔ carrier matching -------------------------------------------------
_LOADS = [
    ("LD-9001", "São Paulo, SP", "Rio de Janeiro, RJ", "Dry Van", 430, "2026-07-23", 42000),
    ("LD-9002", "Belo Horizonte, MG", "São Paulo, SP", "Reefer", 585, "2026-07-23", 38000),
    ("LD-9003", "Curitiba, PR", "Porto Alegre, RS", "Flatbed", 710, "2026-07-24", 46000),
    ("LD-9004", "Salvador, BA", "Recife, PE", "Dry Van", 840, "2026-07-24", 30000),
]


def _matches_for(load_id: str, origin: str) -> list[dict[str, Any]]:
    carriers = [c for c in _carriers() if c["stage"] in ("approved", "monitoring")]
    ranked: list[dict[str, Any]] = []
    for c in carriers:
        h = _seed(load_id + c["id"])
        lane_history = h % 40
        capacity = 1 + h % 5
        on_time = 82 + h % 17
        # Higher lane history / on-time / capacity, lower risk → better fit.
        fit = round(min(99, 0.4 * on_time + lane_history + 4 * capacity - 0.3 * c["risk_score"]))
        ranked.append({
            "carrier": c["name"], "carrier_id": c["id"], "fit_score": max(20, fit),
            "lane_loads": lane_history, "trucks_available": capacity,
            "on_time_pct": on_time, "risk_score": c["risk_score"],
        })
    ranked.sort(key=lambda m: m["fit_score"], reverse=True)
    return ranked[:3]


def _loads() -> list[dict[str, Any]]:
    out = []
    for lid, o, d, eq, miles, pickup, weight in _LOADS:
        out.append({
            "id": lid, "origin": o, "destination": d, "equipment": eq,
            "miles": miles, "pickup": pickup, "weight_lbs": weight,
            "matches": _matches_for(lid, o),
        })
    return out


# ---- Instant spot quote ------------------------------------------------------
_RATE_PER_MILE = {"Dry Van": 2.35, "Reefer": 2.95, "Flatbed": 2.70, "Expedited": 3.80}
_FUEL_PER_MILE = 0.48


def quote(origin: str, destination: str, equipment: str, miles: int = 0) -> dict[str, Any]:
    """Generate a representative spot quote with a transparent rate breakdown."""
    def build() -> dict[str, Any]:
        eq = equipment if equipment in _RATE_PER_MILE else "Dry Van"
        dist = int(miles) if miles else 350 + _seed(f"{origin}{destination}") % 700
        linehaul = round(dist * _RATE_PER_MILE[eq], 2)
        fuel = round(dist * _FUEL_PER_MILE, 2)
        accessorials = round(0.06 * (linehaul + fuel), 2)
        carrier_cost = round(linehaul + fuel + accessorials, 2)
        margin_pct = 0.16
        all_in = round(carrier_cost / (1 - margin_pct), 2)
        return {
            "origin": origin, "destination": destination, "equipment": eq,
            "miles": dist, "transit_days": max(1, round(dist / 500)),
            "breakdown": [
                {"label": "Linehaul", "amount": linehaul, "basis": f"{dist} mi × ${_RATE_PER_MILE[eq]}/mi"},
                {"label": "Fuel surcharge", "amount": fuel, "basis": f"{dist} mi × ${_FUEL_PER_MILE}/mi"},
                {"label": "Accessorials (est.)", "amount": accessorials, "basis": "6% of linehaul+fuel"},
            ],
            "carrier_cost": carrier_cost,
            "margin_pct": round(margin_pct * 100),
            "all_in_rate": all_in,
            "margin_usd": round(all_in - carrier_cost, 2),
            "source": "representative",
        }
    return _safe(build, {"origin": origin, "destination": destination,
                         "equipment": equipment, "breakdown": [], "all_in_rate": 0, "source": "fallback"})


# ---- Inbound email triage ----------------------------------------------------
_EMAILS = [
    ("dispatch@globalmart.com", "Load tender — SP to RJ, dry van, Thu pickup", "load_tender",
     {"lane": "São Paulo → Rio de Janeiro", "equipment": "Dry Van", "pickup": "Thu"}),
    ("logistics@sethmar.com", "Rate request: Belo Horizonte to São Paulo reefer", "quote_request",
     {"lane": "Belo Horizonte → São Paulo", "equipment": "Reefer"}),
    ("driver@apexcarriers.com", "Running 2 hrs late on LD-9001, still on track for ETA", "check_call",
     {"load": "LD-9001", "status": "delayed 2h"}),
    ("ap@northwindfreight.com", "Invoice INV-8842 attached for PO-55210", "invoice",
     {"invoice": "INV-8842", "po": "PO-55210"}),
    ("ops@heartlandlg.com", "OS&D — 2 pallets damaged on delivery, photos attached", "claim",
     {"type": "damage", "quantity": "2 pallets"}),
    ("newsletter@freightwaves.com", "This week in freight markets", "general", {}),
]

_TRIAGE_LABELS = {
    "load_tender": ("Load tender", "good", "Auto-create load & match carriers"),
    "quote_request": ("Quote request", "info", "Generate spot quote & reply"),
    "check_call": ("Check call", "info", "Update shipment status"),
    "invoice": ("Invoice", "warning", "Route to document match"),
    "claim": ("OS&D claim", "critical", "Open claim & notify"),
    "general": ("General", "neutral", "No action needed"),
}


def _triage() -> list[dict[str, Any]]:
    out = []
    for i, (frm, subject, kind, extracted) in enumerate(_EMAILS):
        h = _seed(frm + subject)
        label, status, action = _TRIAGE_LABELS[kind]
        out.append({
            "id": f"EM-{3300 + i}", "from": frm, "subject": subject,
            "type": kind, "type_label": label, "type_status": status,
            "confidence": min(98, 78 + h % 20),
            "extracted": extracted, "suggested_action": action,
            "minutes_ago": 2 + (h % 240),
        })
    out.sort(key=lambda e: e["minutes_ago"])
    return out


# ---- Roadmap (infra-heavy — surfaced, not faked) -----------------------------
_ROADMAP = [
    {"name": "Voice check-calls & tracking", "detail": "AI phones drivers for location/ETA (telephony)."},
    {"name": "Dock appointment scheduling", "detail": "Books warehouse appointments via portals/EDI."},
    {"name": "Broker sales CRM", "detail": "Shipper prospecting & automated outreach."},
    {"name": "Factoring / quick-pay", "detail": "Carrier payment financing integration."},
    {"name": "Broker-TMS / EDI", "detail": "McLeod · Aljex · Turvo · EDI 204/214/210."},
]


def _summary(carriers: list[dict[str, Any]], loads: list[dict[str, Any]],
             triage: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "carriers_onboarded": sum(1 for c in carriers if c["stage"] in ("approved", "monitoring")),
        "pending_vetting": sum(1 for c in carriers if c["stage"] in ("docs", "verification")),
        "high_risk_carriers": sum(1 for c in carriers if c["risk_severity"] == "high"),
        "open_loads": len(loads),
        "open_claims": sum(1 for e in triage if e["type"] == "claim"),
        "triage_queue": len(triage),
    }


def overview() -> dict[str, Any]:
    """Full Freight Operations payload. Representative + labelled."""
    def build() -> dict[str, Any]:
        carriers = _carriers()
        loads = _loads()
        triage = _triage()
        return {
            "summary": _summary(carriers, loads, triage),
            "carriers": carriers,
            "loads": loads,
            "triage": triage,
            "roadmap": _ROADMAP,
            "source": "representative",
        }
    return _safe(build, {"summary": {}, "carriers": [], "loads": [], "triage": [],
                         "roadmap": _ROADMAP, "source": "fallback"})
