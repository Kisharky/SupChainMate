"""
api/risk_radar.py — Disruption & Risk Radar (signal convergence).

A situational-awareness surface inspired by geopolitical monitors, translated to
supply chain: disruption *layers* (weather, port congestion, supplier failure,
labour, customs, geopolitical, carrier), a composite **Supply Chain Risk Index**,
and a **convergence engine** — a node or lane escalates only when several
independent signals line up at once, which is what cuts alert fatigue.

Signals are representative and labelled (deterministic). Node coordinates are
real so the flat map and 3D globe render legitimately. The layers are the seam
where real weather / AIS / news / financial feeds plug in — the API surface and
UI stay identical. No AI engine or business logic is modified; the "why"
narrative uses a deterministic template (AI-router enhancement is optional).
"""

from __future__ import annotations

from typing import Any

from api.services import _safe

# Disruption layers — the toggleable categories (structure copied, not the count).
LAYERS = [
    {"id": "weather", "name": "Weather & storms", "icon": "🌀", "color": "#38BDF8"},
    {"id": "port", "name": "Port congestion", "icon": "⚓", "color": "#F59E0B"},
    {"id": "supplier", "name": "Supplier failure", "icon": "🏭", "color": "#EF4444"},
    {"id": "labor", "name": "Labour & strikes", "icon": "✊", "color": "#A855F7"},
    {"id": "customs", "name": "Customs & border", "icon": "🛃", "color": "#14B8A6"},
    {"id": "geopolitical", "name": "Geopolitical", "icon": "⚑", "color": "#F43F5E"},
    {"id": "carrier", "name": "Carrier risk", "icon": "🚚", "color": "#EAB308"},
]
_LAYER_NAME = {l["id"]: l["name"] for l in LAYERS}
_LAYER_IDS = [l["id"] for l in LAYERS]

# Threshold (0-100) above which a layer counts as an "active" signal on a node.
_ACTIVE = 40
# Number of converging layers at which a node/lane is flagged critical.
_CONVERGE_AT = 3

# Network nodes: real coordinates so the map/globe look legitimate. Each carries
# a per-layer signal severity (0-100). Convergence & risk are derived from these.
# (id, name, type, lat, lon, region, {layer: severity})
_NODES = [
    ("santos", "Port of Santos", "port", -23.96, -46.33, "South America",
     {"port": 68, "labor": 72, "customs": 55}),
    ("saopaulo", "São Paulo DC", "hub", -23.55, -46.63, "South America",
     {"labor": 45, "carrier": 38}),
    ("rio", "Rio de Janeiro", "hub", -22.91, -43.17, "South America",
     {"weather": 44, "carrier": 30}),
    ("manaus", "Manaus supplier", "supplier", -3.12, -60.02, "South America",
     {"supplier": 78, "customs": 40}),
    ("shanghai", "Port of Shanghai", "port", 31.23, 121.47, "Asia",
     {"port": 82, "weather": 74, "customs": 48}),
    ("shenzhen", "Shenzhen supplier", "supplier", 22.54, 114.06, "Asia",
     {"supplier": 52, "carrier": 46, "labor": 35}),
    ("singapore", "Port of Singapore", "port", 1.29, 103.85, "Asia",
     {"port": 41, "weather": 30}),
    ("mumbai", "Nhava Sheva (Mumbai)", "port", 18.95, 72.95, "Asia",
     {"customs": 66, "port": 44, "carrier": 40}),
    ("rotterdam", "Port of Rotterdam", "port", 51.95, 4.14, "Europe",
     {"labor": 58, "weather": 36}),
    ("hamburg", "Hamburg", "port", 53.55, 9.99, "Europe", {"labor": 42}),
    ("lalb", "LA / Long Beach", "port", 33.77, -118.19, "North America",
     {"port": 61, "carrier": 44}),
    ("houston", "Houston DC", "dc", 29.76, -95.37, "North America", {"weather": 33}),
    ("suez", "Suez Canal", "chokepoint", 30.02, 32.55, "Middle East",
     {"geopolitical": 71, "port": 52}),
    ("hormuz", "Strait of Hormuz", "chokepoint", 26.57, 56.25, "Middle East",
     {"geopolitical": 84, "carrier": 40}),
    ("panama", "Panama Canal", "chokepoint", 9.08, -79.68, "Central America",
     {"weather": 63, "port": 45}),
]

# Lanes (arcs) between nodes.
_LANES = [
    ("shanghai", "santos"), ("santos", "saopaulo"), ("rotterdam", "santos"),
    ("shenzhen", "singapore"), ("singapore", "suez"), ("suez", "rotterdam"),
    ("mumbai", "hormuz"), ("lalb", "houston"), ("panama", "lalb"), ("shanghai", "lalb"),
]


def _band(score: float) -> tuple[str, str]:
    if score >= 75:
        return "Severe", "critical"
    if score >= 50:
        return "High", "warning"
    if score >= 25:
        return "Elevated", "info"
    return "Low", "good"


def _node_risk(signals: dict[str, int]) -> int:
    """Composite node risk: the peak signal dominates, with convergence adding
    pressure (multiple active signals are worse than one loud one)."""
    if not signals:
        return 0
    peak = max(signals.values())
    active = [v for v in signals.values() if v >= _ACTIVE]
    convergence_bonus = 6 * max(0, len(active) - 1)
    return min(99, round(peak + convergence_bonus))


def _nodes() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for nid, name, kind, lat, lon, region, signals in _NODES:
        risk = _node_risk(signals)
        band, status = _band(risk)
        active = sorted(((k, v) for k, v in signals.items() if v >= _ACTIVE),
                        key=lambda kv: kv[1], reverse=True)
        out.append({
            "id": nid, "name": name, "type": kind, "lat": lat, "lon": lon, "region": region,
            "risk_score": risk, "band": band, "status": status,
            "signals": signals,
            "convergence": len(active),
            "top_signals": [{"layer": k, "layer_name": _LAYER_NAME[k], "severity": v} for k, v in active],
        })
    return out


def _node_map(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {n["id"]: n for n in nodes}


def _lanes(nmap: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for a, b in _LANES:
        na, nb = nmap[a], nmap[b]
        # Lane risk: the worse endpoint drives it, nudged by the other.
        risk = round(0.7 * max(na["risk_score"], nb["risk_score"])
                     + 0.3 * min(na["risk_score"], nb["risk_score"]))
        band, status = _band(risk)
        # Converging categories seen anywhere on the lane.
        cats = {k for n in (na, nb) for k, v in n["signals"].items() if v >= _ACTIVE}
        out.append({
            "id": f"{a}-{b}", "from_id": a, "to_id": b, "from": na["name"], "to": nb["name"],
            "from_lat": na["lat"], "from_lon": na["lon"], "to_lat": nb["lat"], "to_lon": nb["lon"],
            "risk_score": risk, "band": band, "status": status,
            "convergence": len(cats),
            "categories": sorted(cats),
        })
    out.sort(key=lambda l: l["risk_score"], reverse=True)
    return out


def _layers(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for layer in LAYERS:
        lid = layer["id"]
        events = [{
            "node_id": n["id"], "node": n["name"], "lat": n["lat"], "lon": n["lon"],
            "severity": n["signals"][lid], "band": _band(n["signals"][lid])[0],
        } for n in nodes if n["signals"].get(lid, 0) >= _ACTIVE]
        events.sort(key=lambda e: e["severity"], reverse=True)
        out.append({**layer, "active_events": len(events), "events": events})
    return out


def _why(node: dict[str, Any]) -> str:
    top = node["top_signals"]
    if not top:
        return "No active disruption signals."
    parts = [f"{s['layer_name'].lower()} ({s['severity']})" for s in top]
    joined = ", ".join(parts[:-1]) + (f", and {parts[-1]}" if len(parts) > 1 else parts[0])
    return (f"{node['name']} shows {len(top)} converging signals — {joined}. "
            f"When these line up, delivery risk compounds rather than adds.")


def _recommended(node: dict[str, Any]) -> str:
    cats = {s["layer"] for s in node["top_signals"]}
    if "geopolitical" in cats:
        return "Pre-position inventory and secure alternate routing around the chokepoint."
    if "supplier" in cats:
        return "Qualify a backup supplier and increase safety stock on affected SKUs."
    if "port" in cats or "customs" in cats:
        return "Divert to an alternate port and expedite customs pre-clearance."
    if "labor" in cats:
        return "Book capacity ahead of the strike window and notify affected customers."
    return "Add buffer to ETAs and monitor the converging signals."


def _alerts(nodes: list[dict[str, Any]], lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convergence alerts: only nodes/lanes where multiple independent signals
    line up. This is the whole point — noise stays off the board."""
    out: list[dict[str, Any]] = []
    for n in nodes:
        if n["convergence"] >= 2:
            out.append({
                "id": f"CV-{n['id']}", "scope": "node", "ref_id": n["id"], "name": n["name"],
                "region": n["region"], "convergence": n["convergence"],
                "categories": [s["layer_name"] for s in n["top_signals"]],
                "composite_score": n["risk_score"], "band": n["band"], "status": n["status"],
                "critical": n["convergence"] >= _CONVERGE_AT,
                "why": _why(n), "recommended_action": _recommended(n),
            })
    out.sort(key=lambda a: (a["convergence"], a["composite_score"]), reverse=True)
    return out


def _index(nodes: list[dict[str, Any]], alerts: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [n["risk_score"] for n in nodes]
    # Index leans on the worst nodes (a network is as risky as its hot spots).
    top = sorted(scores, reverse=True)[:5]
    score = round(sum(top) / len(top)) if top else 0
    band, status = _band(score)
    by_category = []
    for layer in LAYERS:
        sev = max((n["signals"].get(layer["id"], 0) for n in nodes), default=0)
        by_category.append({"id": layer["id"], "name": layer["name"], "severity": sev,
                            "band": _band(sev)[0]})
    by_category.sort(key=lambda c: c["severity"], reverse=True)
    by_region: dict[str, list[int]] = {}
    for n in nodes:
        by_region.setdefault(n["region"], []).append(n["risk_score"])
    regions = [{"region": r, "score": round(max(v)), "band": _band(max(v))[0]}
               for r, v in by_region.items()]
    regions.sort(key=lambda r: r["score"], reverse=True)
    return {
        "score": score, "band": band, "status": status,
        "critical_alerts": sum(1 for a in alerts if a["critical"]),
        "converging_alerts": len(alerts),
        "by_category": by_category, "by_region": regions,
    }


def _brief(alerts: list[dict[str, Any]], index: dict[str, Any]) -> str:
    if not alerts:
        return "Network stable — no converging disruption signals."
    top = alerts[:2]
    lead = "; ".join(f"{a['name']} ({a['convergence']} signals: {', '.join(a['categories'][:3])})"
                     for a in top)
    return (f"Supply Chain Risk Index at {index['score']} ({index['band']}). "
            f"{index['critical_alerts']} critical convergence(s). Watch: {lead}.")


def node_detail(node_id: str) -> dict[str, Any]:
    """Per-node signal breakdown, the convergence 'why', and a recommended
    action — plus the lanes it touches."""
    def build() -> dict[str, Any]:
        nodes = _nodes()
        n = next((x for x in nodes if x["id"] == node_id), None)
        if n is None:
            return {"ok": False, "error": "unknown node", "node_id": node_id}
        lanes = _lanes(_node_map(nodes))
        touched = [l for l in lanes if node_id in (l["from_id"], l["to_id"])]
        breakdown = [{"layer": lid, "layer_name": _LAYER_NAME[lid],
                      "severity": n["signals"].get(lid, 0),
                      "active": n["signals"].get(lid, 0) >= _ACTIVE} for lid in _LAYER_IDS]
        breakdown.sort(key=lambda b: b["severity"], reverse=True)
        return {
            "ok": True, "node_id": node_id, "name": n["name"], "type": n["type"],
            "region": n["region"], "risk_score": n["risk_score"], "band": n["band"],
            "status": n["status"], "convergence": n["convergence"],
            "signals": breakdown, "why": _why(n), "recommended_action": _recommended(n),
            "lanes": [{"to": l["to"] if l["from_id"] == node_id else l["from"],
                       "risk_score": l["risk_score"], "band": l["band"]} for l in touched],
            "source": "representative",
        }
    return _safe(build, {"ok": False, "node_id": node_id, "error": "unavailable"})


def overview() -> dict[str, Any]:
    """Full radar payload: index, nodes, lanes, layers, convergence alerts, brief."""
    def build() -> dict[str, Any]:
        nodes = _nodes()
        lanes = _lanes(_node_map(nodes))
        alerts = _alerts(nodes, lanes)
        index = _index(nodes, alerts)
        return {
            "index": index,
            "nodes": nodes,
            "lanes": lanes,
            "layers": _layers(nodes),
            "alerts": alerts,
            "brief": _brief(alerts, index),
            "converge_at": _CONVERGE_AT,
            "source": "representative",
        }
    return _safe(build, {"index": {}, "nodes": [], "lanes": [], "layers": LAYERS,
                         "alerts": [], "brief": "", "source": "fallback"})
