"""
api/agentic_ops.py — Agentic Ops Workflows.

Makes SupChainMate's agentic engine legible: a small set of signature freight
workflows shown as a full **detect → diagnose → decide → execute → report** loop,
each ending in a one-line resolution narrative ("resolved at 02:47 · saved $X ·
why"). This is the "the agent acted while you slept" story that turns the
existing Workforce + Decision Center + Planner + Risk Radar layers into something
an executive immediately understands.

Representative and labelled — the *execute* step is simulated (no live carrier
API in a demo), but the loop, guardrails, and human-approval routing are real and
reuse the existing layers. No AI engine or business logic is modified.
"""

from __future__ import annotations

from typing import Any

from api.services import _safe

_PHASES = ["detect", "diagnose", "decide", "execute", "report"]
_PHASE_LABEL = {p: p.title() for p in _PHASES}

# Curated signature workflows (representative). Each step: phase, actor, detail,
# and whether it has completed (execute/report pend when a human must approve).
_WORKFLOWS = [
    {
        "id": "AOW-OTIF-01", "kind": "otif_drift", "kind_label": "Carrier OTIF drift",
        "title": "Carrier OTIF drift — Apex Carriers",
        "trigger": "Apex Carriers on-time-in-full trending down 4.2 pts over 6 weeks",
        "status": "awaiting_approval", "when": "caught 6 weeks early",
        "saved_usd": 18400, "confidence": 88,
        "guardrails": ["≤30% volume shift per carrier", "contract change needs approval"],
        "one_liner": "Caught 6 weeks early vs a quarter-end review — reallocating volume avoids ~$18.4k in SLA penalties (pending approval).",
        "steps": [
            ("detect", "agent", "Rolling OTIF for Apex Carriers slid 94.0% → 89.8% over 6 weeks across 340 loads.", True),
            ("diagnose", "agent", "Drift concentrates on the SP→RJ lane; root cause is a sub-contracted leg added in week 3.", True),
            ("decide", "agent", "Shift 30% of SP→RJ volume to Braspress (92% OTIF) before the penalty threshold trips.", True),
            ("execute", "human", "Volume reallocation staged — awaiting approval (it changes a contracted lane).", False),
            ("report", "agent", "Six weeks of runway vs a quarter-end surprise; ~$18.4k SLA exposure removed.", False),
        ],
    },
    {
        "id": "AOW-REROUTE-02", "kind": "reroute", "kind_label": "Autonomous re-route",
        "title": "At-risk consignments re-routed overnight",
        "trigger": "02:47 depot delay puts 14 consignments at risk of missing delivery windows",
        "status": "resolved", "when": "resolved 02:51",
        "saved_usd": 26200, "confidence": 93,
        "guardrails": ["alternate rate ≤ +8%", "carrier OTIF ≥ 90%", "insured carriers only", "auto-execute within guardrails"],
        "one_liner": "Resolved at 02:51 while the team slept — 2 penalty breaches avoided, ~$26.2k saved.",
        "steps": [
            ("detect", "agent", "Scan-event monitor flagged a 2h47 depot delay at the Sydney hub; 14 consignments affected.", True),
            ("diagnose", "agent", "Freight model scored each for OTIF-breach risk; 2 bound for a penalty-bearing retail DC (>80%).", True),
            ("decide", "agent", "Sourced an alternate carrier inside guardrails (rate +6%, OTIF 94%, insured).", True),
            ("execute", "agent", "Re-booked both consignments and updated delivery instructions automatically at 02:51.", True),
            ("report", "agent", "Ops lead woke to a resolved exception — 2 breaches avoided, ~$26.2k saved.", True),
        ],
    },
    {
        "id": "AOW-CAPACITY-03", "kind": "capacity_lock", "kind_label": "Peak-capacity lock",
        "title": "Peak capacity locked ahead of November",
        "trigger": "Forecast projects lane-level capacity shortfalls 8 weeks out; peak triples volume across 3 states",
        "status": "awaiting_approval", "when": "8 weeks out",
        "saved_usd": 140000, "confidence": 84,
        "guardrails": ["lock contracted rates only", "CFO approval for capacity commitments"],
        "one_liner": "Locking contracted rates 8 weeks out vs the November spot scramble saves ~$140k (pending CFO approval).",
        "steps": [
            ("detect", "agent", "Seasonal volume curve modelled from historical consignment data across every lane.", True),
            ("diagnose", "agent", "Forecast agent projects capacity shortfalls on the 3 highest-risk lanes, 8 weeks out.", True),
            ("decide", "agent", "Recommend a carrier mix that locks contracted rates on those lanes before the squeeze.", True),
            ("execute", "human", "Allocations staged across carriers with fallback options — awaiting CFO approval.", False),
            ("report", "agent", "Committed capacity vs forecast demand per lane/shift; contracted vs spot saves ~$140k.", False),
        ],
    },
]

_STATUS_META = {
    "resolved": ("Auto-resolved", "good"),
    "awaiting_approval": ("Awaiting approval", "warning"),
    "monitoring": ("Monitoring", "info"),
}


def _shape(w: dict[str, Any]) -> dict[str, Any]:
    label, kind = _STATUS_META[w["status"]]
    steps = [{
        "phase": p, "phase_label": _PHASE_LABEL[p], "actor": actor,
        "detail": detail, "done": done,
    } for (p, actor, detail, done) in w["steps"]]
    return {
        **{k: w[k] for k in ("id", "kind", "kind_label", "title", "trigger", "status",
                             "when", "saved_usd", "confidence", "guardrails", "one_liner")},
        "status_label": label, "status_kind": kind,
        "steps": steps,
        "auto": w["status"] == "resolved",
    }


def workflows() -> dict[str, Any]:
    """The signature agentic workflows + a productivity summary."""
    def build() -> dict[str, Any]:
        items = [_shape(w) for w in _WORKFLOWS]
        return {
            "workflows": items,
            "summary": {
                "workflows_run": len(items),
                "auto_resolved": sum(1 for w in items if w["auto"]),
                "awaiting_approval": sum(1 for w in items if w["status"] == "awaiting_approval"),
                "total_saved": sum(w["saved_usd"] for w in items),
                "avg_confidence": round(sum(w["confidence"] for w in items) / len(items)),
            },
            "loop": _PHASES,
            "source": "representative",
        }
    return _safe(build, {"workflows": [], "summary": {}, "loop": _PHASES, "source": "fallback"})
