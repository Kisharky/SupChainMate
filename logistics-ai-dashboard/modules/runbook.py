"""
modules/runbook.py
Runbooks — plain-English standing rules the AI Workers apply on every
data load (Sema4.ai-style runbooks / Manhattan-style autonomous agents).

A rule is written in natural language ("flag any shipment over $50",
"alert me when SwiftLine on-time drops below 95%"), parsed into a
structured condition by a deterministic grammar, auto-assigned to the
right worker, persisted in SQLite, and evaluated against live data on
every load. Evaluation is pure computation — no LLM in the loop.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from modules import store

# metric → (worker, description)
_METRICS = {
    "shipment_cost": ("Auditor", "freight cost per shipment ($)"),
    "on_time": ("Carrier Manager", "carrier on-time performance (%)"),
    "ml_risk": ("Tracker", "ML delay probability (%)"),
    "delay_days": ("Tracker", "days delivered past promise"),
    "at_risk_count": ("Tracker", "shipments flagged at risk"),
    "late_count": ("Tracker", "late shipments"),
    "health_score": ("Planner", "supply chain health score"),
    "retender": ("Procurement", "re-tender opportunity ($)"),
}

_NUM = r"\$?\s*([\d,]+(?:\.\d+)?)\s*%?"
_ABOVE = r"(?:over|above|more than|exceed(?:s|ing)?|greater than|at least|>=?)"
_BELOW = r"(?:under|below|less than|drops? below|falls? below|worse than|<=?)"


def parse_rule(text: str) -> Optional[dict]:
    """
    Parse a plain-English rule into {text, metric, op, threshold, carrier,
    action, worker}. Returns None when no pattern matches.
    """
    t = " ".join(str(text).lower().split())
    if not t:
        return None
    action = "alert" if re.search(r"\b(alert|email|notify|tell) me\b|\bemail\b", t) else "flag"

    def _num(m) -> float:
        return float(m.replace(",", ""))

    # carrier on-time: "when <carrier> on-time drops below 95" / "any carrier below 90% on-time"
    m = re.search(rf"(?P<carrier>[\w .\-]+?)?\s*on.?time\b.*?{_BELOW}\s*{_NUM}", t)
    if m:
        carrier = (m.group("carrier") or "").strip()
        carrier = re.sub(r"^(?:alert me|notify me|email me|tell me|flag|warn me)?\s*"
                         r"(?:when|if|whenever)?\s*", "", carrier).strip()
        carrier = None if carrier in ("", "any carrier", "a carrier", "carrier", "any", "the") else carrier
        return {"text": text, "metric": "on_time", "op": "<", "threshold": _num(m.group(2)),
                "carrier": carrier, "action": action, "worker": "Carrier Manager"}

    patterns = [
        ("delay_days", rf"(?:deliver|delay|late|shipment).*?{_ABOVE}\s*{_NUM}\s*days?\b", ">"),
        ("shipment_cost", rf"(?:shipment|freight|charge)s?\b.*?(?:cost(?:ing)?\s*)?{_ABOVE}\s*{_NUM}", ">"),
        ("ml_risk", rf"(?:ml |delay )?risk\b.*?{_ABOVE}\s*{_NUM}", ">"),
        ("at_risk_count", rf"at.?risk\b.*?{_ABOVE}\s*{_NUM}", ">"),
        ("late_count", rf"late (?:shipments?|count)\b.*?{_ABOVE}\s*{_NUM}", ">"),
        ("health_score", rf"health\b.*?{_BELOW}\s*{_NUM}", "<"),
        ("retender", rf"re.?tender\b.*?{_ABOVE}\s*{_NUM}", ">"),
    ]
    for metric, pattern, op in patterns:
        m = re.search(pattern, t)
        if m:
            return {"text": text, "metric": metric, "op": op,
                    "threshold": _num(m.group(1)), "carrier": None,
                    "action": action, "worker": _METRICS[metric][0]}
    return None


def evaluate_rule(rule: dict, ctx: dict) -> dict:
    """
    Evaluate one rule against live context (shipments, kpis, scorecard,
    audit, health). Returns {**rule, triggered, detail}.
    """
    metric, thr = rule["metric"], float(rule["threshold"])
    ships: Optional[pd.DataFrame] = ctx.get("shipments")
    triggered, detail = False, "no data for this rule"

    if metric == "shipment_cost" and ships is not None and "freight_cost" in ships.columns:
        hits = ships[ships["freight_cost"] > thr]
        triggered = len(hits) > 0
        detail = (f"{len(hits):,} shipment(s) over ${thr:,.2f} "
                  f"(total ${hits['freight_cost'].sum():,.0f})") if triggered else \
            f"no shipments over ${thr:,.2f}"
    elif metric == "on_time":
        score = ctx.get("scorecard")
        if score is not None and len(score):
            rows = score
            if rule.get("carrier"):
                rows = score[score["Carrier"].str.lower() == rule["carrier"].lower()]
                if rows.empty:
                    return {**rule, "triggered": False,
                            "detail": f"carrier '{rule['carrier']}' not found"}
            bad = rows[rows["On-Time %"] < thr]
            triggered = len(bad) > 0
            detail = (", ".join(f"{r['Carrier']} at {r['On-Time %']}%" for _, r in bad.iterrows())
                      + f" — below {thr:g}%") if triggered else \
                f"all monitored carriers at or above {thr:g}% on-time"
    elif metric == "ml_risk" and ships is not None and "delay_proba" in ships.columns:
        open_mask = ~ships["health"].isin(["DELIVERED ON TIME", "DELIVERED LATE", "CANCELLED"])
        hits = ships[open_mask & (ships["delay_proba"] > thr)]
        triggered = len(hits) > 0
        detail = f"{len(hits):,} open shipment(s) above {thr:g}% ML risk" if triggered else \
            f"no open shipments above {thr:g}% ML risk"
    elif metric == "delay_days" and ships is not None and "delay_days" in ships.columns:
        hits = ships[ships["delay_days"] > thr]
        triggered = len(hits) > 0
        detail = f"{len(hits):,} shipment(s) delivered more than {thr:g} days late" if triggered else \
            f"no shipments more than {thr:g} days late"
    elif metric == "at_risk_count":
        v = (ctx.get("kpis") or {}).get("at_risk")
        if v is not None:
            triggered = v > thr
            detail = f"{v:,} at-risk shipment(s) (threshold {thr:g})"
    elif metric == "late_count":
        v = (ctx.get("kpis") or {}).get("late")
        if v is not None:
            triggered = v > thr
            detail = f"{v:,} late shipment(s) (threshold {thr:g})"
    elif metric == "health_score":
        hc = ctx.get("health")
        if hc:
            triggered = hc["score"] < thr
            detail = f"health score {hc['score']:.0f}/100 (threshold {thr:g})"
    elif metric == "retender":
        audit = ctx.get("audit")
        if audit:
            v = audit["kpis"]["retender_opportunity"]
            triggered = v > thr
            detail = f"re-tender opportunity ${v:,.0f} (threshold ${thr:,.0f})"

    return {**rule, "triggered": bool(triggered), "detail": detail}


def evaluate_all(rules: list[dict], ctx: dict) -> list[dict]:
    return [evaluate_rule(r, ctx) for r in rules]


# ── Persistence ────────────────────────────────────────────────────────────────

def load_rules() -> list[dict]:
    return store.load_setting("runbook_rules", []) or []


def save_rules(rules: list[dict]) -> bool:
    return store.save_setting("runbook_rules", rules)


def add_rule(text: str) -> Optional[dict]:
    rule = parse_rule(text)
    if rule is None:
        return None
    rules = load_rules()
    rules.append(rule)
    save_rules(rules)
    return rule


def remove_rule(index: int) -> None:
    rules = load_rules()
    if 0 <= index < len(rules):
        rules.pop(index)
        save_rules(rules)


def runbook_digest(results: list[dict]) -> str:
    """Plain-text section for the alert digest."""
    fired = [r for r in results if r["triggered"]]
    if not fired:
        return "RUNBOOK: all standing rules clear."
    lines = [f"RUNBOOK: {len(fired)} rule(s) triggered:"]
    lines += [f"  - [{r['worker']}] \"{r['text']}\" -> {r['detail']}" for r in fired]
    return "\n".join(lines)
