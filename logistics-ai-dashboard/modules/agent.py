"""
modules/agent.py
Agentic Copilot — an AI that doesn't just answer, it acts.

The agent has a registry of tools that operate on the live app data
(shipment board, carrier scorecard, decision engine outputs). Queries run
through Groq (LLaMA-3.3-70B) function calling when a key is configured;
otherwise a deterministic keyword router picks the tool, so every action
works with zero API keys. LLM output is used for wording only — every
number in a result comes straight from the dataframes.

run_agent() returns:
    {"reply": str, "artifacts": [ {type, title, data, filename}, ... ],
     "actions": [tool names executed], "engine": "groq" | "offline"}

Artifact types: "dataframe" (pd.DataFrame), "text" (str, downloadable).
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional

import pandas as pd

from modules import groq_ai
from modules import cost_audit as cost_audit_mod
from modules import health_check as health_mod
from modules import tender as tender_mod

MAX_AGENT_TURNS = 4

AGENT_SYSTEM = """You are SupChainMate's agentic copilot for supply chain operations.
You can CALL TOOLS to act on the user's live data: list at-risk shipments,
inspect carrier scorecards, draft carrier SLA-review emails, generate the
reorder/execution plan, and build exception summaries.

Rules:
- Prefer calling a tool over answering from memory whenever the question
  touches shipments, carriers, exceptions, reordering, or emails.
- After tools return, answer concisely (2-4 sentences) citing the specific
  numbers from the tool results. The full tables/drafts are shown to the
  user separately — do not repeat them verbatim.
- Never invent metrics. If a tool reports data is unavailable, say so.
- Plain text only, no markdown headers."""


# ══════════════════════════════════════════════════════════════════════════════
# Tools — each returns (summary_text_for_llm, artifacts_list)
# ══════════════════════════════════════════════════════════════════════════════

def _tool_at_risk(ctx: dict, limit: int = 20) -> tuple[str, list]:
    ships = ctx.get("shipments")
    if ships is None:
        return "No shipment data loaded.", []
    flagged = ships[ships["health"].isin(["AT RISK", "LATE"])].copy()
    if flagged.empty:
        return "No shipments are currently flagged AT RISK or LATE.", []
    flagged = flagged.sort_values("delay_proba", ascending=False).head(int(limit))
    cols = [c for c in ["shipment_id", "carrier", "order_date", "promised_date",
                        "health", "delay_proba"] if c in flagged.columns]
    view = flagged[cols].round({"delay_proba": 1})
    summary = (
        f"{len(ships[ships['health'].isin(['AT RISK', 'LATE'])]):,} shipments flagged "
        f"(showing top {len(view)} by ML delay risk). "
        f"Highest risk: {view.iloc[0]['shipment_id']} at {view.iloc[0]['delay_proba']:.1f}%."
    )
    return summary, [{"type": "dataframe", "title": "At-risk / late shipments",
                      "data": view, "filename": "at_risk_shipments.csv"}]


def _tool_scorecard(ctx: dict, carrier: Optional[str] = None) -> tuple[str, list]:
    score = ctx.get("scorecard")
    if score is None or len(score) == 0:
        return "No carrier data available (no carrier column in the loaded dataset).", []
    if carrier:
        match = score[score["Carrier"].str.lower() == str(carrier).lower()]
        if match.empty:
            return (f"Carrier '{carrier}' not found. Known carriers: "
                    f"{', '.join(score['Carrier'])}."), []
        r = match.iloc[0]
        txt = (f"{r['Carrier']}: grade {r['Grade']}, {r['On-Time %']}% on-time, "
               f"{r['Late']:,} late of {r['Shipments']:,} shipments, "
               f"avg delay {r['Avg Delay (days)']} days when late.")
        return txt, [{"type": "dataframe", "title": f"Scorecard — {r['Carrier']}",
                      "data": match, "filename": "carrier_scorecard.csv"}]
    best, worst = score.iloc[0], score.iloc[-1]
    txt = (f"{len(score)} carriers scored. Best: {best['Carrier']} "
           f"({best['On-Time %']}% on-time, grade {best['Grade']}). "
           f"Worst: {worst['Carrier']} ({worst['On-Time %']}% on-time, grade {worst['Grade']}).")
    return txt, [{"type": "dataframe", "title": "Carrier scorecard",
                  "data": score, "filename": "carrier_scorecard.csv"}]


def _tool_draft_email(ctx: dict, carrier: Optional[str] = None) -> tuple[str, list]:
    score = ctx.get("scorecard")
    if score is None or len(score) == 0:
        return "Cannot draft a carrier email — no carrier data available.", []
    if carrier:
        match = score[score["Carrier"].str.lower() == str(carrier).lower()]
        if match.empty:
            return (f"Carrier '{carrier}' not found. Known carriers: "
                    f"{', '.join(score['Carrier'])}."), []
        row = match.iloc[0]
    else:
        row = score.iloc[-1]  # worst performer (scorecard is sorted best-first)

    facts = (
        f"Carrier: {row['Carrier']}\n"
        f"Shipments handled: {row['Shipments']:,}\n"
        f"On-time performance: {row['On-Time %']}%\n"
        f"Late shipments: {row['Late']:,}\n"
        f"Average delay when late: {row['Avg Delay (days)']} days\n"
        f"Internal performance grade: {row['Grade']}"
    )

    body = None
    if groq_ai.is_available():
        drafted = groq_ai._call(
            messages=[
                {"role": "system", "content":
                    "Write a firm but professional email from a shipper's supply chain "
                    "manager to a freight carrier's account manager requesting a service "
                    "performance review. Use ONLY the facts provided — do not invent "
                    "numbers, names, or dates. Include: the performance data, a request "
                    "for root-cause analysis and a recovery plan, and a proposed review "
                    "call. Placeholders like [Name] are fine. Start with 'Subject:'."},
                {"role": "user", "content": facts},
            ],
            max_tokens=450, temperature=0.4,
        )
        if drafted and not drafted.startswith("["):
            body = drafted

    if body is None:  # offline template
        body = f"""Subject: Service performance review — {row['Carrier']}

Dear [Account Manager],

I am writing regarding {row['Carrier']}'s recent delivery performance on our account.

Across the {row['Shipments']:,} shipments you have handled for us, on-time
performance stands at {row['On-Time %']}%, with {row['Late']:,} shipments
delivered late by an average of {row['Avg Delay (days)']} days. This currently
grades {row['Grade']} on our internal carrier scorecard and falls short of the
service level we require.

We would like to request:
  1. A root-cause analysis of the late deliveries,
  2. A corrective action / service recovery plan with timelines,
  3. A performance review call within the next two weeks.

Please propose some times. We value the partnership and want to see the
lane performance return to target.

Kind regards,
[Your name]
Supply Chain Manager"""

    return (f"Drafted SLA-review email to {row['Carrier']} "
            f"({row['On-Time %']}% on-time, grade {row['Grade']})."), [
        {"type": "text", "title": f"Email draft — {row['Carrier']}",
         "data": body, "filename": f"email_{re.sub(r'[^A-Za-z0-9]+', '_', str(row['Carrier']).lower())}.txt"}]


def _tool_reorder_plan(ctx: dict) -> tuple[str, list]:
    plan = ctx.get("exec_plan")
    outputs = ctx.get("decision_outputs")
    if plan is None or outputs is None:
        return "Decision engine outputs not available.", []
    txt = (f"Reorder plan generated: order {outputs.eoq:,.0f} units per order, "
           f"reorder at {outputs.reorder_point:,.0f} units, keep "
           f"{outputs.safety_stock:,.0f} units safety stock. "
           f"{len(plan)} execution actions in the plan.")
    return txt, [{"type": "dataframe", "title": "Reorder / execution plan",
                  "data": plan, "filename": "reorder_plan.csv"}]


def _tool_exception_summary(ctx: dict) -> tuple[str, list]:
    ships = ctx.get("shipments")
    if ships is None:
        return "No shipment data loaded.", []
    kpis = ctx.get("kpis") or {}
    counts = ships["health"].value_counts().to_dict()
    lines = ["SUPCHAINMATE EXCEPTION DIGEST", "=" * 32, ""]
    on_time = kpis.get("on_time_pct")
    lines.append(f"Total shipments: {len(ships):,}")
    if on_time is not None and not pd.isna(on_time):
        lines.append(f"On-time delivery: {on_time:.1f}% (vs promised dates)")
    for label in ["LATE", "AT RISK", "DELIVERED LATE", "ON TRACK", "CANCELLED"]:
        if label in counts:
            lines.append(f"{label}: {counts[label]:,}")
    score = ctx.get("scorecard")
    if score is not None and len(score) > 0:
        lines.append("")
        lines.append("CARRIERS TO WATCH:")
        for _, r in score[score["Grade"].isin(["C", "D"])].iterrows():
            lines.append(f"  - {r['Carrier']}: {r['On-Time %']}% on-time, "
                         f"{r['Late']:,} late (grade {r['Grade']})")
    flagged = ships[ships["health"].isin(["AT RISK", "LATE"])]
    if len(flagged):
        top = flagged.sort_values("delay_proba", ascending=False).head(5)
        lines.append("")
        lines.append("HIGHEST-RISK OPEN SHIPMENTS:")
        for _, r in top.iterrows():
            carrier = f" ({r['carrier']})" if pd.notna(r.get("carrier")) else ""
            lines.append(f"  - {r['shipment_id']}{carrier}: {r['health']}, "
                         f"ML risk {r['delay_proba']:.1f}%")
    digest = "\n".join(lines)
    total_late = counts.get("LATE", 0) + counts.get("DELIVERED LATE", 0)
    summary = (f"Exception digest built: {total_late:,} late (incl. delivered late), "
               f"{counts.get('AT RISK', 0):,} at risk"
               + (f", on-time {on_time:.1f}%." if on_time is not None and not pd.isna(on_time) else "."))
    return summary, [{"type": "text", "title": "Exception digest",
                      "data": digest, "filename": "exception_digest.txt"}]


def _tool_cost_audit(ctx: dict) -> tuple[str, list]:
    ships = ctx.get("shipments")
    if ships is None:
        return "No shipment data loaded.", []
    audit = cost_audit_mod.run_audit(ships)
    if audit is None:
        return "No freight cost data available — add a cost column to the delivery file.", []
    k = audit["kpis"]
    digest = cost_audit_mod.audit_digest(audit)
    artifacts = [{"type": "text", "title": "Freight cost audit report",
                  "data": digest, "filename": "freight_cost_audit.txt"}]
    if len(audit["flagged"]):
        artifacts.append({"type": "dataframe", "title": "Flagged charges",
                          "data": audit["flagged"].head(200),
                          "filename": "flagged_charges.csv"})
    summary = (f"Audited ${k['total_spend']:,.0f} of freight spend: "
               f"{k['flagged_count']:,} charges flagged worth ${k['flagged_value']:,.0f} "
               f"(outliers ${k['outlier_overcharge']:,.0f}, duplicates ${k['duplicate_value']:,.0f}, "
               f"late-premiums ${k['late_premium_value']:,.0f}); re-tender opportunity "
               f"${k['retender_opportunity']:,.0f}.")
    return summary, artifacts


def _tool_health_check(ctx: dict) -> tuple[str, list]:
    ships = ctx.get("shipments")
    if ships is None:
        return "No shipment data loaded.", []
    audit = cost_audit_mod.run_audit(ships)
    hc = health_mod.run_health_check(
        shipments=ships,
        kpis=ctx.get("kpis"),
        audit=audit,
        decision_outputs=ctx.get("decision_outputs"),
        delay_risk=ctx.get("delay_risk"),
        centroid_stats=ctx.get("centroid_stats"),
    )
    report = health_mod.health_report(hc)
    summary = (f"Health check complete: {hc['score']:.0f}/100, grade {hc['grade']}"
               + (f", DIFOT {hc['difot']:.1f}%" if hc.get("difot") is not None else "")
               + f". Weakest dimension: "
               + (min(hc['dimensions'], key=lambda d: d['score'])['dimension']
                  if hc['dimensions'] else "n/a") + ".")
    return summary, [{"type": "text", "title": "Supply chain health check",
                      "data": report, "filename": "health_check.txt"}]


def _tool_tender_pack(ctx: dict) -> tuple[str, list]:
    ships = ctx.get("shipments")
    pack = tender_mod.build_tender_pack(ships, ctx.get("scorecard")) if ships is not None else None
    if pack is None:
        return "Cannot build a tender pack — no shipment data with order dates loaded.", []
    s = pack["stats"]
    artifacts = [
        {"type": "text", "title": "RFP draft", "data": pack["rfp_text"],
         "filename": "freight_rfp_draft.txt"},
        {"type": "dataframe", "title": "Tender lane summary", "data": pack["lanes"],
         "filename": "tender_lane_summary.csv"},
    ]
    if pack["carriers"] is not None:
        artifacts.append({"type": "dataframe", "title": "Incumbent carrier summary",
                          "data": pack["carriers"], "filename": "tender_carrier_summary.csv"})
    summary = (f"Tender pack built from {s['total_shipments']:,} shipments ({s['period']}): "
               f"avg {s['monthly_avg']:,.0f}/month, peak {s['peak_shipments']:,} in {s['peak_month']}"
               + (f", spend ${s['annual_spend']:,.0f}" if s['annual_spend'] else "")
               + ". RFP draft + lane summary ready.")
    return summary, artifacts


_TOOL_FUNCS: dict[str, Callable] = {
    "get_at_risk_shipments": _tool_at_risk,
    "get_carrier_scorecard": _tool_scorecard,
    "draft_carrier_email": _tool_draft_email,
    "generate_reorder_plan": _tool_reorder_plan,
    "exception_summary": _tool_exception_summary,
    "freight_cost_audit": _tool_cost_audit,
    "supply_chain_health_check": _tool_health_check,
    "generate_tender_pack": _tool_tender_pack,
}

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_at_risk_shipments",
        "description": "List shipments flagged AT RISK or LATE, highest ML delay risk first.",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer", "description": "Max rows to return (default 20)"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_carrier_scorecard",
        "description": "Get carrier performance scorecard — all carriers, or one by name.",
        "parameters": {"type": "object", "properties": {
            "carrier": {"type": "string", "description": "Carrier name (optional)"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "draft_carrier_email",
        "description": "Draft an SLA/service-review email to a carrier citing its scorecard. "
                       "Defaults to the worst-performing carrier.",
        "parameters": {"type": "object", "properties": {
            "carrier": {"type": "string", "description": "Carrier name (optional)"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "generate_reorder_plan",
        "description": "Generate the inventory reorder / execution plan (EOQ, reorder point, safety stock).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "exception_summary",
        "description": "Build a digest of all current exceptions: late, at-risk, weak carriers.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "freight_cost_audit",
        "description": "Audit freight charges for billing anomalies: cost outliers, potential "
                       "duplicate charges, premiums paid on late deliveries, re-tender opportunity.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "supply_chain_health_check",
        "description": "Run the scored supply chain health check (0-100, grade A-F) across "
                       "delivery, risk, cost, inventory, network, and data-quality dimensions.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "generate_tender_pack",
        "description": "Build a freight tender / RFP pack: monthly lane volumes, incumbent "
                       "carrier summary, and a ready-to-edit RFP document from real data.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


def _execute(name: str, args: dict, ctx: dict) -> tuple[str, list]:
    func = _TOOL_FUNCS.get(name)
    if func is None:
        return f"Unknown tool: {name}", []
    try:
        return func(ctx, **{k: v for k, v in args.items() if v is not None})
    except TypeError:
        return func(ctx)
    except Exception as e:
        return f"Tool {name} failed: {e}", []


# ══════════════════════════════════════════════════════════════════════════════
# Offline fallback — deterministic keyword routing (zero API keys needed)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_carrier(query: str, ctx: dict) -> Optional[str]:
    score = ctx.get("scorecard")
    if score is None:
        return None
    q = query.lower()
    for name in score["Carrier"]:
        if str(name).lower() in q:
            return str(name)
    return None


def _route_offline(query: str, ctx: dict) -> list[tuple[str, dict]]:
    q = query.lower()
    carrier = _extract_carrier(query, ctx)
    if re.search(r"\b(tender|rfp|bid|proposal|procurement)\b", q):
        return [("generate_tender_pack", {})]
    if re.search(r"\b(health|assessment|scorecard of (the )?(chain|network)|how healthy|grade (the|my))\b", q):
        return [("supply_chain_health_check", {})]
    if re.search(r"\b(email|draft|write|letter|sla)\b", q):
        return [("draft_carrier_email", {"carrier": carrier})]
    if re.search(r"\b(audit|invoice|billing|overcharge|duplicate|cost anomal|freight (cost|spend))\b", q):
        return [("freight_cost_audit", {})]
    if re.search(r"\b(reorder|replenish|order plan|execution plan|how much.*order)\b", q):
        return [("generate_reorder_plan", {})]
    if re.search(r"\b(summar|digest|report|overview|status update|brief)\b", q):
        return [("exception_summary", {})]
    if re.search(r"\b(at.risk|late|delayed|exception|flag)\b", q):
        return [("get_at_risk_shipments", {})]
    if re.search(r"\b(carrier|scorecard|performance|on.time|grade)\b", q) or carrier:
        return [("get_carrier_scorecard", {"carrier": carrier})]
    return [("exception_summary", {})]


# ══════════════════════════════════════════════════════════════════════════════
# Agent loop
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(query: str, ctx: dict, client=None) -> dict:
    """
    Run one agentic turn: pick tool(s), execute on live data, compose a reply.
    `client` is injectable for testing; defaults to the configured Groq client.
    """
    if client is None:
        client = groq_ai._groq_client()

    if client is not None:
        try:
            return _run_llm_agent(query, ctx, client)
        except Exception:
            pass  # fall through to offline routing

    trace = [{"step": "route", "label": "Offline keyword router",
              "detail": f"Groq unavailable — deterministic routing for: \"{query[:80]}\""}]
    artifacts, actions, notes = [], [], []
    for name, args in _route_offline(query, ctx):
        t0 = time.perf_counter()
        summary, arts = _execute(name, args, ctx)
        trace.append({"step": "tool", "label": name,
                      "detail": (f"args {json.dumps({k: v for k, v in args.items() if v})} · "
                                 if any(args.values()) else "")
                                + f"{summary[:160]}",
                      "ms": round((time.perf_counter() - t0) * 1000)})
        notes.append(summary)
        artifacts += arts
        actions.append(name)
    trace.append({"step": "answer", "label": "Compose reply",
                  "detail": f"{len(artifacts)} artifact(s) attached"})
    return {"reply": " ".join(notes), "artifacts": artifacts,
            "actions": actions, "engine": "offline", "trace": trace}


def _run_llm_agent(query: str, ctx: dict, client) -> dict:
    metrics = ctx.get("metrics") or {}
    ctx_lines = "\n".join(f"  {k}: {v}" for k, v in metrics.items())
    messages = [
        {"role": "system", "content": AGENT_SYSTEM + f"\n\nLIVE SYSTEM METRICS:\n{ctx_lines}"},
        {"role": "user", "content": query},
    ]
    artifacts, actions = [], []
    trace = [{"step": "route", "label": "Groq LLaMA-3.3-70B",
              "detail": f"Tool-calling agent · {len(TOOLS_SCHEMA)} tools available"}]

    for turn in range(MAX_AGENT_TURNS):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=groq_ai.MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            max_tokens=600,
            temperature=0.2,
        )
        llm_ms = round((time.perf_counter() - t0) * 1000)
        msg = resp.choices[0].message
        if not getattr(msg, "tool_calls", None):
            reply = (msg.content or "").strip()
            trace.append({"step": "answer", "label": "Compose reply",
                          "detail": f"LLM turn {turn + 1} · {len(artifacts)} artifact(s)",
                          "ms": llm_ms})
            return {"reply": reply or "Done.", "artifacts": artifacts,
                    "actions": actions, "engine": "groq", "trace": trace}

        trace.append({"step": "think", "label": f"LLM turn {turn + 1}",
                      "detail": "Decided to call: "
                                + ", ".join(tc.function.name for tc in msg.tool_calls),
                      "ms": llm_ms})
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            t1 = time.perf_counter()
            summary, arts = _execute(tc.function.name, args, ctx)
            trace.append({"step": "tool", "label": tc.function.name,
                          "detail": (f"args {json.dumps(args)} · " if args else "")
                                    + f"{summary[:160]}",
                          "ms": round((time.perf_counter() - t1) * 1000)})
            artifacts += arts
            actions.append(tc.function.name)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": tc.function.name, "content": summary})

    trace.append({"step": "answer", "label": "Turn limit reached",
                  "detail": f"{len(actions)} tool call(s) executed"})
    return {"reply": "Executed: " + ", ".join(actions) + ".", "artifacts": artifacts,
            "actions": actions, "engine": "groq", "trace": trace}


# Quick-action prompts surfaced as buttons in the UI.
QUICK_ACTIONS = [
    ("⚠ Exception digest", "Give me a summary of all current exceptions."),
    ("⛟ At-risk shipments", "List the shipments most at risk of delay."),
    ("✉ Email worst carrier", "Draft an SLA-review email to our worst-performing carrier."),
    ("📦 Reorder plan", "Generate the reorder plan."),
    ("⚖ Cost audit", "Audit our freight costs for billing anomalies."),
    ("🩺 Health check", "Run a supply chain health check."),
    ("📑 Tender pack", "Build a freight tender pack with an RFP draft."),
]

# ── AI Workers — the tools organised as a named team ──────────────────────────
# Purely an organisational layer over the same tool registry: each worker owns
# a subset of tools and the quick actions that exercise them.
WORKERS = {
    "Tracker": {
        "emoji": "🛰", "role": "Track & Trace",
        "desc": "Watches every shipment, surfaces exceptions first.",
        "tools": ["get_at_risk_shipments", "exception_summary"],
        "actions": [("At-risk", "List the shipments most at risk of delay."),
                    ("Digest", "Give me a summary of all current exceptions.")],
    },
    "Auditor": {
        "emoji": "⚖", "role": "Invoicing & Audit",
        "desc": "Audits freight bills: outliers, duplicates, late-premiums.",
        "tools": ["freight_cost_audit"],
        "actions": [("Audit bills", "Audit our freight costs for billing anomalies.")],
    },
    "Carrier Manager": {
        "emoji": "🤝", "role": "Carrier Vetting",
        "desc": "Grades carriers and drafts the difficult emails.",
        "tools": ["get_carrier_scorecard", "draft_carrier_email"],
        "actions": [("Scorecard", "Show me the carrier scorecard."),
                    ("Email worst", "Draft an SLA-review email to our worst-performing carrier.")],
    },
    "Procurement": {
        "emoji": "📑", "role": "Quoting & Tenders",
        "desc": "Builds data-backed RFPs and rate strategies.",
        "tools": ["generate_tender_pack"],
        "actions": [("Tender pack", "Build a freight tender pack with an RFP draft.")],
    },
    "Planner": {
        "emoji": "📦", "role": "Inventory Planning",
        "desc": "Reorder plans, safety stock, network health.",
        "tools": ["generate_reorder_plan", "supply_chain_health_check"],
        "actions": [("Reorder", "Generate the reorder plan."),
                    ("Health", "Run a supply chain health check.")],
    },
}

_TOOL_TO_WORKER = {tool: name for name, w in WORKERS.items() for tool in w["tools"]}


def autonomous_sweep(ctx: dict) -> list[dict]:
    """
    Manhattan-style background monitoring: every worker reports its live
    status from already-computed context — no LLM, no recompute.
    Returns [{worker, status, level}] with level in green/yellow/red/grey.
    """
    out = []
    kpis = ctx.get("kpis") or {}

    at_risk, late = kpis.get("at_risk", 0) or 0, kpis.get("late", 0) or 0
    out.append({"worker": "Tracker",
                "status": f"{at_risk:,} at risk · {late:,} late",
                "level": "red" if late else ("yellow" if at_risk else "green")})

    audit = ctx.get("audit")
    if audit:
        k = audit["kpis"]
        out.append({"worker": "Auditor",
                    "status": f"${k['flagged_value']:,.0f} across {k['flagged_count']:,} flagged charges",
                    "level": "red" if k["flagged_count"] else "green"})
    else:
        out.append({"worker": "Auditor", "status": "no freight cost data", "level": "grey"})

    score = ctx.get("scorecard")
    if score is not None and len(score):
        worst = score.iloc[-1]
        out.append({"worker": "Carrier Manager",
                    "status": f"worst: {worst['Carrier']} ({worst['On-Time %']}% · grade {worst['Grade']})",
                    "level": "red" if worst["Grade"] in ("C", "D") else "green"})
    else:
        out.append({"worker": "Carrier Manager", "status": "no carrier data", "level": "grey"})

    if audit and audit["kpis"]["retender_opportunity"] > 0:
        out.append({"worker": "Procurement",
                    "status": f"${audit['kpis']['retender_opportunity']:,.0f} re-tender opportunity",
                    "level": "yellow"})
    else:
        out.append({"worker": "Procurement", "status": "no re-tender opportunity found",
                    "level": "green" if audit else "grey"})

    hc = ctx.get("health")
    if hc:
        out.append({"worker": "Planner",
                    "status": f"network health {hc['grade']} ({hc['score']:.0f}/100)",
                    "level": {"A": "green", "B": "green", "C": "yellow"}.get(hc["grade"], "red")})
    else:
        out.append({"worker": "Planner", "status": "health check pending", "level": "grey"})
    return out


def workers_for_actions(actions: list[str]) -> list[str]:
    """Which named workers handled these executed tools (ordered, deduped)."""
    seen: list[str] = []
    for a in actions:
        w = _TOOL_TO_WORKER.get(a)
        if w and w not in seen:
            seen.append(w)
    return seen
