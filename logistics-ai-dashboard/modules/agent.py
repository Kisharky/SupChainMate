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
from typing import Callable, Optional

import pandas as pd

from modules import groq_ai

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


_TOOL_FUNCS: dict[str, Callable] = {
    "get_at_risk_shipments": _tool_at_risk,
    "get_carrier_scorecard": _tool_scorecard,
    "draft_carrier_email": _tool_draft_email,
    "generate_reorder_plan": _tool_reorder_plan,
    "exception_summary": _tool_exception_summary,
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
    if re.search(r"\b(email|draft|write|letter|sla)\b", q):
        return [("draft_carrier_email", {"carrier": carrier})]
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

    artifacts, actions, notes = [], [], []
    for name, args in _route_offline(query, ctx):
        summary, arts = _execute(name, args, ctx)
        notes.append(summary)
        artifacts += arts
        actions.append(name)
    return {"reply": " ".join(notes), "artifacts": artifacts,
            "actions": actions, "engine": "offline"}


def _run_llm_agent(query: str, ctx: dict, client) -> dict:
    metrics = ctx.get("metrics") or {}
    ctx_lines = "\n".join(f"  {k}: {v}" for k, v in metrics.items())
    messages = [
        {"role": "system", "content": AGENT_SYSTEM + f"\n\nLIVE SYSTEM METRICS:\n{ctx_lines}"},
        {"role": "user", "content": query},
    ]
    artifacts, actions = [], []

    for _ in range(MAX_AGENT_TURNS):
        resp = client.chat.completions.create(
            model=groq_ai.MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            max_tokens=600,
            temperature=0.2,
        )
        msg = resp.choices[0].message
        if not getattr(msg, "tool_calls", None):
            reply = (msg.content or "").strip()
            return {"reply": reply or "Done.", "artifacts": artifacts,
                    "actions": actions, "engine": "groq"}

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
            summary, arts = _execute(tc.function.name, args, ctx)
            artifacts += arts
            actions.append(tc.function.name)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": tc.function.name, "content": summary})

    return {"reply": "Executed: " + ", ".join(actions) + ".", "artifacts": artifacts,
            "actions": actions, "engine": "groq"}


# Quick-action prompts surfaced as buttons in the UI.
QUICK_ACTIONS = [
    ("⚠ Exception digest", "Give me a summary of all current exceptions."),
    ("⛟ At-risk shipments", "List the shipments most at risk of delay."),
    ("✉ Email worst carrier", "Draft an SLA-review email to our worst-performing carrier."),
    ("📦 Reorder plan", "Generate the reorder plan."),
]
