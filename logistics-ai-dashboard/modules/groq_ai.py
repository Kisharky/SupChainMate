"""
modules/groq_ai.py
Groq-powered AI features for SupChainMate.
Groq provides sub-second LLM inference — ideal for real-time insights.

Features:
  1. supply_chain_copilot() — primary AI copilot (replaces NVIDIA stream)
  2. generate_auto_insights() — 3 instant AI insights surfaced on dashboard load
  3. generate_executive_narrative() — AI-written executive summary paragraph
  4. smart_column_detect() — AI-assisted column mapping for ambiguous uploads
"""

from __future__ import annotations

import os
from typing import Optional

# ── Load Groq key ──────────────────────────────────────────────────────────────
def _get_key() -> Optional[str]:
    val = os.environ.get("GROQ_API_KEY")
    if val:
        return val
    for path in [".env", "logistics-ai-dashboard/.env"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        return line.split("=", 1)[1].strip()
    return None


GROQ_KEY  = _get_key()
MODEL     = "llama-3.3-70b-versatile"   # Best quality/speed ratio on Groq
MODEL_FAST = "llama-3.1-8b-instant"     # For non-critical fast calls


def _groq_client():
    """Return an initialised Groq client (lazy import)."""
    if not GROQ_KEY:
        return None
    try:
        from groq import Groq
        return Groq(api_key=GROQ_KEY)
    except ImportError:
        return None


def _call(
    messages: list[dict],
    model: str = MODEL,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """Core Groq call — returns response text or an error string."""
    client = _groq_client()
    if client is None:
        return "[Groq not configured — install groq package and set GROQ_API_KEY in .env]"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq error: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Supply Chain Copilot (Primary AI — replaces hardcoded responses)
# ══════════════════════════════════════════════════════════════════════════════

COPILOT_SYSTEM = """You are SupChainMate, an expert autonomous supply chain decision AI.
You have real-time access to the user's supply chain metrics shown below.

Rules:
- Always cite specific numbers from the metrics when relevant
- Give a precise, actionable recommendation — not vague advice
- Use supply chain terminology: lead time, safety stock, EOQ, ROP, SKU, 3PL, bullwhip effect, etc.
- Be concise: 3-5 sentences maximum unless the user asks for more detail
- Never say "I don't know" — use the available metrics to reason and estimate
- Format plainly — no bullet points, no markdown headers"""


def supply_chain_copilot(user_query: str, context: dict) -> str:
    """
    Primary AI copilot powered by Groq (LLaMA-3.3-70B).
    Context is injected as structured metrics into the system prompt.

    Returns the AI response as a plain string.
    """
    ctx_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())
    system = COPILOT_SYSTEM + f"\n\nLIVE SYSTEM METRICS:\n{ctx_lines}"

    return _call(
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user_query},
        ],
        model=MODEL,
        max_tokens=400,
        temperature=0.35,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Auto-Insights (Instant AI Commentary on Dashboard Load)
# ══════════════════════════════════════════════════════════════════════════════

AUTO_INSIGHTS_SYSTEM = """You are a senior supply chain analyst at a top consulting firm.
Given the following supply chain metrics, generate exactly 3 sharp, specific insights.

Format your response as exactly 3 lines, each starting with:
INSIGHT 1: ...
INSIGHT 2: ...
INSIGHT 3: ...

Rules:
- Each insight must reference at least one specific number from the metrics
- Lead with the "so what" — the business implication, not just the observation
- Max 25 words per insight
- Prioritise highest-risk or highest-impact findings
- No bullet points, no markdown, no preamble"""


def generate_auto_insights(context: dict) -> list[dict]:
    """
    Generate 3 AI insights from live dashboard metrics.
    Returns a list of dicts: [{"title": ..., "text": ..., "severity": ...}]
    Very fast (uses 8B model for speed — sub-second on Groq).
    """
    ctx_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())

    raw = _call(
        messages=[
            {"role": "system",  "content": AUTO_INSIGHTS_SYSTEM},
            {"role": "user",    "content": f"Metrics:\n{ctx_lines}\n\nGenerate 3 insights."},
        ],
        model=MODEL,          # Use full model for quality insights
        max_tokens=250,
        temperature=0.25,
    )

    # Parse the structured output
    insights = []
    for line in raw.split("\n"):
        line = line.strip()
        for prefix in ["INSIGHT 1:", "INSIGHT 2:", "INSIGHT 3:"]:
            if line.startswith(prefix):
                text = line[len(prefix):].strip()
                # Determine severity from keywords
                severity = "HIGH" if any(w in text.lower() for w in
                    ["risk", "critical", "surge", "stockout", "exceed", "breach", "delay"]) \
                    else "MEDIUM" if any(w in text.lower() for w in
                    ["increase", "reduce", "optimis", "below", "above"]) \
                    else "LOW"
                insights.append({
                    "number": prefix.replace("INSIGHT ", "").replace(":", ""),
                    "text":     text,
                    "severity": severity,
                })

    # Fallback if parsing fails
    if not insights:
        insights = [{"number": "!", "text": raw[:200], "severity": "MEDIUM"}]

    return insights


# ══════════════════════════════════════════════════════════════════════════════
# 3. AI Executive Report Narrative
# ══════════════════════════════════════════════════════════════════════════════

NARRATIVE_SYSTEM = """You are a senior supply chain consultant writing an executive brief.
Write a single coherent paragraph (5-7 sentences) that synthesises the key findings
from the supply chain metrics below into a board-ready narrative.

Rules:
- Write in the third person ("The analysis reveals..." / "The system recommends...")
- Reference at least 4 specific metrics with their values
- Include: what the data shows, what the key risk is, what action is recommended
- Tone: authoritative, concise, precise — like McKinsey or BCG
- No bullet points, headers, or markdown. Plain paragraph only."""


def generate_executive_narrative(context: dict) -> str:
    """
    Generate an AI-written executive summary paragraph for the report section.
    """
    ctx_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())

    return _call(
        messages=[
            {"role": "system", "content": NARRATIVE_SYSTEM},
            {"role": "user",   "content": f"Supply chain metrics:\n{ctx_lines}"},
        ],
        model=MODEL,
        max_tokens=300,
        temperature=0.4,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Smart Column Detection (for ambiguous upload files)
# ══════════════════════════════════════════════════════════════════════════════

COLUMN_DETECT_SYSTEM = """You are a data engineering assistant. 
Given a list of column names from a CSV file, identify which column most likely 
represents each of: date, quantity, status, lead_time, latitude, longitude, cost.

Respond ONLY as JSON, no explanation:
{
  "date": "column_name or null",
  "quantity": "column_name or null", 
  "status": "column_name or null",
  "lead_time": "column_name or null",
  "latitude": "column_name or null",
  "longitude": "column_name or null",
  "cost": "column_name or null"
}"""


def smart_column_detect(columns: list[str]) -> dict:
    """
    Use Groq to intelligently map ambiguous column names to standard fields.
    Much more robust than regex for non-standard column naming.
    Falls back to empty dict if Groq is unavailable.
    """
    import json

    col_list = ", ".join(f'"{c}"' for c in columns[:30])  # Cap at 30 columns

    raw = _call(
        messages=[
            {"role": "system", "content": COLUMN_DETECT_SYSTEM},
            {"role": "user",   "content": f"Columns: [{col_list}]"},
        ],
        model=MODEL_FAST,   # Fast model fine here
        max_tokens=150,
        temperature=0.1,    # Very low — we need exact JSON
    )

    try:
        # Extract JSON from response
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

    return {}


def is_available() -> bool:
    """Return True if Groq is configured and the package is installed."""
    return _groq_client() is not None
