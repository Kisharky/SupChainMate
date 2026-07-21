"""
planner/prompts.py — prompt templates used when the AI Router is reachable. The
aggregator always has a deterministic fallback, so these are enhancements, not
dependencies.
"""

from __future__ import annotations


def executive_synthesis(objective: str, findings: list[str], impact: float) -> str:
    joined = "\n".join(f"- {f}" for f in findings[:10])
    return (
        "You are the executive decision layer of a supply-chain operating system. "
        "Synthesise ONE concise, decision-ready recommendation (3-4 sentences) for "
        f"the objective: \"{objective}\".\n\nSpecialist findings:\n{joined}\n\n"
        f"Aggregate financial impact identified: ${impact:,.0f}. State the decision, "
        "the expected impact, and the single most important next action. Executive tone, "
        "no preamble.")
