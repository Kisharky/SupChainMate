"""
ai/reasoning.py
Reasoning service — task-shaped helpers over the router's reasoning
capabilities. Agents call these; the router picks executive vs operations
models. Numbers always arrive pre-computed in `context`.
"""

from __future__ import annotations

from typing import Any

from ai.router import AI
from ai.types import AIResponse, Capability

_OPS_SYSTEM = (
    "You are an operations-level supply chain analyst. You receive the "
    "structured output of deterministic engines (EOQ, ROP, safety stock, "
    "ABC/XYZ, carrier scores). Reason over those exact numbers — never "
    "recompute or invent figures. Give a crisp, decision-oriented answer.")

_EXEC_SYSTEM = (
    "You are a Chief Supply Chain Officer's copilot. You synthesise the "
    "outputs of multiple specialist agents into a board-ready narrative. "
    "Cite the specific numbers provided, call out the top risks and the "
    "single most important action. Never invent metrics.")


def operations(task: str, context: Any) -> AIResponse:
    """Day-to-day agent reasoning (Inventory, Warehouse, Procurement, ...)."""
    return AI.ask(Capability.REASONING_OPERATIONS, task, context, system=_OPS_SYSTEM)


def executive(task: str, context: Any) -> AIResponse:
    """Strategic synthesis / CEO & board reporting."""
    return AI.ask(Capability.REASONING_EXECUTIVE, task, context, system=_EXEC_SYSTEM)
