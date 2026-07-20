"""
ai/safety.py
Safety / guardrail service abstraction. A hook every outward-facing
generation can pass through before it reaches a user. Ships with a
deterministic check (PII-ish patterns, empty output) and can be pointed at
a dedicated safety model by wiring the SAFETY capability in the registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config
from ai.router import AI
from ai.types import Capability

_log = config.get_logger(__name__)

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                       # SSN-like
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),                     # card-like
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email
]


@dataclass
class SafetyVerdict:
    safe: bool
    reasons: list[str]


def check(text: str) -> SafetyVerdict:
    """Deterministic guardrail. If a SAFETY model is wired, escalate borderline
    cases to it; otherwise the deterministic verdict stands."""
    reasons: list[str] = []
    if not text or not text.strip():
        return SafetyVerdict(safe=False, reasons=["empty output"])
    for pat in _PII_PATTERNS:
        if pat.search(text):
            reasons.append("possible PII in output")
            break
    verdict = SafetyVerdict(safe=not reasons, reasons=reasons)

    if reasons and AI.status().get("safety", False):
        resp = AI.ask(Capability.SAFETY, "safety_review", context=None,
                      user=f"Is this safe to show a business user? Answer SAFE or "
                           f"UNSAFE with one reason.\n\n{text[:1500]}")
        if resp.ok and resp.text.upper().startswith("SAFE"):
            return SafetyVerdict(safe=True, reasons=["model-cleared"])
    return verdict
