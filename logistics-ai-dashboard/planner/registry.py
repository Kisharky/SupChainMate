"""
planner/registry.py — the dynamic capability registry. Capabilities register
themselves here; the Planner *discovers* what to run by matching a business
objective against capability metadata — never with hardcoded branches.

Mirrors ai/registry.py and optimize/registry.py.
"""

from __future__ import annotations

import re
from typing import Optional

from planner.schemas import Capability


class CapabilityRegistry:
    def __init__(self) -> None:
        self._caps: dict[str, Capability] = {}

    # ---- registration ----
    def register(self, cap: Capability) -> Capability:
        self._caps[cap.name] = cap
        return cap

    def get(self, name: str) -> Optional[Capability]:
        return self._caps.get(name)

    def all(self) -> list[Capability]:
        return list(self._caps.values())

    # ---- discovery ----
    def _score(self, cap: Capability, tokens: set[str]) -> int:
        bag = set(re.findall(r"[a-z]+", (cap.name + " " + cap.description + " "
                                         + " ".join(cap.keywords)).lower()))
        return len(tokens & bag)

    def select(self, objective: str, min_capabilities: int = 2) -> list[Capability]:
        """Discover the relevant capabilities for an objective, then pull in
        every transitive dependency so the graph is always executable."""
        tokens = set(re.findall(r"[a-z]+", (objective or "").lower()))
        scored = [(self._score(c, tokens), c) for c in self._caps.values()]
        chosen = {c.name: c for s, c in scored if s > 0}
        # Ensure a minimum breadth: add the highest-priority capabilities.
        if len(chosen) < min_capabilities:
            for c in sorted(self._caps.values(), key=lambda x: x.priority):
                chosen.setdefault(c.name, c)
                if len(chosen) >= min_capabilities:
                    break
        # Transitively include declared dependencies.
        frontier = list(chosen.values())
        while frontier:
            cap = frontier.pop()
            for dep in cap.dependencies:
                if dep not in chosen and dep in self._caps:
                    chosen[dep] = self._caps[dep]
                    frontier.append(self._caps[dep])
        return list(chosen.values())

    def meta(self) -> list[dict]:
        return [c.to_meta() for c in self._caps.values()]
