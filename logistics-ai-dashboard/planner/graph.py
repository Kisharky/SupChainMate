"""
planner/graph.py — turns a set of capabilities into a dependency-ordered
execution graph. Capabilities in the same layer are independent and may run
concurrently; later layers wait for the layers they depend on.
"""

from __future__ import annotations

from planner.schemas import Capability


class ExecutionGraph:
    def __init__(self, capabilities: list[Capability]) -> None:
        self._caps = {c.name: c for c in capabilities}

    def build(self) -> list[list[str]]:
        """Kahn's algorithm → ordered layers. Dependencies outside the selected
        set are ignored (already resolved by the registry). Cycles raise."""
        names = set(self._caps)
        indeg = {n: 0 for n in names}
        adj: dict[str, list[str]] = {n: [] for n in names}
        for n, cap in self._caps.items():
            for dep in cap.dependencies:
                if dep in names:
                    adj[dep].append(n)
                    indeg[n] += 1

        layers: list[list[str]] = []
        ready = sorted([n for n in names if indeg[n] == 0],
                       key=lambda n: self._caps[n].priority)
        seen = 0
        while ready:
            layers.append(ready)
            nxt: list[str] = []
            for n in ready:
                seen += 1
                for m in adj[n]:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        nxt.append(m)
            ready = sorted(nxt, key=lambda n: self._caps[n].priority)
        if seen != len(names):
            raise ValueError("Execution graph contains a cycle")
        return layers
