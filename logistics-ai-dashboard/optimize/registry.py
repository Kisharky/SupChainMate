"""
optimize/registry.py — problem-kind → solver plan. Swapping a solver (or adding
a new backend) is a one-line change here, exactly like the AI capability registry.
"""

from __future__ import annotations

from typing import Optional

from optimize.types import ProblemKind, SolverSpec

# Primary solver per problem kind, with a graceful fallback. cuOpt is the
# GPU-accelerated primary for routing; the local heuristic always works.
_DEFAULT_SPECS: dict[ProblemKind, SolverSpec] = {
    ProblemKind.ROUTING: SolverSpec(ProblemKind.ROUTING, solver="cuopt", fallback="local"),
    ProblemKind.ALLOCATION: SolverSpec(ProblemKind.ALLOCATION, solver="local", fallback=None),
}


class SolverRegistry:
    def __init__(self, specs: Optional[dict[ProblemKind, SolverSpec]] = None) -> None:
        self._specs = dict(specs) if specs is not None else dict(_DEFAULT_SPECS)

    def resolve(self, kind: ProblemKind | str) -> Optional[SolverSpec]:
        if isinstance(kind, str):
            kind = ProblemKind(kind)
        return self._specs.get(kind)

    def register(self, spec: SolverSpec) -> "SolverRegistry":
        self._specs[spec.kind] = spec
        return self

    def plan(self) -> dict[str, str]:
        return {k.value: s.solver for k, s in self._specs.items()}
