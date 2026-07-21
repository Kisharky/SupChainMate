"""
planner/executor.py — runs the execution graph layer by layer. Within a layer,
independent capabilities run concurrently on a thread pool (the domain services
are synchronous). Each handler receives the accumulated context: the base inputs
plus every upstream capability's outputs.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from planner.registry import CapabilityRegistry
from planner.schemas import Capability, TaskResult


class Executor:
    def __init__(self, registry: CapabilityRegistry, max_workers: int = 6) -> None:
        self.registry = registry
        self.max_workers = max_workers

    def _run_one(self, cap: Capability, context: dict) -> TaskResult:
        t0 = time.time()
        try:
            out = cap.handler(context) or {}
            return TaskResult(
                capability=cap.name, ok=bool(out.get("ok", True)),
                summary=str(out.get("summary", "")),
                findings=list(out.get("findings", [])),
                metrics=dict(out.get("metrics", {})),
                impact_usd=out.get("impact_usd"),
                confidence=float(out.get("confidence", cap.confidence)),
                duration_ms=int((time.time() - t0) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 — one capability must never crash the plan
            return TaskResult(capability=cap.name, ok=False, confidence=0.2,
                              duration_ms=int((time.time() - t0) * 1000),
                              error=f"{type(exc).__name__}: {exc}")

    def run(self, layers: list[list[str]], base_context: Optional[dict] = None) -> dict[str, TaskResult]:
        context = dict(base_context or {})
        results: dict[str, TaskResult] = {}
        for layer in layers:
            caps = [self.registry.get(n) for n in layer if self.registry.get(n)]
            if len(caps) == 1:
                res = [self._run_one(caps[0], context)]
            else:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(caps))) as pool:
                    res = list(pool.map(lambda c: self._run_one(c, context), caps))
            for r in res:
                results[r.capability] = r
                # publish this capability's outputs into the shared context
                context[r.capability] = {"metrics": r.metrics, "summary": r.summary,
                                         "impact_usd": r.impact_usd}
        return results
