"""
planner/aggregator.py — merges every capability's result into ONE structured
Decision. It is domain-agnostic: it reads the normalised TaskResult fields
(summary, findings, metrics, impact_usd, confidence) uniformly, so a newly
registered capability is aggregated with no code change here.

The executive summary is composed by the AI Router (reasoning.executive) when
reachable, with a deterministic fallback so the Planner always returns a result.
"""

from __future__ import annotations

from planner.prompts import executive_synthesis
from planner.schemas import Capability, Decision, TaskResult


class Aggregator:
    def aggregate(self, objective: str, results: dict[str, TaskResult],
                  layers: list[list[str]], capabilities: list[Capability]) -> Decision:
        ok = [r for r in results.values() if r.ok]
        findings: list[str] = []
        evidence: list[str] = []
        kpis: list[dict] = []
        actions: list[dict] = []
        total_impact = 0.0

        for r in sorted(results.values(), key=lambda x: -(x.impact_usd or 0)):
            if r.error:
                continue
            if r.summary:
                findings.append(f"{_label(r.capability)}: {r.summary}")
                evidence.append(f"{_label(r.capability)} (confidence {r.confidence:.0%})")
            for f in r.findings[:2]:
                findings.append(f)
            for k, v in list(r.metrics.items())[:3]:
                kpis.append({"name": f"{_label(r.capability)} · {k.replace('_', ' ')}", "value": v})
            if r.impact_usd:
                total_impact += r.impact_usd
                actions.append({"action": r.summary or f"Action from {_label(r.capability)}",
                                "impact_usd": round(r.impact_usd, 0),
                                "confidence": round(r.confidence * 100, 0),
                                "capability": r.capability})

        # Confidence is bounded by the weakest executed capability (honest chaining).
        confidence = round((min((r.confidence for r in ok), default=0.4)) * 100, 0)

        risks = [f"{_label(r.capability)} unavailable — {r.error}"
                 for r in results.values() if r.error]
        if not actions:
            risks.append("No quantified financial action surfaced for this objective.")

        summary = self._summarise(objective, findings, total_impact)

        return Decision(
            objective=objective,
            executive_summary=summary,
            key_findings=findings[:8],
            recommended_actions=actions[:6] or [{"action": "Review specialist findings", "impact_usd": 0, "confidence": confidence}],
            financial_impact={"identified_usd": round(total_impact, 0),
                              "actions": len(actions)},
            operational_impact=self._operational(results),
            risks=risks or ["No blocking risks detected."],
            confidence=confidence,
            evidence=evidence,
            kpis=kpis[:10],
            assumptions=["Figures combine live engine output with modelled cost-to-serve where labelled.",
                         "Capability confidences are transparent heuristics, not calibrated probabilities."],
            next_steps=[a["action"] for a in actions[:3]] or ["Refine the objective and re-plan."],
            capabilities=[c.name for c in capabilities],
            graph=layers,
            tasks=[r.to_dict() for r in results.values()],
        )

    def _operational(self, results: dict[str, TaskResult]) -> dict:
        op = {}
        for r in results.values():
            for k in ("on_time", "service", "improvement_pct", "avg_utilization",
                      "alloc_saving_pct", "net_margin_pct"):
                if k in r.metrics:
                    op[f"{r.capability}.{k}"] = r.metrics[k]
        return op

    def _summarise(self, objective: str, findings: list[str], impact: float) -> str:
        try:
            from ai import AI
            resp = AI.ask("reasoning.executive", task="planner_synthesis",
                          user=executive_synthesis(objective, findings, impact))
            text = getattr(resp, "text", None)
            if text and len(text) > 40:
                return text.strip()
        except Exception:  # noqa: BLE001
            pass
        # Deterministic fallback
        lead = findings[0] if findings else "specialists returned no material signal"
        return (f"To {objective.rstrip('.').lower()}: coordinated {len(findings)} specialist "
                f"finding(s) into one plan. Lead signal — {lead.split(':', 1)[-1].strip()}. "
                f"A combined ${impact:,.0f} of financial impact is addressable; execute the "
                f"top recommended action first and measure against the KPIs below.")


def _label(name: str) -> str:
    return name.replace("_", " ").title()
