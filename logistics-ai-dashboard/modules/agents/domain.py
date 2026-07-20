"""
modules/agents/domain.py
The domain agents. Each wraps existing deterministic engines — nothing is
rebuilt — declares exactly the context it needs, and requests AI
*capabilities* (never a model) through the router.

Communication: downstream agents read upstream outputs via ctx.upstream
(e.g. Inventory reads the Demand agent's horizon; Procurement reads
Inventory's urgent SKUs; Executive reads everyone).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import config
from modules import carbon, ensemble, tender, trust
from modules.agents.base import AgentResult, BaseAgent, ScopedContext
from modules.trust import Impact

_log = config.get_logger(__name__)

CONCENTRATION_ALERT_PCT = 40.0
EFFICIENCY_ALERT = 60.0


class DemandForecastAgent(BaseAgent):
    name = "demand_forecast"
    objective = "Predict demand accurately and quantify forecast reliability."
    required_context = ["daily_df", "forecast_df", "days"]

    def run(self, ctx: ScopedContext) -> AgentResult:
        daily_df, forecast_df = ctx.get("daily_df"), ctx.get("forecast_df")
        days = int(ctx.get("days", 7))
        res = AgentResult(agent=self.name, objective=self.objective)

        tourney = ensemble.run_tournament(daily_df, forecast_df, horizon_days=days)
        avg_daily = float(daily_df["y"].mean())
        horizon = float(forecast_df["yhat"].tail(days).clip(lower=0).sum())
        growth = ((horizon / days - avg_daily) / avg_daily * 100) if avg_daily > 0 else 0.0

        res.findings.append(f"Forecast {horizon:,.0f} units over the next {days} days "
                            f"({growth:+.1f}% vs the historical daily average).")
        if tourney:
            res.findings.append(
                f"Model tournament: {tourney['champion']} wins at "
                f"{tourney['champion_mape']:.1f}% MAPE on a {tourney['holdout_days']}-day backtest"
                + (f", beating Prophet ({tourney['prophet_mape']:.1f}%)."
                   if tourney["prophet_mape"] and tourney["champion"] != "Prophet" else "."))
            mape = tourney["champion_mape"]
        else:
            res.findings.append("History too short for a backtest — using Prophet unvalidated.")
            mape = 60.0

        support = min(1.0, len(daily_df) / 365.0)
        strength = max(0.0, 1.0 - mape / 60.0)
        res.confidence = float(np.clip(30 + 40 * support + 30 * strength, 20, 95))
        res.confidence_basis = (f"{len(daily_df)} days of history; champion MAPE {mape:.1f}% "
                                f"on a real holdout")
        res.impact = Impact(other=f"forecast error ±{mape:.0f}% (MAPE)")
        res.outputs = {"horizon_demand": horizon, "growth_pct": round(growth, 1),
                       "champion": tourney["champion"] if tourney else "Prophet",
                       "champion_mape": mape, "horizon_days": days}
        return res


class InventoryAgent(BaseAgent):
    name = "inventory"
    objective = "Hold exactly enough stock: set SS/ROP/EOQ and flag urgent reorders."
    required_context = ["demand_profile", "decision_outputs", "sku_plan",
                        "service_level", "history_days", "avg_lead_time"]
    depends_on = ["demand_forecast"]

    def run(self, ctx: ScopedContext) -> AgentResult:
        res = AgentResult(agent=self.name, objective=self.objective)
        outputs = ctx.get("decision_outputs")
        profile = ctx.get("demand_profile")
        sl = float(ctx.get("service_level", 0.95))

        recs = trust.from_decision_engine(profile, outputs,
                                          history_days=int(ctx.get("history_days", 0)),
                                          service_level=sl)
        sku_recs = trust.from_sku_engine(ctx.get("sku_plan"),
                                         avg_lead_time_days=float(ctx.get("avg_lead_time", 7)))
        res.recommendations = recs + sku_recs

        demand_up = ctx.upstream.get("demand_forecast", {})
        if demand_up.get("growth_pct", 0) > 10:
            res.findings.append(
                f"Demand agent forecasts {demand_up['growth_pct']:+.1f}% growth — "
                f"buffers below are the floor, not the ceiling.")
        res.findings.append(
            f"Policy: SS {outputs.safety_stock:,.0f} / ROP {outputs.reorder_point:,.0f} / "
            f"EOQ {outputs.eoq:,.0f} at {sl*100:.0f}% service (Z={outputs.z_value}).")
        if sku_recs:
            res.findings.append(f"{len(sku_recs)} SKU(s) at/below reorder point — "
                                f"routed for approval.")

        res.confidence = recs[0].confidence if recs else 40.0
        res.confidence_basis = recs[0].confidence_basis if recs else "no engine output"
        res.impact = Impact(
            cost_savings_yr=float(outputs.savings_vs_current),
            stockout_risk_pct=round((1 - sl) * 100, 1),
            service_level_pct=round(sl * 100, 1))
        res.outputs = {"safety_stock": float(outputs.safety_stock),
                       "reorder_point": float(outputs.reorder_point),
                       "eoq": float(outputs.eoq),
                       "urgent_skus": [r.title for r in sku_recs]}
        return res


class ProcurementAgent(BaseAgent):
    name = "procurement"
    objective = "Buy well: purchase-order plan, tender readiness, rate opportunities."
    required_context = ["shipments", "scorecard", "audit"]
    depends_on = ["inventory"]

    def run(self, ctx: ScopedContext) -> AgentResult:
        res = AgentResult(agent=self.name, objective=self.objective)
        inv = ctx.upstream.get("inventory", {})
        urgent = inv.get("urgent_skus", [])
        if urgent:
            res.findings.append(
                f"Inventory agent flagged {len(urgent)} urgent reorder(s) — "
                f"POs should be cut this cycle: {', '.join(urgent[:3])}"
                + ("…" if len(urgent) > 3 else "."))

        pack = tender.build_tender_pack(ctx.get("shipments"), ctx.get("scorecard"))
        retender = 0.0
        audit = ctx.get("audit")
        if audit:
            retender = float(audit["kpis"]["retender_opportunity"])
        if pack:
            s = pack["stats"]
            res.findings.append(
                f"Tender-ready: {s['total_shipments']:,} shipments of volume history "
                f"({s['period']}), peak {s['peak_shipments']:,}/month.")
        if retender > 0:
            res.findings.append(f"${retender:,.0f} of freight spend sits above the "
                                f"network-median rate — re-tender leverage.")

        support = min(1.0, (pack["stats"]["total_shipments"] / 10000) if pack else 0.0)
        strength = min(1.0, retender / max(float(audit["kpis"]["total_spend"]), 1) * 10) if audit else 0.3
        res.confidence = float(np.clip(30 + 40 * support + 30 * strength, 20, 95))
        res.confidence_basis = ("volume history depth + re-tender share of spend"
                                if pack else "no order-date history")
        res.impact = Impact(cost_savings_yr=retender if retender > 0 else None)
        res.outputs = {"po_lines": len(urgent), "retender_opportunity": retender,
                       "tender_ready": pack is not None}
        return res


class LogisticsAgent(BaseAgent):
    name = "logistics"
    objective = "Move goods on time: shipment health, carrier performance, billing."
    required_context = ["shipments", "kpis", "scorecard", "audit"]

    def run(self, ctx: ScopedContext) -> AgentResult:
        res = AgentResult(agent=self.name, objective=self.objective)
        kpis = ctx.get("kpis") or {}
        scorecard = ctx.get("scorecard")
        audit = ctx.get("audit")

        on_time = kpis.get("on_time_pct")
        if on_time is not None and not pd.isna(on_time):
            res.findings.append(f"On-time delivery {on_time:.1f}% across "
                                f"{kpis.get('total', 0):,} shipments; "
                                f"{kpis.get('late', 0):,} late, {kpis.get('at_risk', 0):,} at risk.")
        res.recommendations = trust.from_carrier_scorecard(
            scorecard, (audit or {}).get("by_carrier")) + trust.from_cost_audit(audit)
        for r in res.recommendations:
            res.findings.append(f"Recommends: {r.title}.")

        n = kpis.get("total", 0)
        support = min(1.0, n / 10000)
        strength = (on_time / 100) if on_time is not None and not pd.isna(on_time) else 0.3
        res.confidence = float(np.clip(30 + 40 * support + 30 * strength, 20, 95))
        res.confidence_basis = f"{n:,} shipments with promised-vs-actual dates"
        worst = (scorecard.iloc[-1]["Carrier"]
                 if scorecard is not None and len(scorecard) else None)
        res.impact = Impact(
            service_level_pct=round(float(on_time), 1) if on_time is not None and not pd.isna(on_time) else None,
            cost_savings_yr=sum(r.impact.cost_savings_yr or 0 for r in res.recommendations) or None)
        res.outputs = {"on_time_pct": on_time, "late": kpis.get("late", 0),
                       "at_risk": kpis.get("at_risk", 0), "worst_carrier": worst}
        return res


class SupplierRiskAgent(BaseAgent):
    name = "supplier_risk"
    objective = "See risk before it bites: reliability variance and concentration."
    required_context = ["shipments", "scorecard"]

    def run(self, ctx: ScopedContext) -> AgentResult:
        res = AgentResult(agent=self.name, objective=self.objective)
        ships: Optional[pd.DataFrame] = ctx.get("shipments")
        if ships is None or "carrier" not in ships.columns or ships["carrier"].isna().all():
            res.findings.append("No carrier data — risk profile unavailable.")
            res.confidence, res.confidence_basis = 20.0, "no carrier column"
            return res

        share = ships["carrier"].value_counts(normalize=True)
        top_carrier, top_pct = share.index[0], float(share.iloc[0] * 100)
        hhi = float((share ** 2).sum() * 10000)
        res.findings.append(f"Volume concentration: {top_carrier} carries {top_pct:.0f}% "
                            f"of shipments (HHI {hhi:,.0f}).")

        var_by = None
        if "delay_days" in ships.columns and ships["delay_days"].notna().any():
            var_by = (ships.dropna(subset=["delay_days"])
                      .groupby("carrier")["delay_days"].std().sort_values(ascending=False))
            if len(var_by):
                res.findings.append(f"Least predictable: {var_by.index[0]} "
                                    f"(delivery σ {var_by.iloc[0]:.1f} days).")

        if top_pct > CONCENTRATION_ALERT_PCT:
            res.recommendations.append(trust.Recommendation(
                source="Supplier Risk", category="CONCENTRATION RISK",
                title=f"Reduce dependence on {top_carrier} ({top_pct:.0f}% of volume)",
                action=(f"Qualify a secondary carrier for {top_carrier}'s main lanes and "
                        f"move toward a ≤{CONCENTRATION_ALERT_PCT:.0f}% single-carrier cap."),
                drivers=[trust.Driver("Volume share", f"{top_pct:.0f}% with one carrier"),
                         trust.Driver("Concentration index", f"HHI {hhi:,.0f}")],
                confidence=float(np.clip(30 + 40 * min(1, len(ships) / 10000)
                                         + 30 * min(1, (top_pct - 40) / 40), 20, 95)),
                confidence_basis=f"{len(ships):,} shipments; share {top_pct:.0f}% vs "
                                 f"{CONCENTRATION_ALERT_PCT:.0f}% threshold",
                impact=Impact(other=f"single-point-of-failure exposure on {top_pct:.0f}% of volume"),
            ))
            res.findings.append("Concentration exceeds the risk threshold — "
                                "mitigation routed for approval.")

        res.confidence = float(np.clip(30 + 40 * min(1, len(ships) / 10000) + 30 * 0.7, 20, 95))
        res.confidence_basis = f"{len(ships):,} shipments across {len(share)} carriers"
        res.outputs = {"top_carrier": str(top_carrier), "concentration_pct": round(top_pct, 1),
                       "hhi": round(hhi), "most_variable": str(var_by.index[0]) if var_by is not None and len(var_by) else None}
        return res


class WarehouseAgent(BaseAgent):
    name = "warehouse"
    objective = "Run an efficient network: hub coverage and zone efficiency."
    required_context = ["centroid_stats", "n_clusters"]

    def run(self, ctx: ScopedContext) -> AgentResult:
        res = AgentResult(agent=self.name, objective=self.objective)
        stats = ctx.get("centroid_stats")
        if stats is None or "efficiency_score" not in getattr(stats, "columns", []):
            res.findings.append("No network cluster metrics available.")
            res.confidence, res.confidence_basis = 20.0, "no geo data"
            return res
        df = stats.reset_index() if "cluster" not in stats.columns else stats
        avg_eff = float(df["efficiency_score"].mean())
        worst = df.loc[df["efficiency_score"].idxmin()]
        res.findings.append(f"{int(ctx.get('n_clusters', len(df)))} hubs; average zone "
                            f"efficiency {avg_eff:.0f}%.")
        res.findings.append(f"Weakest zone: {int(worst['cluster'])} at "
                            f"{worst['efficiency_score']:.0f}% "
                            f"(avg distance {worst['avg_dist_km']:.0f} km).")
        if avg_eff < EFFICIENCY_ALERT:
            res.findings.append("Below the efficiency threshold — test alternative hub counts.")
        support = min(1.0, float(df.get("customers", pd.Series([0])).sum()) / 50000)
        res.confidence = float(np.clip(30 + 40 * support + 30 * (avg_eff / 100), 20, 95))
        res.confidence_basis = f"{int(df.get('customers', pd.Series([0])).sum()):,} customer locations clustered"
        res.impact = Impact(other=f"network efficiency {avg_eff:.0f}%")
        res.outputs = {"avg_efficiency": round(avg_eff, 1), "worst_zone": int(worst["cluster"])}
        return res


class SustainabilityAgent(BaseAgent):
    name = "sustainability"
    objective = "Cut freight emissions without cutting service."
    required_context = ["shipments", "centroid_stats", "scorecard", "shipment_weight_kg"]

    def run(self, ctx: ScopedContext) -> AgentResult:
        res = AgentResult(agent=self.name, objective=self.objective)
        avg_dist = carbon.network_avg_distance_km(ctx.get("centroid_stats"))
        if avg_dist is None:
            res.findings.append("No route distances — emissions unavailable.")
            res.confidence, res.confidence_basis = 20.0, "no network metrics"
            return res
        weight = float(ctx.get("shipment_weight_kg", 20.0))
        table = carbon.carrier_emissions(ctx.get("shipments"), avg_dist, weight,
                                         ctx.get("scorecard"))
        if table is None or not len(table):
            res.findings.append("No carrier data for emissions attribution.")
            res.confidence, res.confidence_basis = 25.0, "no carrier column"
            return res
        total_t = float(table["Total tCO2e"].sum())
        green, dirty = table.iloc[0], table.iloc[-1]
        res.findings.append(f"Network footprint ≈ {total_t:,.1f} tCO₂e "
                            f"(@{weight:.0f} kg/shipment, {avg_dist:,.0f} km avg).")
        for note in carbon.carbon_insights(table):
            res.findings.append(note)
        multi_mode = table["Mode"].nunique() > 1
        res.confidence = float(np.clip(30 + 40 * (1.0 if multi_mode else 0.3) + 30 * 0.6, 20, 95))
        res.confidence_basis = ("distance × weight × DEFRA-style mode factors; "
                                + ("per-carrier modes known" if multi_mode
                                   else "single mode assumed — add transport_mode data"))
        res.impact = Impact(other=f"{total_t:,.0f} tCO₂e network footprint")
        res.outputs = {"total_tco2e": round(total_t, 1),
                       "greenest": str(green["Carrier"]), "dirtiest": str(dirty["Carrier"])}
        return res


def _memory_deltas(memory: dict, upstream: dict[str, dict]) -> list[str]:
    """Run-over-run changes: compare current upstream outputs vs each
    agent's previous persisted run (numeric keys only)."""
    notes: list[str] = []
    for agent_name, current in upstream.items():
        prev = (memory.get(agent_name) or {}).get("outputs", {})
        for key, now_val in current.items():
            if not isinstance(now_val, (int, float)) or key not in prev:
                continue
            then_val = prev[key]
            if not isinstance(then_val, (int, float)) or then_val == now_val:
                continue
            notes.append(f"{agent_name}.{key}: {then_val:,.1f} → {now_val:,.1f}")
    return notes[:6]


class KnowledgeAgent(BaseAgent):
    name = "knowledge"
    objective = "Ground decisions in company policy, SOPs, and contracts."
    required_context = ["kpis", "scorecard"]

    # Standing themes the agent checks the knowledge base for, derived from
    # the operational picture. The KB is this agent's domain resource.
    _THEMES = {
        "carrier SLA / on-time performance policy": ("scorecard", "grade"),
        "late delivery and penalty policy": ("kpis", "late"),
        "reorder and safety stock policy": (None, None),
        "freight invoice payment terms": (None, None),
    }

    def run(self, ctx: ScopedContext) -> AgentResult:
        res = AgentResult(agent=self.name, objective=self.objective)
        from modules import knowledge
        stats = knowledge.kb_stats()
        if stats["documents"] == 0:
            res.findings.append("Knowledge base is empty — upload SOPs, policies, and "
                                "contracts so decisions can be grounded in them.")
            res.confidence, res.confidence_basis = 25.0, "no documents indexed"
            res.outputs = {"policies_found": 0, "documents": 0}
            return res

        found, covered = 0, []
        for theme in self._THEMES:
            hits = knowledge.retrieve(theme, top_k=1)
            if hits:
                found += 1
                covered.append(theme)
                res.findings.append(f"Policy found for '{theme}': "
                                    f"\"{hits[0]['text'][:120]}…\" (from {hits[0]['doc']}).")
        if not covered:
            res.findings.append(f"{stats['documents']} document(s) indexed, but none match "
                                f"the current operational themes — check coverage.")
        retr = stats.get("indexed_chunks", 0)
        res.confidence = float(min(90, 40 + found * 12 + min(stats["documents"], 5) * 2))
        res.confidence_basis = (f"{stats['documents']} docs / {stats['chunks']} chunks "
                                f"({retr} vector-indexed); {found}/{len(self._THEMES)} themes covered")
        res.outputs = {"policies_found": found, "documents": stats["documents"],
                       "themes_covered": covered}
        return res


class ExecutiveAgent(BaseAgent):
    name = "executive"
    objective = "Synthesize everything into one decision-ready brief."
    required_context = ["health", "memory"]
    depends_on = ["demand_forecast", "inventory", "procurement", "logistics",
                  "supplier_risk", "warehouse", "sustainability", "knowledge"]
    reasoning_capability = "reasoning.executive"   # router picks the executive model

    def run(self, ctx: ScopedContext) -> AgentResult:
        res = AgentResult(agent=self.name, objective=self.objective)
        up = ctx.upstream
        hc = ctx.get("health")

        if hc:
            res.findings.append(f"Network health {hc['grade']} ({hc['score']:.0f}/100).")
        d = up.get("demand_forecast", {})
        if d:
            res.findings.append(f"Demand: {d.get('horizon_demand', 0):,.0f} units next "
                                f"{d.get('horizon_days', 7)} days ({d.get('growth_pct', 0):+.1f}%), "
                                f"forecast by {d.get('champion', '?')} at "
                                f"{d.get('champion_mape', 0):.0f}% MAPE.")
        i = up.get("inventory", {})
        if i:
            res.findings.append(f"Inventory: ROP {i.get('reorder_point', 0):,.0f} / EOQ "
                                f"{i.get('eoq', 0):,.0f}; {len(i.get('urgent_skus', []))} urgent SKU(s).")
        l = up.get("logistics", {})
        if l and l.get("on_time_pct") is not None:
            res.findings.append(f"Logistics: {l['on_time_pct']:.1f}% on-time, "
                                f"{l.get('late', 0):,} late; worst carrier {l.get('worst_carrier')}.")
        s = up.get("supplier_risk", {})
        if s:
            res.findings.append(f"Risk: {s.get('top_carrier')} holds "
                                f"{s.get('concentration_pct', 0):.0f}% of volume "
                                f"(HHI {s.get('hhi', 0):,}).")
        p = up.get("procurement", {})
        if p and p.get("retender_opportunity"):
            res.findings.append(f"Procurement: ${p['retender_opportunity']:,.0f} re-tender "
                                f"leverage; {p.get('po_lines', 0)} PO line(s) pending.")
        w = up.get("warehouse", {})
        if w:
            res.findings.append(f"Warehouse: network efficiency {w.get('avg_efficiency', 0):.0f}%.")
        kn = up.get("knowledge", {})
        if kn:
            res.findings.append(f"Policy: {kn.get('policies_found', 0)} relevant "
                                f"policy area(s) grounded from {kn.get('documents', 0)} document(s).")
        su = up.get("sustainability", {})
        if su:
            res.findings.append(f"Sustainability: {su.get('total_tco2e', 0):,.1f} tCO₂e; "
                                f"greenest {su.get('greenest')}, highest {su.get('dirtiest')}.")

        # Memory: run-over-run movement (only when a previous run exists)
        deltas = _memory_deltas(ctx.get("memory") or {}, up)
        if deltas:
            res.findings.append("Since the last run: " + "; ".join(deltas) + ".")

        # Chain confidence honestly: an executive summary is only as reliable
        # as its weakest upstream input.
        confs = ctx.upstream_confidence
        res.outputs = {"brief": "\n".join(f"- {f}" for f in res.findings)}
        if confs:
            weakest = min(confs, key=confs.get)
            res.confidence = float(confs[weakest])
            res.confidence_basis = (f"bounded by the weakest upstream agent "
                                    f"({weakest}: {confs[weakest]:.0f}%)")
        else:
            res.confidence, res.confidence_basis = 50.0, "no upstream runs"
        return res
