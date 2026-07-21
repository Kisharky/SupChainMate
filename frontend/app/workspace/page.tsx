"use client";
/**
 * Decision Intelligence — the AI decision & scenario command centre.
 * Executive brief · What Changed Today · AI planner · courses of action ·
 * scenario simulator · Detect→…→Learn decision timeline.
 */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, Button, Badge } from "@/components/ui/primitives";
import {
  api, ExecBrief, WhatChanged, Timeline, PlanResponse, CoaResponse,
  ScenarioResponse, WorkspaceCatalog,
} from "@/lib/api";

const money = (n: number) => `$${Math.round(Math.abs(n)).toLocaleString()}`;
const sev = { high: "critical", medium: "warning", low: "info" } as const;
const riskColor = { low: "good", medium: "warning", high: "critical" } as const;

export default function Workspace() {
  const [brief, setBrief] = useState<ExecBrief | null>(null);
  const [changed, setChanged] = useState<WhatChanged | null>(null);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [cat, setCat] = useState<WorkspaceCatalog | null>(null);

  useEffect(() => {
    api.wsBrief().then(setBrief).catch(() => {});
    api.wsChanged().then(setChanged).catch(() => {});
    api.wsTimeline().then(setTimeline).catch(() => {});
    api.wsCatalog().then(setCat).catch(() => {});
  }, []);

  return (
    <AppShell title="Decision Intelligence">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>AI Decision &amp; Scenario Intelligence · command centre</div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Decision Intelligence</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Operational decisions, not reports — every issue yields courses of action, estimated outcomes, and an execution workflow.</p>
      </div>

      <ExecutiveBrief brief={brief} />
      <div className="grid gap-4 mt-4 items-start" style={{ gridTemplateColumns: "1.4fr 1fr" }}>
        <AiPlanner />
        <WhatChangedToday data={changed} />
      </div>
      <CoursesOfAction cat={cat} />
      <ScenarioSimulator cat={cat} />
      <DecisionTimeline data={timeline} />
    </AppShell>
  );
}

/* ═══ Executive Decision Brief ═══ */
function ExecutiveBrief({ brief }: { brief: ExecBrief | null }) {
  const fi = brief?.financial_impact;
  return (
    <Card className="p-[18px]" style={{ background: "linear-gradient(180deg,color-mix(in srgb,var(--accent) 7%,var(--panel)),var(--panel))", borderColor: "color-mix(in srgb,var(--accent) 26%,var(--hairline))" }}>
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="inline-flex items-center gap-2 text-[0.75rem] font-bold uppercase tracking-wider" style={{ color: "var(--accent)" }}>
          <span className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ background: "var(--accent)" }} /> Executive Decision Brief
        </span>
        <div className="flex gap-2 items-center">
          {brief && <Badge status="info">confidence {brief.confidence}%</Badge>}
          {brief && <Badge status={brief.awaiting_approval ? "warning" : "good"}>{brief.awaiting_approval} awaiting approval</Badge>}
        </div>
      </div>
      <p className="text-[1.0625rem] leading-snug my-3 text-ink">{brief?.summary ?? "Synthesising the operational picture…"}</p>

      <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))" }}>
        {[
          { l: "Health", v: brief ? `${brief.kpis.health}%` : "—", s: "good" },
          { l: "On-time", v: brief ? `${brief.kpis.on_time}%` : "—", s: "good" },
          { l: "At-risk exposure", v: fi ? money(fi.at_risk_usd) : "—", s: "critical" },
          { l: "Opportunity", v: fi ? money(fi.opportunity_usd) : "—", s: "good" },
          { l: "Net position", v: fi ? `${fi.net_usd < 0 ? "−" : "+"}${money(fi.net_usd)}` : "—", s: fi && fi.net_usd < 0 ? "critical" : "good" },
        ].map((m) => (
          <div key={m.l} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
            <div className="eyebrow">{m.l}</div>
            <div className="text-[1.25rem] font-bold tnum mt-0.5" style={{ color: `var(--${m.s})` }}>{m.v}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 mt-4" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <div>
          <div className="eyebrow mb-2">Operational risks</div>
          <div className="flex flex-col gap-2">
            {(brief?.risks ?? []).map((r, i) => (
              <div key={i} className="flex gap-2.5 items-start rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                <Badge status={sev[r.severity]}>{r.severity}</Badge>
                <div className="min-w-0">
                  <div className="text-[0.875rem] font-semibold">{r.title} <span className="text-ink-3 font-normal">· {r.area}</span></div>
                  <div className="text-[0.75rem] text-ink-3">{r.detail}</div>
                </div>
              </div>
            ))}
            {brief && brief.risks.length === 0 && <div className="text-ink-3 text-[0.8125rem]">No open risks.</div>}
          </div>
        </div>
        <div>
          <div className="eyebrow mb-2">Recommended decisions</div>
          <div className="flex flex-col gap-2">
            {(brief?.recommended ?? []).slice(0, 4).map((d, i) => (
              <div key={i} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                <div className="flex justify-between gap-2">
                  <div className="text-[0.875rem] font-semibold truncate">{d.title}</div>
                  <Badge status="good">{d.confidence}%</Badge>
                </div>
                <div className="text-[0.75rem] text-ink-3 mt-0.5">{d.impact_usd ? `${money(d.impact_usd)}/yr · ` : ""}{d.area}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ═══ AI Planner ═══ */
function AiPlanner() {
  const [q, setQ] = useState("Stockout risk on controller boards while demand spikes in Victoria");
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => { setLoading(true); try { setPlan(await api.wsPlan(q)); } finally { setLoading(false); } };

  return (
    <Card>
      <CardHead title="AI Planner" hint="decomposes a request into agent tasks" />
      <div className="p-[18px]">
        <div className="flex gap-2 items-center rounded border p-1.5 pl-3" style={{ borderColor: "var(--hairline-strong)", background: "var(--bg-sunken)" }}>
          <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()}
            className="flex-1 bg-transparent outline-none text-[0.875rem] text-ink" placeholder="Describe an operational situation…" />
          <Button variant="primary" sm onClick={run} disabled={loading}>{loading ? "Planning…" : "Plan"}</Button>
        </div>
        {plan && (
          <>
            <p className="text-[0.8125rem] text-ink-2 mt-3">{plan.narrative}</p>
            <div className="mt-3 flex flex-col gap-2">
              {plan.plan.map((s) => (
                <div key={s.step} className="flex gap-3 items-center">
                  <div className="grid place-items-center h-6 w-6 rounded-full flex-none text-[0.75rem] font-bold"
                    style={{ background: s.agent === "executive" ? "var(--accent)" : "var(--panel-2)", color: s.agent === "executive" ? "var(--accent-ink)" : "var(--text-2)", border: "1px solid var(--hairline)" }}>{s.step}</div>
                  <div className="flex-1 rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                    <div className="flex items-center gap-2">
                      <span className="text-[0.8125rem] font-semibold uppercase">{s.agent}</span>
                      {s.uses_optimizer && <Badge status="info">optimizer skill</Badge>}
                    </div>
                    <div className="text-[0.8125rem] text-ink-2 mt-0.5">{s.task}</div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
        {!plan && <p className="text-ink-3 text-[0.8125rem] mt-3">Enter a situation — the planner routes sub-decisions to Forecasting, Inventory, Procurement, Logistics, Commercial, and Knowledge agents, then the Executive synthesises.</p>}
      </div>
    </Card>
  );
}

/* ═══ What Changed Today ═══ */
function WhatChangedToday({ data }: { data: WhatChanged | null }) {
  const Section = ({ label, items, color }: { label: string; items: string[]; color: string }) => (
    <div>
      <div className="eyebrow mb-1" style={{ color }}>{label}</div>
      {items.map((t, i) => <div key={i} className="text-[0.8125rem] text-ink-2 py-0.5">· {t}</div>)}
    </div>
  );
  return (
    <Card>
      <CardHead title="What Changed Today" hint={data?.date} right={data ? <Badge status="good">{money(data.realized_savings)} realized</Badge> : undefined} />
      <div className="p-[18px] flex flex-col gap-3">
        {data ? (
          <>
            <Section label="Completed actions" items={data.completed} color="var(--good)" />
            <Section label="Newly detected risks" items={data.new_risks} color="var(--warning)" />
            <Section label="Operational changes" items={data.changes} color="var(--info)" />
            {data.unresolved.length > 0 && (
              <div>
                <div className="eyebrow mb-1" style={{ color: "var(--critical)" }}>Needs executive attention</div>
                {data.unresolved.map((u, i) => <div key={i} className="text-[0.8125rem] text-ink py-0.5">⚑ {u.title} <span className="text-ink-3">— {u.reason}</span></div>)}
              </div>
            )}
          </>
        ) : <p className="text-ink-3 text-[0.8125rem]">Loading briefing…</p>}
      </div>
    </Card>
  );
}

/* ═══ Courses of Action ═══ */
function CoursesOfAction({ cat }: { cat: WorkspaceCatalog | null }) {
  const [issue, setIssue] = useState("stockout");
  const [coa, setCoa] = useState<CoaResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const load = async (k: string) => { setIssue(k); setLoading(true); try { setCoa(await api.wsCoa(k)); } finally { setLoading(false); } };
  useEffect(() => { load("stockout"); }, []);

  return (
    <Card className="mt-4">
      <CardHead title="Courses of Action" hint="multiple options · compare before approving"
        right={<div className="flex gap-1.5 flex-wrap">{(cat?.issues ?? []).map((it) => (
          <button key={it.key} onClick={() => load(it.key)}>
            <Badge status={issue === it.key ? "info" : "neutral"}>{it.label}</Badge>
          </button>))}</div>} />
      <div className="p-[18px]">
        {loading && <p className="text-ink-3 text-[0.8125rem]">Generating options…</p>}
        {coa && !loading && (
          <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(260px,1fr))" }}>
            {coa.options.map((o) => {
              const rec = o.id === coa.recommended;
              return (
                <div key={o.id} className="rounded-lg border p-3.5 flex flex-col gap-2.5"
                  style={{ borderColor: rec ? "color-mix(in srgb,var(--accent) 45%,transparent)" : "var(--hairline)", background: rec ? "color-mix(in srgb,var(--accent) 7%,var(--panel-2))" : "var(--panel-2)" }}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="text-[0.9375rem] font-semibold">{o.name}</div>
                    {rec && <Badge status="good">recommended</Badge>}
                  </div>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[0.75rem]">
                    <Metric l="Cost" v={money(o.implementation_cost)} />
                    <Metric l="Savings/yr" v={money(o.expected_savings)} good />
                    <Metric l="ROI" v={`${o.roi}×`} good />
                    <Metric l="Risk" v={o.operational_risk} riskColor={riskColor[o.operational_risk]} />
                    <Metric l="Service" v={`${o.service_level_impact > 0 ? "+" : ""}${o.service_level_impact} pp`} />
                    <Metric l="Execute in" v={o.execution_time} />
                    <Metric l="Confidence" v={`${o.confidence}%`} />
                    <Metric l="Inventory" v={o.inventory_impact} />
                  </div>
                  <div className="text-[0.75rem] text-ink-2 border-t pt-2" style={{ borderColor: "var(--hairline)" }}>{o.business_outcome}</div>
                  <div className="text-[0.6875rem] text-ink-3">
                    <b>Evidence:</b> {o.evidence.join("; ")}{o.optimization !== "—" ? ` · ${o.optimization}` : ""}
                  </div>
                  <Button variant={rec ? "primary" : "secondary"} sm>{rec ? "✓ Approve this option" : "Approve"}</Button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Card>
  );
}
function Metric({ l, v, good, riskColor }: { l: string; v: string | number; good?: boolean; riskColor?: string }) {
  return (
    <div>
      <div className="text-ink-3">{l}</div>
      <div className="font-semibold tnum capitalize" style={{ color: riskColor ? `var(--${riskColor})` : good ? "var(--good)" : "var(--ink)" }}>{v}</div>
    </div>
  );
}

/* ═══ Scenario Simulator ═══ */
function ScenarioSimulator({ cat }: { cat: WorkspaceCatalog | null }) {
  const [kind, setKind] = useState("warehouse_shutdown");
  const [mag, setMag] = useState(0.6);
  const [res, setRes] = useState<ScenarioResponse | null>(null);
  const run = async (k: string, m: number) => { setKind(k); setMag(m); try { setRes(await api.wsScenario(k, m)); } catch {} };
  useEffect(() => { run("warehouse_shutdown", 0.6); }, []);

  return (
    <Card className="mt-4">
      <CardHead title="Scenario Simulator" hint="model a disruption · see impact & mitigations" />
      <div className="p-[18px]">
        <div className="flex gap-1.5 flex-wrap mb-3">
          {(cat?.scenarios ?? []).map((s) => (
            <button key={s.key} onClick={() => run(s.key, mag)}>
              <Badge status={kind === s.key ? "info" : "neutral"}>{s.label}</Badge>
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 mb-4">
          <span className="text-[0.75rem] text-ink-3">Severity</span>
          <input type="range" min={0.1} max={1} step={0.1} value={mag}
            onChange={(e) => run(kind, parseFloat(e.target.value))}
            className="flex-1 accent-emerald-500" style={{ accentColor: "var(--accent)" }} />
          <span className="text-[0.8125rem] font-semibold tnum w-10">{Math.round(mag * 100)}%</span>
        </div>
        {res && (
          <>
            <p className="text-[0.875rem] text-ink-2 mb-3">{res.narrative}</p>
            <div className="grid gap-3 mb-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))" }}>
              {[
                { l: "Financial", v: `${res.impact.financial_usd < 0 ? "−" : "+"}${money(res.impact.financial_usd)}`, s: res.impact.positive ? "good" : "critical" },
                { l: "Service level", v: `${res.impact.service_pp > 0 ? "+" : ""}${res.impact.service_pp} pp`, s: res.impact.service_pp < 0 ? "critical" : "good" },
                { l: "Logistics", v: `${res.impact.logistics_pp > 0 ? "+" : ""}${res.impact.logistics_pp} pp`, s: res.impact.logistics_pp < 0 ? "warning" : "good" },
                { l: "Inventory", v: `${res.impact.inventory_pct > 0 ? "+" : ""}${res.impact.inventory_pct}%`, s: "info" },
                { l: "Customers affected", v: res.impact.customers_affected.toLocaleString(), s: "warning" },
              ].map((m) => (
                <div key={m.l} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                  <div className="eyebrow">{m.l}</div>
                  <div className="text-[1.15rem] font-bold tnum mt-0.5" style={{ color: `var(--${m.s})` }}>{m.v}</div>
                </div>
              ))}
            </div>
            <div className="grid gap-4" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <div>
                <div className="eyebrow mb-2">Before → after</div>
                {[["Service", "service", "%"], ["Logistics", "logistics", "%"], ["Health", "health", ""]].map(([lbl, key, unit]) => (
                  <div key={key} className="flex justify-between text-[0.8125rem] py-1 border-b" style={{ borderColor: "var(--hairline)" }}>
                    <span className="text-ink-2">{lbl}</span>
                    <span className="tnum"><span className="text-ink-3">{res.before[key]}{unit}</span> → <b>{res.after[key]}{unit}</b></span>
                  </div>
                ))}
              </div>
              <div>
                <div className="eyebrow mb-2">Recommended mitigations</div>
                {res.mitigations.map((mi, i) => (
                  <div key={i} className="flex justify-between gap-2 text-[0.8125rem] py-1.5 border-b" style={{ borderColor: "var(--hairline)" }}>
                    <span className="text-ink">{mi.action}</span>
                    <span className="text-ink-3 whitespace-nowrap">{mi.effect} · {mi.cost}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </Card>
  );
}

/* ═══ Decision Timeline ═══ */
function DecisionTimeline({ data }: { data: Timeline | null }) {
  return (
    <Card className="mt-4">
      <CardHead title="Live Decision Timeline" hint="Detect → Analyse → Optimise → Recommend → Approve → Execute → Measure → Learn" />
      <div className="p-[18px]">
        <div className="flex gap-1.5 overflow-x-auto pb-3">
          {(data?.stages ?? []).map((s, i) => {
            const n = data?.counts[s] ?? 0;
            return (
              <div key={s} className="flex items-center gap-1.5 flex-none">
                <div className="rounded border px-2.5 py-1.5 text-center" style={{ borderColor: n ? "color-mix(in srgb,var(--accent) 40%,transparent)" : "var(--hairline)", background: n ? "color-mix(in srgb,var(--accent) 10%,var(--panel-2))" : "var(--panel-2)" }}>
                  <div className="text-[0.6875rem] text-ink-3 uppercase tracking-wide">{s}</div>
                  <div className="text-[1.05rem] font-bold tnum" style={{ color: n ? "var(--accent)" : "var(--text-3)" }}>{n}</div>
                </div>
                {i < (data?.stages.length ?? 0) - 1 && <span className="text-ink-3">→</span>}
              </div>
            );
          })}
        </div>
        <div className="mt-2 flex flex-col gap-1.5 max-h-[300px] overflow-y-auto">
          {(data?.items ?? []).map((it) => (
            <div key={it.id} className="flex gap-3 items-center py-2 border-b" style={{ borderColor: "var(--hairline)" }}>
              <Badge status="neutral">{it.stage}</Badge>
              <div className="min-w-0 flex-1">
                <div className="text-[0.8125rem] text-ink font-medium truncate">{it.title}</div>
                {it.outcome && <div className="text-[0.6875rem] text-ink-3">{it.outcome}</div>}
              </div>
              <span className="text-[0.75rem] tnum text-ink-3 whitespace-nowrap">{it.confidence}%{it.impact_usd ? ` · ${money(it.impact_usd)}` : ""}</span>
            </div>
          ))}
          {data && data.items.length === 0 && <p className="text-ink-3 text-[0.8125rem]">No decisions in flight.</p>}
        </div>
      </div>
    </Card>
  );
}
