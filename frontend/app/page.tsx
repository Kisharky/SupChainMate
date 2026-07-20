"use client";
/** Executive Control Tower — the landing screen. */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, KpiCard, Button, Badge, Alert } from "@/components/ui/primitives";
import { api, Kpi, WorkflowRun } from "@/lib/api";

const KPI_ORDER: { key: string; label: string; seed: number }[] = [
  { key: "supply_chain_health", label: "Supply Chain Health", seed: 7 },
  { key: "todays_risks", label: "Today's Risks", seed: 2 },
  { key: "late_shipments", label: "Late Shipments", seed: 5 },
  { key: "inventory_value", label: "Inventory Value", seed: 3 },
  { key: "forecast_accuracy", label: "Forecast Accuracy", seed: 9 },
  { key: "supplier_health", label: "Supplier Health", seed: 6 },
];

export default function ControlTower() {
  const [kpis, setKpis] = useState<Record<string, Kpi> | null>(null);
  const [run, setRun] = useState<WorkflowRun | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => { api.kpis().then((r) => setKpis(r.kpis)).catch(() => setKpis(null)); }, []);

  const runAgents = async () => {
    setRunning(true);
    try { setRun(await api.runWorkflow("full_control_tower", false)); }
    finally { setRunning(false); }
  };

  return (
    <AppShell title="Executive Control Tower">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-1">
        <div>
          <div className="text-[0.75rem] uppercase tracking-[.16em] font-semibold" style={{ color: "var(--accent)" }}>Live · decision intelligence</div>
          <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Executive Control Tower</h1>
          <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Network-wide health, risk, and the actions your AI agents recommend right now.</p>
        </div>
        <Button variant="secondary" sm>⇩ Export brief</Button>
      </div>

      <div className="grid gap-4 mt-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(190px,1fr))" }}>
        {KPI_ORDER.map(({ key, label, seed }) => {
          const k = kpis?.[key];
          return (
            <KpiCard key={key} label={label} seed={seed}
              value={k ? k.value : "—"} unit={k?.unit} prefix={k?.prefix}
              delta={k?.delta} status={k?.status ?? "good"} />
          );
        })}
      </div>

      <div className="grid gap-4 mt-4 items-start" style={{ gridTemplateColumns: "1.5fr 1fr" }}>
        {/* AI Executive Summary */}
        <Card className="p-[18px]" style={{ background: "linear-gradient(180deg,color-mix(in srgb,var(--accent) 8%,var(--panel)),var(--panel))", borderColor: "color-mix(in srgb,var(--accent) 28%,var(--hairline))" }}>
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-2 text-[0.75rem] font-bold uppercase tracking-wider" style={{ color: "var(--accent)" }}>
              <span className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ background: "var(--accent)" }} /> AI Executive Summary
            </span>
            <Badge status="neutral">Executive agent · 8 specialists</Badge>
          </div>
          <p className="text-[1.125rem] leading-snug my-3 text-ink">
            Demand is <b>rising in Victoria</b> while two upstream suppliers have <b>slipped their lead times</b>.
            Stockout exposure is concentrated in the controller-board line. I&apos;ve prepared actions to hold
            service level above 98%.
          </p>
          {run && run.results.length > 0 ? (
            <div className="my-3 flex flex-col gap-1.5">
              {run.results.slice(0, 5).map((r) => (
                <div key={r.agent} className="flex items-center gap-2 text-[0.8125rem] text-ink-2 border-b border-dashed py-1.5" style={{ borderColor: "var(--hairline)" }}>
                  <span style={{ color: "var(--accent)" }}>→</span>
                  <span className="text-ink font-semibold uppercase">{r.agent.replace(/_/g, " ")}</span>
                  <span className="ml-auto tnum text-ink-3">{r.confidence}% · {r.duration_ms}ms</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-[0.8125rem] text-ink-3 my-3">Run the control-tower workflow to synthesise live recommendations from all specialists.</p>
          )}
          <div className="flex gap-2.5 flex-wrap">
            <Button variant="primary" onClick={runAgents} disabled={running}>{running ? "Running…" : "✓ Approve Recommendations"}</Button>
            <Button variant="secondary" onClick={runAgents} disabled={running}>Review Details</Button>
          </div>
        </Card>

        {/* Priority risks */}
        <Card>
          <CardHead title="Priority risks" hint="last 24h" />
          <div className="px-[18px] py-1.5">
            {[
              { s: "critical", t: "Stockout risk · SKM-9931", m: "2 days cover · controller board", time: "now" },
              { s: "warning", t: "Port congestion · Ningbo", m: "3 shipments delayed ~36h", time: "18m" },
              { s: "warning", t: "Supplier slip · Anhui Cell Co.", m: "Lead time 12→16 days", time: "1h" },
              { s: "info", t: "Demand surge · Victoria", m: "+11% wk/wk, 3 categories", time: "2h" },
            ].map((f) => (
              <div key={f.t} className="flex gap-3 py-2.5 border-b last:border-0" style={{ borderColor: "var(--hairline)" }}>
                <div className="grid h-6 w-6 place-items-center rounded-md text-xs flex-none"
                  style={{ background: `var(--${f.s}-bg)`, color: `var(--${f.s})` }}>●</div>
                <div className="min-w-0">
                  <div className="text-[0.8125rem] text-ink font-semibold">{f.t}</div>
                  <div className="text-[0.75rem] text-ink-3">{f.m}</div>
                </div>
                <span className="ml-auto text-[10.5px] text-ink-3 whitespace-nowrap">{f.time}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {kpis && (
        <p className="mt-4 text-[0.75rem] text-ink-3">
          Headline KPIs are a representative board-level snapshot; the Inventory screen shows genuinely
          computed engine output. Agent runs above execute the real 9-agent orchestrator.
        </p>
      )}
    </AppShell>
  );
}
