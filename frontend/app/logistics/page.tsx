"use client";
/** Logistics Command Center — shipments, lanes, delays. */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, KpiCard, Badge } from "@/components/ui/primitives";
import { api, LogisticsResponse } from "@/lib/api";

export default function Logistics() {
  const [data, setData] = useState<LogisticsResponse | null>(null);
  useEffect(() => { api.logistics().then(setData).catch(() => setData(null)); }, []);
  const k = data?.kpis;

  return (
    <AppShell title="Logistics Command Center">
      <div className="mb-4">
        <div className="text-[0.75rem] uppercase tracking-[.16em] font-semibold" style={{ color: "var(--accent)" }}>
          {k?.in_transit ?? "—"} shipments in transit
        </div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Logistics Command Center</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Active lanes and delays, with re-route options from the logistics agent.</p>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))" }}>
        <KpiCard label="In Transit" value={k?.in_transit ?? "—"} status="info" seed={2} />
        <KpiCard label="Delayed" value={k?.delayed ?? "—"} status="warning" seed={5} delta={1} />
        <KpiCard label="On-Time Rate" value={k?.on_time_rate ?? "—"} unit="%" status="good" seed={7} />
        <KpiCard label="Avg Freight" value={k?.avg_cost ?? "—"} unit="k" prefix="$" status="good" seed={6} />
      </div>

      <div className="grid gap-4 mt-4 items-start" style={{ gridTemplateColumns: "1.4fr 1fr" }}>
        <Card>
          <CardHead title="Active lanes" hint="origin → destination" />
          <div className="p-[18px] flex flex-col gap-2">
            {(data?.lanes ?? []).map((l, i) => (
              <div key={i} className="flex items-center gap-3 rounded border p-2.5 text-[0.8125rem] bg-[var(--panel-2)]" style={{ borderColor: "var(--hairline)" }}>
                <Badge status={l.status}>{l.status === "good" ? "On time" : "Delayed"}</Badge>
                <span className="font-mono text-ink">{l.from}</span>
                <span className="text-ink-3">→</span>
                <span className="font-mono text-ink">{l.to}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <CardHead title="Delayed shipments" />
          <div className="px-[18px] py-1.5">
            {(data?.delayed ?? []).map((d) => (
              <div key={d.id} className="flex gap-3 py-2.5 border-b last:border-0" style={{ borderColor: "var(--hairline)" }}>
                <div className="grid h-6 w-6 place-items-center rounded-md text-xs flex-none" style={{ background: "var(--warning-bg)", color: "var(--warning)" }}>◎</div>
                <div className="min-w-0">
                  <div className="text-[0.8125rem] text-ink font-semibold font-mono">{d.id} · {d.lane}</div>
                  <div className="text-[0.75rem] text-ink-3">{d.reason}</div>
                </div>
                <span className="ml-auto text-[10.5px] text-ink-3 whitespace-nowrap">ETA {d.eta_slip}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
