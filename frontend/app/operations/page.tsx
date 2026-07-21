"use client";
/** Operations — network throughput & order-status mix (live). */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, KpiCard, EmptyState } from "@/components/ui/primitives";
import { api, OperationsResponse } from "@/lib/api";

const STATUS_COLOR: Record<string, string> = {
  Delivered: "var(--good)", Shipped: "var(--info)", Processing: "var(--warning)", Cancelled: "var(--critical)",
};

export default function Operations() {
  const [data, setData] = useState<OperationsResponse | null>(null);
  useEffect(() => { api.operations().then(setData).catch(() => setData(null)); }, []);
  const k = data?.kpis ?? {};
  const counts = data?.status_counts ?? {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;

  return (
    <AppShell title="Operations">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>Global transit control · {data?.source === "live" ? "live" : "loading"}</div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Operations Command Center</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Network throughput, lead time, and order-status mix across the delivered base.</p>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(190px,1fr))" }}>
        <KpiCard label="In Transit" value={(k.in_transit ?? 0).toLocaleString()} status="info" seed={2} />
        <KpiCard label="On-Time" value={k.on_time_pct ?? "—"} unit="%" status="good" seed={7} />
        <KpiCard label="Avg Lead Time" value={k.avg_lead_days ?? "—"} unit="d" status="warning" seed={5} />
        <KpiCard label="Orders Observed" value={(k.delivered_observed ?? 0).toLocaleString()} status="good" seed={6} />
      </div>

      <Card className="mt-4">
        <CardHead title="Order status mix" hint="share of the network" />
        <div className="p-[18px] flex flex-col gap-3">
          {Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([s, n]) => (
            <div key={s}>
              <div className="flex justify-between text-[0.8125rem] mb-1.5">
                <span>{s}</span><span className="tnum text-ink-2">{n.toLocaleString()} · {((n / total) * 100).toFixed(1)}%</span>
              </div>
              <div className="h-1.5 rounded overflow-hidden" style={{ background: "var(--hairline)" }}>
                <i className="block h-full rounded" style={{ width: `${(n / total) * 100}%`, background: STATUS_COLOR[s] ?? "var(--accent)" }} />
              </div>
            </div>
          ))}
          {Object.keys(counts).length === 0 && <EmptyState kind="loading" />}
        </div>
      </Card>
    </AppShell>
  );
}
