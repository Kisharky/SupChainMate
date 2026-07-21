"use client";
/** Logistics Command Center — live shipment KPIs, real geo map, carrier scorecard. */
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, KpiCard, Badge, DataTable, Th, Td, Button, TableState } from "@/components/ui/primitives";
import { api, LogisticsResponse, MapResponse, OptimizeResponse } from "@/lib/api";

const LaneMap = dynamic(() => import("@/components/LaneMap"), {
  ssr: false,
  loading: () => <div className="h-[340px] grid place-items-center text-ink-3 text-[0.8125rem]">Loading map…</div>,
});

export default function Logistics() {
  const [data, setData] = useState<LogisticsResponse | null>(null);
  const [map, setMap] = useState<MapResponse | null>(null);
  const [opt, setOpt] = useState<OptimizeResponse | null>(null);
  const [optimizing, setOptimizing] = useState(false);
  useEffect(() => {
    api.logistics().then(setData).catch(() => setData(null));
    api.logisticsMap().then(setMap).catch(() => setMap(null));
  }, []);
  const k = data?.kpis;

  const optimize = async () => {
    if (opt) { setOpt(null); return; }  // toggle back to lanes view
    setOptimizing(true);
    try { setOpt(await api.optimizeRoute()); }
    finally { setOptimizing(false); }
  };
  const gradeStatus = (g: string) => (g === "A" || g === "B" ? "good" : g === "C" ? "warning" : "critical");

  return (
    <AppShell title="Logistics Command Center">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>
          {k?.in_transit?.toLocaleString() ?? "—"} shipments in transit · {data?.source === "live" ? "live control tower" : "loading"}
        </div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Logistics Command Center</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Shipments, hubs, and carrier performance from the live tracking pipeline.</p>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))" }}>
        <KpiCard label="In Transit" value={k?.in_transit?.toLocaleString() ?? "—"} status="info" seed={2} />
        <KpiCard label="Late Deliveries" value={k?.delayed?.toLocaleString() ?? "—"} status="warning" seed={5} />
        <KpiCard label="On-Time Rate" value={k?.on_time_rate ?? "—"} unit="%" status="good" seed={7} />
        <KpiCard label="Avg Freight/Shpmt" value={k?.avg_cost ?? "—"} prefix="$" status="good" seed={6} />
      </div>

      <div className="grid gap-4 mt-4 items-start" style={{ gridTemplateColumns: "1.5fr 1fr" }}>
        <Card>
          <CardHead title="Network hubs & routes"
            hint={opt ? `optimised · ${opt.solver}${opt.fell_back ? " (fallback)" : ""}` : (map?.tiles_url ? "MapTiler · live geo" : "set MAPTILER key")}
            right={<Button variant={opt ? "secondary" : "primary"} sm onClick={optimize} disabled={optimizing}>
              {optimizing ? "Optimising…" : opt ? "↺ Show lanes" : "⚡ Optimise routes"}
            </Button>} />
          <div className="p-2">
            {map ? <LaneMap data={map} tour={opt?.tour} /> : <div className="h-[340px] grid place-items-center text-ink-3 text-[0.8125rem]">Loading map…</div>}
          </div>
          {opt && opt.solved && (
            <div className="px-[18px] pb-3 grid grid-cols-3 gap-3">
              {[
                { l: "Optimised", v: `${Math.round(opt.objective).toLocaleString()} km` },
                { l: "vs naive", v: `${Math.round(opt.baseline).toLocaleString()} km` },
                { l: "Saved", v: `${opt.improvement_pct.toFixed(0)}%` },
              ].map((m) => (
                <div key={m.l} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                  <div className="eyebrow">{m.l}</div>
                  <div className="text-[1.15rem] font-bold tnum mt-0.5" style={{ color: m.l === "Saved" ? "var(--good)" : "var(--ink)" }}>{m.v}</div>
                </div>
              ))}
              <div className="col-span-3 text-[0.6875rem] text-ink-3">
                Solver: <b style={{ color: "var(--accent)" }}>{opt.solver}</b>
                {opt.status?.plan?.routing && <> · plan routes → {opt.status.plan.routing}{opt.fell_back && " → local fallback"}</>} · {opt.detail}
              </div>
            </div>
          )}
        </Card>
        <Card>
          <CardHead title="Carrier scorecard" hint="on-time · grade" />
          <DataTable head={<><Th>Carrier</Th><Th num>Shipments</Th><Th num>On-time</Th><Th>Grade</Th></>}>
            {(data?.carriers ?? []).map((c) => (
              <tr key={c.carrier}>
                <Td strong>{c.carrier}</Td>
                <Td num>{c.shipments.toLocaleString()}</Td>
                <Td num>{c.on_time == null ? "—" : `${c.on_time.toFixed(1)}%`}</Td>
                <Td><Badge status={gradeStatus(c.grade) as any}>{c.grade}</Badge></Td>
              </tr>
            ))}
            {(!data?.carriers || data.carriers.length === 0) && <TableState cols={4} kind="loading" />}
          </DataTable>
        </Card>
      </div>
    </AppShell>
  );
}
