"use client";
/** Logistics Command Center — live shipment KPIs, real geo map, carrier scorecard. */
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, KpiCard, Badge, DataTable, Th, Td } from "@/components/ui/primitives";
import { api, LogisticsResponse, MapResponse } from "@/lib/api";

const LaneMap = dynamic(() => import("@/components/LaneMap"), {
  ssr: false,
  loading: () => <div className="h-[340px] grid place-items-center text-ink-3 text-[0.8125rem]">Loading map…</div>,
});

export default function Logistics() {
  const [data, setData] = useState<LogisticsResponse | null>(null);
  const [map, setMap] = useState<MapResponse | null>(null);
  useEffect(() => {
    api.logistics().then(setData).catch(() => setData(null));
    api.logisticsMap().then(setMap).catch(() => setMap(null));
  }, []);
  const k = data?.kpis;
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
          <CardHead title="Network hubs & routes" hint={map?.tiles_url ? "MapTiler · live geo" : "set MAPTILER key for basemap"} />
          <div className="p-2">
            {map ? <LaneMap data={map} /> : <div className="h-[340px] grid place-items-center text-ink-3 text-[0.8125rem]">Loading map…</div>}
          </div>
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
            {(!data?.carriers || data.carriers.length === 0) && (
              <tr><Td>Loading…</Td><Td> </Td><Td> </Td><Td> </Td></tr>
            )}
          </DataTable>
        </Card>
      </div>
    </AppShell>
  );
}
