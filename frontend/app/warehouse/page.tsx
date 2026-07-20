"use client";
/** Warehouse — network zones & utilisation from real geo clustering (live). */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, KpiCard, DataTable, Th, Td, Badge } from "@/components/ui/primitives";
import { api, WarehouseResponse } from "@/lib/api";

export default function Warehouse() {
  const [data, setData] = useState<WarehouseResponse | null>(null);
  useEffect(() => { api.warehouse().then(setData).catch(() => setData(null)); }, []);
  const utilStatus = (u: number) => (u >= 85 ? "critical" : u >= 60 ? "warning" : "good");

  return (
    <AppShell title="Warehouse">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>Network zones · {data?.source === "live" ? "live geo clustering" : "loading"}</div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Warehouse & Network</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Demand-weighted hub zones derived from real customer geolocation (KMeans).</p>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))" }}>
        <KpiCard label="Hub Zones" value={data?.hub_count ?? "—"} status="info" seed={2} />
        <KpiCard label="Avg Utilisation" value={data?.avg_utilization ?? "—"} unit="%" status={data ? utilStatus(data.avg_utilization) : "good"} seed={5} />
        <KpiCard label="Locations Served" value={data ? data.zones.reduce((a, z) => a + z.locations, 0).toLocaleString() : "—"} status="good" seed={7} />
      </div>

      <Card className="mt-4">
        <CardHead title="Zone utilisation" hint="demand-weighted share of network" />
        <DataTable head={<><Th>Zone</Th><Th num>Lat</Th><Th num>Lon</Th><Th num>Locations</Th><Th num>Utilisation</Th><Th>Status</Th></>}>
          {(data?.zones ?? []).map((z) => (
            <tr key={z.zone}>
              <Td strong>{z.zone}</Td>
              <Td num>{z.lat.toFixed(2)}</Td>
              <Td num>{z.lon.toFixed(2)}</Td>
              <Td num>{z.locations.toLocaleString()}</Td>
              <Td num>{z.utilization.toFixed(1)}%</Td>
              <Td><Badge status={utilStatus(z.utilization) as any}>{utilStatus(z.utilization) === "critical" ? "Hot" : utilStatus(z.utilization) === "warning" ? "Busy" : "Healthy"}</Badge></Td>
            </tr>
          ))}
          {(!data?.zones || data.zones.length === 0) && <tr><Td>Loading…</Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td></tr>}
        </DataTable>
      </Card>
    </AppShell>
  );
}
