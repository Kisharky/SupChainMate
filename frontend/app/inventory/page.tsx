"use client";
/** Inventory Intelligence — genuinely live engine output. */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, DataTable, Th, Td, Badge, Button, TableState } from "@/components/ui/primitives";
import { api, InventoryResponse } from "@/lib/api";

export default function Inventory() {
  const [data, setData] = useState<InventoryResponse | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => { api.inventory().then(setData).catch(() => setErr(true)); }, []);

  const rows = data?.rows ?? [];
  const abcBadge = (abc: string) => abc === "A" ? "good" : abc === "B" ? "info" : "warning";

  return (
    <AppShell title="Inventory Intelligence">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-4">
        <div>
          <div className="text-[0.75rem] uppercase tracking-[.16em] font-semibold" style={{ color: "var(--accent)" }}>
            {rows.length} SKUs · shared decision engine {data?.source === "live" ? "· live" : ""}
          </div>
          <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Inventory Intelligence</h1>
          <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Reorder points, EOQ, and safety stock computed per SKU from real demand profiles.</p>
        </div>
        <Button variant="primary" sm>✓ Approve reorders</Button>
      </div>

      <Card>
        <CardHead title="Stock plan & reorder recommendations"
          hint={data ? `${(data.kpis as any)?.n_skus ?? rows.length} SKUs · ${(data.kpis as any)?.a_class ?? "—"} A-class` : "loading…"} />
        <DataTable head={<>
          <Th>SKU</Th><Th>Class</Th><Th num>Reorder point</Th><Th num>EOQ</Th>
          <Th num>Safety stock</Th><Th>Service</Th><Th num>Est. savings/yr</Th>
        </>}>
          {err && <TableState cols={7} kind="error" />}
          {rows.map((r) => (
            <tr key={r.sku} className="hover:bg-[color-mix(in_srgb,var(--accent)_6%,transparent)]">
              <Td strong><span className="font-mono">{r.sku}</span></Td>
              <Td><Badge status={abcBadge(r.abc) as any}>{r.abc}</Badge></Td>
              <Td num>{r.reorder_point.toLocaleString()}</Td>
              <Td num>{r.eoq.toLocaleString()}</Td>
              <Td num>{r.safety_stock.toLocaleString()}</Td>
              <Td>{r.service_level}</Td>
              <Td num strong>${Math.round(r.savings_yr).toLocaleString()}</Td>
            </tr>
          ))}
          {!err && rows.length === 0 && <TableState cols={7} kind="loading" />}
        </DataTable>
      </Card>

      {/* Multi-DC allocation via the optimization skill (real haversine costs) */}
      {data?.allocation?.solved && (
        <Card className="mt-4">
          <CardHead title="Multi-DC replenishment allocation"
            hint={`optimizer · ${data.allocation.solver}${data.allocation.fell_back ? " (fallback)" : ""}`} />
          <div className="p-[18px]">
            <div className="grid gap-3 mb-3" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))" }}>
              {[
                { l: "Optimised (km·units)", v: Math.round(data.allocation.objective).toLocaleString() },
                { l: "vs naive baseline", v: Math.round(data.allocation.baseline).toLocaleString() },
                { l: "Saved", v: `${data.allocation.improvement_pct.toFixed(0)}%`, good: true },
              ].map((m) => (
                <div key={m.l} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                  <div className="eyebrow">{m.l}</div>
                  <div className="text-[1.15rem] font-bold tnum mt-0.5" style={{ color: m.good ? "var(--good)" : "var(--ink)" }}>{m.v}</div>
                </div>
              ))}
            </div>
            <DataTable head={<><Th>Distribution centre</Th><Th>Serves region</Th><Th num>Units</Th><Th num>Transport (km·u)</Th></>}>
              {data.allocation.assignments.map((a, i) => (
                <tr key={i}>
                  <Td strong>{a.source}</Td>
                  <Td><Badge status="info">{a.sink}</Badge></Td>
                  <Td num>{Math.round(a.units).toLocaleString()}</Td>
                  <Td num>{Math.round(a.cost).toLocaleString()}</Td>
                </tr>
              ))}
            </DataTable>
            <div className="text-[0.6875rem] text-ink-3 mt-2">
              Sources = 3 largest hubs (DCs); sinks = regional demand ∝ customer count; cost = real Haversine distance. Solver: <b style={{ color: "var(--accent)" }}>{data.allocation.solver}</b>
              {data.allocation.status?.plan?.allocation && <> · plan allocation → {data.allocation.status.plan.allocation}</>}.
            </div>
          </div>
        </Card>
      )}
    </AppShell>
  );
}
