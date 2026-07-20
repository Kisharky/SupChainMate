"use client";
/** Inventory Intelligence — genuinely live engine output. */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, DataTable, Th, Td, Badge, Button } from "@/components/ui/primitives";
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
          {err && <tr><Td>—</Td><Td>{" "}</Td><Td>{" "}</Td><Td>{" "}</Td><Td>{" "}</Td><Td>API unreachable — start the FastAPI backend</Td><Td>{" "}</Td></tr>}
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
          {!err && rows.length === 0 && <tr><Td>Loading…</Td><Td>{" "}</Td><Td>{" "}</Td><Td>{" "}</Td><Td>{" "}</Td><Td>{" "}</Td><Td>{" "}</Td></tr>}
        </DataTable>
      </Card>
    </AppShell>
  );
}
