"use client";
/** Customers — directory of enterprise accounts. Each row opens Customer 360. */
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, DataTable, Th, Td, Badge, TableState } from "@/components/ui/primitives";
import { api, CustomerListItem } from "@/lib/api";

const compact = (n: number) => n >= 1e6 ? `$${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `$${(n / 1e3).toFixed(0)}K` : `$${n}`;

export default function Customers() {
  const [rows, setRows] = useState<CustomerListItem[] | null>(null);
  const [err, setErr] = useState(false);
  const router = useRouter();
  useEffect(() => { api.customers().then((r) => setRows(r.customers)).catch(() => setErr(true)); }, []);

  return (
    <AppShell title="Customers">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>Customer 360 · single source of truth</div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Customers</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Every account, one view — click any customer for their full 360.</p>
      </div>
      <Card>
        <CardHead title="Enterprise accounts" hint={rows ? `${rows.length}` : ""} />
        <DataTable head={<><Th>Customer</Th><Th>Industry</Th><Th>Status</Th><Th num>Revenue</Th><Th num>Margin</Th><Th num>Orders</Th><Th>Risk</Th></>}>
          {rows?.map((c) => (
            <tr key={c.id} className="hover:bg-[color-mix(in_srgb,var(--accent)_6%,transparent)] cursor-pointer" onClick={() => router.push(`/customers/${c.id}`)}>
              <Td strong><span style={{ color: "var(--accent)" }}>{c.name}</span><div className="text-[0.6875rem] text-ink-3 font-normal">{c.region} · {c.country}</div></Td>
              <Td>{c.industry}</Td>
              <Td><Badge status={c.account_status === "Active" ? "good" : c.account_status === "Watch" ? "warning" : "critical"}>{c.account_status}</Badge></Td>
              <Td num>{compact(c.revenue)}</Td>
              <Td num>{c.net_margin_pct}%</Td>
              <Td num>{c.orders.toLocaleString()}</Td>
              <Td><Badge status={c.risk_status as "good" | "warning" | "critical" | "info"}>{c.risk_band}</Badge></Td>
            </tr>
          ))}
          {err && <TableState cols={7} kind="error" />}
          {!rows && !err && <TableState cols={7} kind="loading" />}
        </DataTable>
      </Card>
    </AppShell>
  );
}
