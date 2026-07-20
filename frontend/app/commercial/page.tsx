"use client";
/**
 * Commercial Intelligence — customer profitability, revenue leakage, margin
 * waterfall, and one-click repricing tickets with AI-drafted outreach.
 * Order volumes are real; margin economics are modelled (assumptions shown).
 */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, DataTable, Th, Td, Badge, Button } from "@/components/ui/primitives";
import { api, CommercialResponse, EmailResponse } from "@/lib/api";

const money = (n: number) => `$${Math.round(n).toLocaleString()}`;

function Waterfall({ data }: { data: CommercialResponse["waterfall"] }) {
  const max = Math.max(...data.map((d) => Math.abs(d.value)));
  return (
    <div className="flex flex-col gap-2.5">
      {data.map((d) => {
        const color = d.kind === "start" ? "var(--info)" : d.kind === "end" ? "var(--good)" : "var(--critical)";
        return (
          <div key={d.label} className="flex items-center gap-3">
            <div className="w-32 text-[0.8125rem] text-ink-2 flex-none">{d.label}</div>
            <div className="flex-1 h-5 rounded-sm overflow-hidden" style={{ background: "var(--hairline)" }}>
              <div className="h-full rounded-sm" style={{ width: `${(Math.abs(d.value) / max) * 100}%`, background: color, opacity: 0.85 }} />
            </div>
            <div className="w-28 text-right tnum text-[0.8125rem] flex-none" style={{ color: d.value < 0 ? "var(--critical)" : "var(--ink)" }}>
              {d.value < 0 ? "−" : ""}{money(Math.abs(d.value))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Commercial() {
  const [d, setD] = useState<CommercialResponse | null>(null);
  const [email, setEmail] = useState<EmailResponse | null>(null);
  const [drafting, setDrafting] = useState<string | null>(null);
  useEffect(() => { api.commercial().then(setD).catch(() => setD(null)); }, []);
  const k = d?.kpis;

  const draft = async (sku: string) => {
    setDrafting(sku);
    try { setEmail(await api.repricingEmail(sku)); }
    finally { setDrafting(null); }
  };

  return (
    <AppShell title="Commercial Intelligence">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>Revenue & margin command · {d?.source === "live" ? "live" : "loading"}</div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Commercial Intelligence</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Customer profitability, revenue leakage, and pricing actions from real order volumes.</p>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))" }}>
        {[
          { l: "Total Revenue", v: k ? money(k.total_revenue) : "—", s: "info" },
          { l: "Net Margin", v: k ? `${k.net_margin_pct}%` : "—", s: "good" },
          { l: "Revenue Leakage", v: k ? money(k.revenue_leakage) : "—", s: "critical" },
          { l: "Underpriced SKUs", v: k?.underpriced_skus ?? "—", s: "warning" },
          { l: "Repricing Upside/yr", v: k ? money(k.repricing_upside) : "—", s: "good" },
        ].map((m) => (
          <Card key={m.l} className="p-4 relative overflow-hidden">
            <span className="absolute left-0 top-0 bottom-0 w-[3px]" style={{ background: `var(--${m.s})` }} />
            <div className="eyebrow">{m.l}</div>
            <div className="text-[1.6rem] font-bold tnum mt-1">{m.v}</div>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 mt-4 items-start" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Card>
          <CardHead title="Margin waterfall" hint="gross → net" />
          <div className="p-[18px]">{d ? <Waterfall data={d.waterfall} /> : <p className="text-ink-3 text-[0.8125rem]">Loading…</p>}</div>
        </Card>
        <Card>
          <CardHead title="Customer profitability" hint="by region · top segments" />
          <DataTable head={<><Th>Segment</Th><Th num>Orders</Th><Th num>Revenue</Th><Th num>Margin</Th></>}>
            {(d?.segments ?? []).slice(0, 8).map((s) => (
              <tr key={s.segment}>
                <Td strong>{s.segment}</Td>
                <Td num>{s.orders.toLocaleString()}</Td>
                <Td num>{money(s.revenue)}</Td>
                <Td num>{money(s.margin)}</Td>
              </tr>
            ))}
          </DataTable>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHead title="Repricing tickets" hint="underpriced vs 35% target margin · one-click outreach" />
        <DataTable head={<><Th>SKU</Th><Th num>Current</Th><Th num>Recommended</Th><Th num>Uplift</Th><Th num>Margin now</Th><Th num>Annual impact</Th><Th>Action</Th></>}>
          {(d?.tickets ?? []).map((t) => (
            <tr key={t.sku}>
              <Td strong>{t.sku}</Td>
              <Td num>${t.current_price}</Td>
              <Td num strong>${t.recommended_price}</Td>
              <Td num><Badge status="good">+{t.uplift_pct}%</Badge></Td>
              <Td num>{t.current_margin_pct}%</Td>
              <Td num strong>{money(t.annual_impact)}</Td>
              <Td><Button variant="secondary" sm onClick={() => draft(t.sku)} disabled={drafting === t.sku}>{drafting === t.sku ? "Drafting…" : "✎ Draft email"}</Button></Td>
            </tr>
          ))}
          {(!d?.tickets || d.tickets.length === 0) && <tr><Td>Loading…</Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td></tr>}
        </DataTable>
        {d?.assumptions && (
          <div className="px-[18px] py-3 text-[0.6875rem] text-ink-3 border-t" style={{ borderColor: "var(--hairline)" }}>
            Modelled assumptions: AOV ${d.assumptions.aov} · gross margin {d.assumptions.gross_margin_pct}% · target {d.assumptions.target_margin_pct}%. Order volumes are real; wire line-item revenue for exact figures.
          </div>
        )}
      </Card>

      {email && (
        <Card className="mt-4" style={{ borderColor: "color-mix(in srgb,var(--accent) 30%,var(--hairline))" }}>
          <CardHead title={`Draft — ${email.subject}`} hint="AI-generated · edit before sending"
            right={<Button variant="ghost" sm onClick={() => setEmail(null)}>✕ Close</Button>} />
          <div className="p-[18px]">
            <pre className="whitespace-pre-wrap text-[0.875rem] text-ink font-sans leading-relaxed">{email.body}</pre>
            <div className="flex gap-2 mt-3">
              <Button variant="primary" sm>Send to approver</Button>
              <Button variant="ghost" sm onClick={() => navigator.clipboard?.writeText(email.body)}>⧉ Copy</Button>
            </div>
          </div>
        </Card>
      )}
    </AppShell>
  );
}
