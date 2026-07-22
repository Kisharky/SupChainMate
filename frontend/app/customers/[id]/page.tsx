"use client";
/**
 * Customer 360 — the single source of truth for one customer. Aggregates the
 * existing Commercial Intelligence figures, Decision Brain memories, Knowledge/
 * RAG documents, forecasting, logistics, recommendations, and an AI chat — one
 * page reused everywhere a customer is clicked (/customers/{id}).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import dynamic from "next/dynamic";
import { AppShell } from "@/components/AppShell";
import {
  Card, CardHead, KpiCard, Badge, Button, Progress, Sparkline, DataTable, Th, Td,
  Modal, EmptyState, Alert, Skeleton,
} from "@/components/ui/primitives";
import {
  api, CustomerDetail, CustomerOrders, CustomerShipments, CustomerForecast,
  CustomerRecs, CustomerTimeline, CustomerBrain, MapResponse,
} from "@/lib/api";

const CustomerMap = dynamic(() => import("@/components/CustomerMap"), {
  ssr: false, loading: () => <div style={{ height: 260 }} className="grid place-items-center"><EmptyState kind="loading" /></div>,
});

const money = (n: number) => `$${Math.round(n).toLocaleString()}`;
const compact = (n: number) => n >= 1e6 ? `$${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `$${(n / 1e3).toFixed(0)}K` : `$${n}`;
const statusBadge = (s: string) => /deliver|approved|complet|on.?time/i.test(s) ? "good" : /delay|reject|late|pending/i.test(s) ? "warning" : "info";

export default function Customer360() {
  const params = useParams();
  const id = String(params?.id ?? "");
  const [d, setD] = useState<CustomerDetail | null>(null);
  const [err, setErr] = useState(false);
  const [orders, setOrders] = useState<CustomerOrders | null>(null);
  const [ship, setShip] = useState<CustomerShipments | null>(null);
  const [fc, setFc] = useState<CustomerForecast | null>(null);
  const [recs, setRecs] = useState<CustomerRecs | null>(null);
  const [tl, setTl] = useState<CustomerTimeline | null>(null);
  const [brain, setBrain] = useState<CustomerBrain | null>(null);
  const [map, setMap] = useState<MapResponse | null>(null);
  const [order, setOrder] = useState<CustomerOrders["orders"][number] | null>(null);
  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!id) return;
    setD(null); setErr(false);
    api.customer(id).then((r) => (r.ok ? setD(r) : setErr(true))).catch(() => setErr(true));
    api.customerOrders(id).then(setOrders).catch(() => {});
    api.customerShipments(id).then(setShip).catch(() => {});
    api.customerForecast(id).then(setFc).catch(() => {});
    api.customerRecommendations(id).then(setRecs).catch(() => {});
    api.customerTimeline(id).then(setTl).catch(() => {});
    api.customerBrain(id).then(setBrain).catch(() => {});
    api.logisticsMap().then(setMap).catch(() => {});
  }, [id]);

  const exportJson = () => {
    if (!d) return;
    const blob = new Blob([JSON.stringify(d, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `${d.id}-customer360.json`; a.click();
    URL.revokeObjectURL(url);
  };

  if (err) return <AppShell title="Customer"><Alert status="critical" title="Customer not found">Couldn&apos;t load this customer. Start the backend or check the ID.</Alert></AppShell>;
  if (!d) return <AppShell title="Customer 360"><HeaderSkeleton /></AppShell>;

  return (
    <AppShell title={d.name}>
      {/* ---- Header ---- */}
      <div className="flex items-start justify-between gap-4 flex-wrap mb-5">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <Badge status={d.account_status === "Active" ? "good" : d.account_status === "Watch" ? "warning" : "critical"}>{d.account_status}</Badge>
            <Badge status={d.risk_status as "good" | "warning" | "critical" | "info"}>{d.risk_band} risk</Badge>
            <span className="text-[0.75rem] text-ink-3 font-mono">{d.id}</span>
          </div>
          <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">{d.name}</h1>
          <div className="flex gap-4 mt-2 text-[0.8125rem] text-ink-2 flex-wrap">
            <span>{d.industry}</span><span>· {d.country}</span><span>· RM {d.relationship_manager}</span>
            <span>· Health <b style={{ color: d.health_score >= 65 ? "var(--good)" : d.health_score >= 45 ? "var(--warning)" : "var(--critical)" }}>{d.health_score}</b></span>
          </div>
        </div>
        <div className="flex gap-2 flex-wrap">
          <Button sm variant="secondary" onClick={exportJson}>Export</Button>
          <Button sm variant="secondary" onClick={() => document.getElementById("sec-exec")?.scrollIntoView({ behavior: "smooth" })}>Generate AI Brief</Button>
          <Button sm variant="primary" onClick={() => chatRef.current?.scrollIntoView({ behavior: "smooth" })}>Chat with Customer Data</Button>
        </div>
      </div>

      {/* ---- 1. Executive Summary ---- */}
      <Section title="Executive summary" hint={`${d.exec_summary.generated_by} · ${d.exec_summary.confidence}% confidence`} id="sec-exec">
        <ul className="flex flex-col gap-1.5">
          {d.exec_summary.bullets.map((b, i) => (
            <li key={i} className="flex gap-2 text-[0.9375rem] text-ink"><span style={{ color: "var(--accent)" }}>›</span>{b}</li>
          ))}
        </ul>
      </Section>

      {/* ---- 2. Commercial Overview ---- */}
      <Section title="Commercial overview">
        <div className="grid gap-3 mb-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))" }}>
          <KpiCard label="Revenue" value={compact(d.commercial.revenue)} status="good" seed={3} />
          <KpiCard label="Profit" value={compact(d.commercial.profit)} status={d.commercial.profit >= 0 ? "good" : "critical"} seed={6} />
          <KpiCard label="Margin" value={d.commercial.margin_pct} unit="%" status={d.commercial.margin_pct >= 6 ? "good" : "warning"} seed={2} />
          <KpiCard label="Orders" value={d.commercial.orders.toLocaleString()} status="info" seed={9} />
          <KpiCard label="Avg Order Value" value={money(d.commercial.aov)} status="info" seed={5} />
          <KpiCard label="Lifetime Value" value={compact(d.commercial.clv)} status="good" seed={8} />
          <KpiCard label="Outstanding" value={compact(d.commercial.outstanding)} status="warning" seed={11} />
        </div>
        <div className="grid gap-4" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
          <TrendCard title="Revenue trend" data={d.revenue_trend} color="var(--good)" />
          <TrendCard title="Margin trend" data={d.margin_trend} color="var(--accent)" />
          <div>
            <div className="eyebrow mb-2">Top products</div>
            <div className="flex flex-col gap-1.5">
              {d.top_products.map((p) => (
                <div key={p.name} className="flex justify-between text-[0.8125rem]"><span className="text-ink-2">{p.name}</span><span className="tnum text-ink">{compact(p.revenue)}</span></div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* ---- 3. Orders ---- */}
      <Section title="Orders" hint={orders ? `${orders.orders.length} recent` : ""}>
        <DataTable head={<><Th>Order</Th><Th>Status</Th><Th>Warehouse</Th><Th>ETA</Th><Th>Carrier</Th><Th num>Value</Th></>}>
          {orders?.orders.map((o) => (
            <tr key={o.order_no} className="hover:bg-[color-mix(in_srgb,var(--accent)_6%,transparent)] cursor-pointer" onClick={() => setOrder(o)}>
              <Td strong><span className="font-mono text-[0.75rem]">{o.order_no}</span></Td>
              <Td><Badge status={statusBadge(o.status) as "good" | "warning" | "info"}>{o.status}</Badge></Td>
              <Td>{o.warehouse}</Td><Td>{o.eta}</Td><Td>{o.carrier}</Td><Td num>{money(o.value)}</Td>
            </tr>
          ))}
          {!orders && <tr><Td>Loading…</Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td></tr>}
        </DataTable>
      </Section>

      {/* ---- 4. Shipments ---- */}
      <Section title="Shipments" hint={ship ? `${ship.late_deliveries} late · ${ship.avg_lead_time}d avg lead` : ""}>
        <div className="grid gap-4" style={{ gridTemplateColumns: "1.1fr 1fr" }}>
          <div className="rounded-lg overflow-hidden border" style={{ borderColor: "var(--hairline)" }}>
            {ship ? <CustomerMap points={ship.map.points} tiles={map?.tiles_url} attribution={map?.attribution} /> : <div style={{ height: 260 }} className="grid place-items-center"><EmptyState kind="loading" /></div>}
          </div>
          <div>
            <div className="grid grid-cols-3 gap-2 mb-3">
              {ship && [["Current", ship.current_deliveries], ["Late", ship.late_deliveries], ["Avg lead", `${ship.avg_lead_time}d`]].map(([l, v]) => (
                <div key={l as string} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                  <div className="eyebrow">{l}</div><div className="text-[1.1rem] font-bold tnum mt-0.5">{v}</div>
                </div>
              ))}
            </div>
            <div className="eyebrow mb-1.5">Carrier performance</div>
            {ship?.carrier_performance.map((c) => (
              <div key={c.carrier} className="flex items-center gap-2 text-[0.75rem] mb-1">
                <span className="flex-1 text-ink-2">{c.carrier}</span><span className="text-ink-3">{c.shipments} shp</span>
                <Badge status={c.on_time_pct >= 90 ? "good" : "warning"}>{c.on_time_pct}%</Badge>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* ---- 5. Forecast ---- */}
      <Section title="Forecast" hint={fc ? `${fc.coverage_days}d cover · ${fc.stockout_probability}% stockout risk` : ""}>
        {fc ? (
          <div className="grid gap-4" style={{ gridTemplateColumns: "2fr 1fr" }}>
            <div>
              <div className="eyebrow mb-2">Demand — historical &amp; predicted</div>
              <div className="flex items-end gap-1 h-24">
                {fc.historical.map((h, i) => <Bar key={`h${i}`} value={h.demand} max={maxDemand(fc)} color="var(--text-3)" label={`${h.period}: ${h.demand}`} />)}
                {fc.predicted.map((p, i) => <Bar key={`p${i}`} value={p.demand} max={maxDemand(fc)} color="var(--accent)" label={`${p.period}: ${p.demand}`} />)}
              </div>
              <div className="flex gap-4 mt-2 text-[0.6875rem] text-ink-3"><span><span style={{ color: "var(--text-3)" }}>●</span> historical</span><span><span style={{ color: "var(--accent)" }}>●</span> predicted</span></div>
            </div>
            <div className="flex flex-col gap-2">
              {[["Inventory coverage", `${fc.coverage_days} days`], ["Stockout probability", `${fc.stockout_probability}%`], ["Suggested replenishment", fc.suggested_replenishment.toLocaleString()]].map(([l, v]) => (
                <div key={l as string} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                  <div className="eyebrow">{l}</div><div className="text-[1.05rem] font-bold tnum mt-0.5">{v}</div>
                </div>
              ))}
            </div>
          </div>
        ) : <EmptyState kind="loading" />}
      </Section>

      {/* ---- 8. Risk Analysis ---- */}
      <Section title="Risk analysis" hint={`overall ${d.risk.overall} · ${d.risk.band}`}>
        <div className="grid gap-4" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <div className="flex flex-col gap-2">
            {d.risk.dimensions.map((dim) => (
              <div key={dim.name}>
                <div className="flex justify-between text-[0.75rem] mb-1"><span className="text-ink-2">{dim.name}</span><span className="tnum text-ink-3">{dim.score}</span></div>
                <Progress value={dim.score} status={dim.score >= 70 ? "critical" : dim.score >= 45 ? "warning" : "good"} />
              </div>
            ))}
          </div>
          <Alert status={d.risk.overall >= 70 ? "critical" : d.risk.overall >= 45 ? "warning" : "good"} title={`AI risk explanation · trend ${d.risk.trend}`}>{d.risk.explanation}</Alert>
        </div>
      </Section>

      {/* ---- 9. Recommendations ---- */}
      <Section title="Recommendations" hint={recs ? `${recs.recommendations.length}` : ""}>
        <div className="flex flex-col gap-3">
          {recs?.recommendations.map((r) => <RecCard key={r.id} rec={r} />)}
          {!recs && <EmptyState kind="loading" />}
        </div>
      </Section>

      {/* ---- 6. Decision Brain ---- */}
      <BrainSection brain={brain} />

      {/* ---- 7. Knowledge ---- */}
      <Section title="Knowledge" hint={`${d.knowledge.count} document(s) · RAG`}>
        {d.knowledge.documents.length ? (
          <div className="flex flex-col gap-2">
            {d.knowledge.documents.map((doc, i) => (
              <div key={i} className="rounded border p-3" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                <div className="text-[0.8125rem] font-semibold">{doc.doc}</div>
                <div className="text-[0.75rem] text-ink-3 mt-1">{doc.snippet}</div>
              </div>
            ))}
          </div>
        ) : <EmptyState title="No documents linked yet" hint="Upload contracts, emails, or policies in the Data Hub — they'll surface here via RAG." />}
      </Section>

      {/* ---- 10. Activity Timeline ---- */}
      <Section title="Activity timeline">
        <div className="flex flex-col">
          {tl?.events.map((e, i) => (
            <div key={i} className="flex gap-3 py-2 border-b last:border-0" style={{ borderColor: "var(--hairline)" }}>
              <span className="mt-1 h-2 w-2 rounded-full flex-none" style={{ background: `var(--${e.status})` }} />
              <div className="flex-1"><div className="text-[0.875rem] font-medium">{e.label}</div><div className="text-[0.6875rem] text-ink-3">{e.detail}</div></div>
              <span className="text-[0.6875rem] text-ink-3 whitespace-nowrap">{e.hours_ago}h ago</span>
            </div>
          ))}
          {!tl && <EmptyState kind="loading" />}
        </div>
      </Section>

      {/* ---- 11. AI Chat ---- */}
      <div ref={chatRef}><ChatSection id={id} name={d.name} /></div>

      {/* Order details modal */}
      <Modal open={!!order} onClose={() => setOrder(null)} title={order ? `Order ${order.order_no}` : "Order"}
        subtitle={order ? `${order.status} · ${order.carrier}` : undefined}>
        {order && (
          <div className="grid grid-cols-2 gap-3">
            {[["Order", order.order_no], ["Status", order.status], ["Warehouse", order.warehouse], ["ETA", order.eta], ["Carrier", order.carrier], ["Value", money(order.value)]].map(([l, v]) => (
              <div key={l} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                <div className="eyebrow">{l}</div><div className="text-[0.9375rem] font-semibold mt-0.5">{v}</div>
              </div>
            ))}
          </div>
        )}
      </Modal>
    </AppShell>
  );
}

/* ---- helpers ---- */
function Section({ title, hint, children, id, defaultOpen = true }: { title: string; hint?: string; children: React.ReactNode; id?: string; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card className="mb-4" id={id}>
      <button onClick={() => setOpen((o) => !o)} className="w-full flex items-center justify-between gap-3 px-[18px] py-3.5"
        style={{ borderBottom: open ? "1px solid var(--hairline)" : "none" }}>
        <span className="text-[0.9375rem] font-semibold">{title}</span>
        <span className="flex items-center gap-2 text-ink-3">{hint && <span className="text-[0.75rem]">{hint}</span>}<span className="text-[13px]">{open ? "▾" : "▸"}</span></span>
      </button>
      {open && <div className="p-[18px]">{children}</div>}
    </Card>
  );
}

const maxDemand = (fc: CustomerForecast) => Math.max(1, ...fc.historical.map((h) => h.demand), ...fc.predicted.map((p) => p.demand));
function Bar({ value, max, color, label }: { value: number; max: number; color: string; label: string }) {
  return <div className="flex-1 rounded-t" style={{ height: `${Math.max(4, (100 * value) / max)}%`, background: color, opacity: 0.85 }} title={label} />;
}
function TrendCard({ title, data, color }: { title: string; data: { period: string; value: number }[]; color: string }) {
  const last = data[data.length - 1]?.value ?? 0;
  const first = data[0]?.value ?? 0;
  const up = last >= first;
  return (
    <div>
      <div className="eyebrow mb-2">{title}</div>
      <Sparkline seed={Math.round(first) % 12 + 1} color={color} trend={up ? "up" : "down"} w={140} h={44} />
      <div className="text-[0.75rem] mt-1" style={{ color: up ? "var(--good)" : "var(--critical)" }}>{up ? "▲" : "▼"} {Math.abs(Math.round((last - first) / Math.max(1, first) * 100))}%</div>
    </div>
  );
}

function RecCard({ rec }: { rec: CustomerRecs["recommendations"][number] }) {
  const [status, setStatus] = useState(rec.status);
  const [showReason, setShowReason] = useState(false);
  return (
    <Card className="p-4">
      <div className="flex items-center gap-2 flex-wrap">
        <Badge status={statusBadge(status) as "good" | "warning" | "info"}>{status}</Badge>
        <span className="font-semibold text-[0.9375rem]">{rec.title}</span>
        <span className="ml-auto text-[0.75rem] text-ink-3">impact <b className="text-ink">{rec.business_impact}</b> · saves <b className="text-ink tnum">{compact(rec.estimated_savings)}</b> · {rec.confidence}%</span>
      </div>
      {showReason && <p className="text-[0.8125rem] text-ink-2 mt-2">{rec.reasoning}</p>}
      <div className="flex gap-2 mt-3">
        <Button sm variant="ghost" onClick={() => setShowReason((s) => !s)}>{showReason ? "Hide reasoning" : "View reasoning"}</Button>
        {status === "Pending" && <>
          <Button sm variant="primary" onClick={() => setStatus("Approved")}>Approve</Button>
          <Button sm variant="danger" onClick={() => setStatus("Rejected")}>Reject</Button>
        </>}
      </div>
    </Card>
  );
}

const BRAIN_KIND_LABEL: Record<string, string> = {
  recommendation: "Recommendations", approval: "Approvals", outcome: "Outcomes",
  feedback: "Feedback", knowledge: "Knowledge", decision: "Decisions", entity: "Entities",
};
function BrainSection({ brain }: { brain: CustomerBrain | null }) {
  const [q, setQ] = useState("");
  const groups = useMemo(() => {
    if (!brain) return {};
    if (!q.trim()) return brain.groups;
    const out: CustomerBrain["groups"] = {};
    for (const [k, items] of Object.entries(brain.groups)) {
      const f = items.filter((i) => (i.title + i.snippet).toLowerCase().includes(q.toLowerCase()));
      if (f.length) out[k] = f;
    }
    return out;
  }, [brain, q]);
  return (
    <Section title="Decision Brain memory" hint={brain ? `${brain.total} memories` : ""}>
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Semantic search inside this customer's memory…"
        className="w-full mb-3 rounded-sm border bg-[var(--panel-2)] px-3 py-2 text-[0.8125rem] text-ink" style={{ borderColor: "var(--hairline-strong)" }} />
      {!brain ? <EmptyState kind="loading" /> : Object.keys(groups).length === 0 ? (
        <EmptyState title="No memories yet" hint="Decisions, approvals, and outcomes for this customer will appear here as the platform learns." />
      ) : (
        <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))" }}>
          {Object.entries(groups).map(([kind, items]) => (
            <div key={kind} className="rounded-lg border p-3" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
              <div className="flex items-center justify-between mb-2"><span className="text-[0.8125rem] font-semibold">{BRAIN_KIND_LABEL[kind] ?? kind}</span><Badge status="neutral">{items.length}</Badge></div>
              {items.slice(0, 4).map((m, i) => (
                <div key={i} className="text-[0.6875rem] text-ink-3 py-1 border-b last:border-0" style={{ borderColor: "var(--hairline)" }}>{m.title}</div>
              ))}
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}

const EXAMPLES = ["Why is this customer high risk?", "Summarise this customer.", "Show shipment delays.", "How profitable is this customer?"];
function ChatSection({ id, name }: { id: string; name: string }) {
  const [msgs, setMsgs] = useState<{ role: "user" | "ai"; text: string }[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const send = async (text: string) => {
    if (!text.trim() || busy) return;
    setMsgs((m) => [...m, { role: "user", text }]); setInput(""); setBusy(true);
    try {
      const res = await api.customerChat(id, text);
      setMsgs((m) => [...m, { role: "ai", text: res.answer + (res.context ? `\n\n— grounded in: ${res.context}` : "") }]);
    } catch { setMsgs((m) => [...m, { role: "ai", text: "Chat unavailable — start the backend." }]); }
    finally { setBusy(false); }
  };
  return (
    <Section title="AI chat" hint={`ask about ${name}`}>
      <div className="flex flex-wrap gap-1.5 mb-3">
        {EXAMPLES.map((e) => <button key={e} onClick={() => send(e)} className="rounded-full border px-2.5 py-1 text-[0.6875rem] text-ink-2" style={{ borderColor: "var(--hairline)" }}>{e}</button>)}
      </div>
      <div className="flex flex-col gap-2 mb-3 max-h-[320px] overflow-y-auto">
        {msgs.length === 0 && <div className="text-[0.8125rem] text-ink-3">Ask anything — answers draw on the Decision Brain, Commercial Intelligence, and Knowledge Center.</div>}
        {msgs.map((m, i) => (
          <div key={i} className={`rounded-lg px-3 py-2 text-[0.8125rem] whitespace-pre-wrap max-w-[85%] ${m.role === "user" ? "self-end" : "self-start"}`}
            style={{ background: m.role === "user" ? "color-mix(in srgb,var(--accent) 16%,transparent)" : "var(--panel-2)", border: "1px solid var(--hairline)" }}>{m.text}</div>
        ))}
        {busy && <div className="self-start text-[0.75rem] text-ink-3">Thinking…</div>}
      </div>
      <div className="flex gap-2">
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send(input)}
          placeholder="Ask about this customer…" className="flex-1 rounded-sm border bg-[var(--panel-2)] px-3 py-2 text-[0.8125rem] text-ink" style={{ borderColor: "var(--hairline-strong)" }} />
        <Button sm variant="primary" onClick={() => send(input)} disabled={busy}>Send</Button>
      </div>
    </Section>
  );
}

function HeaderSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <div><Skeleton w="30%" h="1rem" /><div className="mt-3"><Skeleton w="45%" h="2rem" /></div></div>
      <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))" }}>
        {Array.from({ length: 6 }).map((_, i) => <Card key={i} className="p-4"><Skeleton w="60%" /><div className="mt-3"><Skeleton w="40%" h="1.6rem" /></div></Card>)}
      </div>
    </div>
  );
}
