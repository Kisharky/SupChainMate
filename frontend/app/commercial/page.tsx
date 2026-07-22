"use client";
/**
 * Commercial Intelligence — an AI commercial decision centre.
 * Executive brief · true customer profitability (ABC) · Customer 360 · revenue
 * leakage · contract intelligence · AI pricing optimiser · customer risk scoring.
 */
import { useEffect, useState } from "react";
import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, Button, Badge, DataTable, Th, Td } from "@/components/ui/primitives";
import {
  api, CiBrief, CiProfitability, CiCustomer, CiLeakage, CiContracts, CiPricing, CiRisk,
} from "@/lib/api";

const money = (n: number) => `$${Math.round(Math.abs(n)).toLocaleString()}`;
const bandColor = { low: "good", medium: "warning", high: "critical" } as const;

export default function Commercial() {
  const [brief, setBrief] = useState<CiBrief | null>(null);
  const [prof, setProf] = useState<CiProfitability | null>(null);
  const [leak, setLeak] = useState<CiLeakage | null>(null);
  const [contracts, setContracts] = useState<CiContracts | null>(null);
  const [pricing, setPricing] = useState<CiPricing | null>(null);
  const [risk, setRisk] = useState<CiRisk | null>(null);
  const [selected, setSelected] = useState<CiCustomer | null>(null);

  useEffect(() => {
    api.ciBrief().then(setBrief).catch(() => {});
    api.ciProfitability().then(setProf).catch(() => {});
    api.ciLeakage().then(setLeak).catch(() => {});
    api.ciContracts().then(setContracts).catch(() => {});
    api.ciPricing().then(setPricing).catch(() => {});
    api.ciRisk().then(setRisk).catch(() => {});
  }, []);

  const open360 = (id: string) => api.ciCustomer(id).then(setSelected).catch(() => {});

  return (
    <AppShell title="Commercial Intelligence">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>AI Commercial Decision Centre · cost-to-serve intelligence</div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Commercial Intelligence</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">True customer profitability, revenue leakage, contract performance, and AI pricing — not a report, a decision centre.</p>
      </div>

      <ExecBrief brief={brief} />
      <Profitability prof={prof} onOpen={open360} />
      {selected && <Customer360 c={selected} onClose={() => setSelected(null)} onInvoice={open360} />}
      <LeakageCenter leak={leak} onInvoice={open360} />
      <div className="grid gap-4 mt-4 items-start" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Contracts data={contracts} />
        <Pricing data={pricing} />
      </div>
      <RiskScoring data={risk} />
      <p className="mt-4 text-[0.6875rem] text-ink-3">Account revenue and order volume are real (Olist, aggregated by region into enterprise accounts). Cost-to-serve, contracts, payments, and SLAs are modelled with transparent per-account factors — wire a real cost/contract feed for exact figures.</p>
    </AppShell>
  );
}

function Scorecard({ l, v, s, sub }: { l: string; v: string; s?: string; sub?: string }) {
  return (
    <div className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
      <div className="eyebrow">{l}</div>
      <div className="text-[1.3rem] font-bold tnum mt-0.5" style={{ color: s ? `var(--${s})` : "var(--ink)" }}>{v}</div>
      {sub && <div className="text-[0.6875rem] text-ink-3">{sub}</div>}
    </div>
  );
}

/* ═══ Executive Commercial Brief ═══ */
function ExecBrief({ brief }: { brief: CiBrief | null }) {
  const [done, setDone] = useState<Record<string, string>>({});
  const act = async (title: string, action: string) => {
    await api.ciDecide(title, action);
    setDone((d) => ({ ...d, [title]: action }));
  };
  return (
    <Card className="p-[18px]" style={{ background: "linear-gradient(180deg,color-mix(in srgb,var(--accent) 7%,var(--panel)),var(--panel))", borderColor: "color-mix(in srgb,var(--accent) 26%,var(--hairline))" }}>
      <div className="flex items-center justify-between">
        <span className="inline-flex items-center gap-2 text-[0.75rem] font-bold uppercase tracking-wider" style={{ color: "var(--accent)" }}>
          <span className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ background: "var(--accent)" }} /> Executive Commercial Brief
        </span>
        {brief && <Badge status={brief.customers_action ? "warning" : "good"}>{brief.customers_action} accounts need action</Badge>}
      </div>
      <p className="text-[1.0625rem] leading-snug my-3 text-ink">{brief?.summary ?? "Analysing account profitability…"}</p>
      <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(140px,1fr))" }}>
        <Scorecard l="Total revenue" v={brief ? money(brief.total_revenue) : "—"} s="info" />
        <Scorecard l="True operating cost" v={brief ? money(brief.true_operating_cost) : "—"} s="critical" />
        <Scorecard l="Gross margin" v={brief ? `${brief.gross_margin_pct}%` : "—"} sub="before cost-to-serve" />
        <Scorecard l="True net margin" v={brief ? `${brief.net_margin_pct}%` : "—"} s={brief && brief.net_margin_pct < 8 ? "warning" : "good"} sub="after cost-to-serve" />
        <Scorecard l="Revenue leakage" v={brief ? money(brief.revenue_leakage) : "—"} s="critical" />
        <Scorecard l="Profit uplift" v={brief ? money(brief.profit_uplift) : "—"} s="good" sub="modelled, /yr" />
      </div>
      {brief && brief.recommendations.length > 0 && (
        <div className="mt-4">
          <div className="eyebrow mb-2">AI commercial recommendations</div>
          <div className="flex flex-col gap-2">
            {brief.recommendations.map((r, i) => (
              <div key={i} className="flex items-center gap-3 rounded border p-2.5 flex-wrap" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                <div className="min-w-0 flex-1">
                  <div className="text-[0.875rem] font-semibold">{r.title} <Badge status="good">{r.confidence}%</Badge></div>
                  <div className="text-[0.75rem] text-ink-3">{r.detail} · impact {money(r.impact_usd)}/yr</div>
                </div>
                {done[r.title]
                  ? <Badge status={done[r.title] === "REJECTED" ? "critical" : "good"}>{done[r.title].toLowerCase()}</Badge>
                  : <div className="flex gap-1.5">
                      <Button variant="primary" sm onClick={() => act(r.title, "APPROVED")}>Approve</Button>
                      <Button variant="secondary" sm onClick={() => act(r.title, "SCHEDULED")}>Schedule</Button>
                      <Button variant="ghost" sm onClick={() => act(r.title, "REJECTED")}>Reject</Button>
                    </div>}
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

/* ═══ True Customer Profitability ═══ */
function heat(pct: number) {
  // higher share of revenue = hotter (red); scale ~0–8%
  const t = Math.min(1, pct / 8);
  return `color-mix(in srgb, var(--critical) ${Math.round(t * 70)}%, transparent)`;
}
function Profitability({ prof, onOpen }: { prof: CiProfitability | null; onOpen: (id: string) => void }) {
  const max = prof ? Math.max(...prof.waterfall.map((w) => Math.abs(w.value))) : 1;
  return (
    <Card className="mt-4">
      <CardHead title="True Customer Profitability" hint="activity-based · click a row for Customer 360" />
      <div className="p-[18px] grid gap-4" style={{ gridTemplateColumns: "1.1fr 1fr" }}>
        <div>
          <div className="eyebrow mb-2">Account ranking (net margin)</div>
          <div className="overflow-x-auto rounded border" style={{ borderColor: "var(--hairline)" }}>
            <table className="w-full border-collapse text-[0.8125rem]">
              <thead><tr>
                <th className="text-left px-3 py-2 text-[10px] uppercase text-ink-3 border-b" style={{ borderColor: "var(--hairline)" }}>Account</th>
                <th className="text-right px-3 py-2 text-[10px] uppercase text-ink-3 border-b" style={{ borderColor: "var(--hairline)" }}>Revenue</th>
                <th className="text-right px-3 py-2 text-[10px] uppercase text-ink-3 border-b" style={{ borderColor: "var(--hairline)" }}>Net</th>
                <th className="text-right px-3 py-2 text-[10px] uppercase text-ink-3 border-b" style={{ borderColor: "var(--hairline)" }}>Margin</th>
              </tr></thead>
              <tbody>
                {(prof?.ranking ?? []).map((r) => (
                  <tr key={r.id} onClick={() => onOpen(r.id)} className="cursor-pointer hover:bg-[color-mix(in_srgb,var(--accent)_7%,transparent)]">
                    <td className="px-3 py-2 border-b" style={{ borderColor: "var(--hairline)" }}>
                      <Link href={`/customers/${r.id}`} onClick={(e) => e.stopPropagation()} className="font-medium hover:underline" style={{ color: "var(--accent)" }}>{r.name}</Link> <span className="text-ink-3 text-[0.6875rem]">{r.region}</span>
                      {r.action && <span className="ml-1.5"><Badge status="warning">action</Badge></span>}
                    </td>
                    <td className="px-3 py-2 border-b text-right tnum text-ink-2" style={{ borderColor: "var(--hairline)" }}>{money(r.revenue)}</td>
                    <td className="px-3 py-2 border-b text-right tnum" style={{ borderColor: "var(--hairline)", color: r.net_margin < 0 ? "var(--critical)" : "var(--ink)" }}>{r.net_margin < 0 ? "−" : ""}{money(r.net_margin)}</td>
                    <td className="px-3 py-2 border-b text-right tnum font-semibold" style={{ borderColor: "var(--hairline)", color: r.net_margin_pct < 0 ? "var(--critical)" : r.net_margin_pct < 8 ? "var(--warning)" : "var(--good)" }}>{r.net_margin_pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <div className="eyebrow mb-2">Margin waterfall (network)</div>
          <div className="flex flex-col gap-1.5">
            {(prof?.waterfall ?? []).map((w) => {
              const color = w.kind === "start" ? "var(--info)" : w.kind === "end" ? "var(--good)" : "var(--critical)";
              return (
                <div key={w.label} className="flex items-center gap-2">
                  <div className="w-28 text-[0.75rem] text-ink-2 flex-none truncate">{w.label}</div>
                  <div className="flex-1 h-4 rounded-sm overflow-hidden" style={{ background: "var(--hairline)" }}>
                    <div className="h-full rounded-sm" style={{ width: `${(Math.abs(w.value) / max) * 100}%`, background: color, opacity: 0.85 }} />
                  </div>
                  <div className="w-24 text-right tnum text-[0.75rem] flex-none" style={{ color: w.value < 0 ? "var(--critical)" : "var(--ink)" }}>{w.value < 0 ? "−" : ""}{money(w.value)}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      {/* Heatmap */}
      {prof?.heatmap?.rows?.length ? (
        <div className="px-[18px] pb-[18px]">
          <div className="eyebrow mb-2">Cost-to-serve heatmap (% of revenue)</div>
          <div className="overflow-x-auto">
            <table className="border-collapse text-[0.6875rem]" style={{ minWidth: 720 }}>
              <thead><tr>
                <th className="text-left px-2 py-1 text-ink-3 sticky left-0" style={{ background: "var(--panel)" }}>Account</th>
                {prof.heatmap.categories.map((c) => <th key={c} className="px-1.5 py-1 text-ink-3 text-center" style={{ writingMode: "vertical-rl" as any, transform: "rotate(180deg)", height: 68 }}>{c.replace(/_/g, " ")}</th>)}
              </tr></thead>
              <tbody>
                {prof.heatmap.rows.map((r) => (
                  <tr key={r.account}>
                    <td className="px-2 py-1 text-ink-2 whitespace-nowrap sticky left-0" style={{ background: "var(--panel)" }}>{r.account}</td>
                    {r.cells.map((v, i) => (
                      <td key={i} className="px-1.5 py-1 text-center tnum" style={{ background: heat(v), color: v > 5 ? "#fff" : "var(--ink-2)" }}>{v}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </Card>
  );
}

/* ═══ Customer 360 ═══ */
function Customer360({ c, onClose, onInvoice }: { c: CiCustomer; onClose: () => void; onInvoice: (id: string) => void }) {
  return (
    <Card className="mt-4" style={{ borderColor: "color-mix(in srgb,var(--accent) 35%,var(--hairline))" }}>
      <CardHead title={`Customer 360 — ${c.name}`} hint={`${c.region} · ${c.orders.toLocaleString()} orders`}
        right={<Button variant="ghost" sm onClick={onClose}>✕ Close</Button>} />
      <div className="p-[18px]">
        <div className="grid gap-3 mb-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))" }}>
          <Scorecard l="Revenue" v={money(c.revenue)} s="info" />
          <Scorecard l="True servicing cost" v={money(c.true_cost)} s="critical" />
          <Scorecard l="Net margin" v={`${c.net_margin_pct}%`} s={c.net_margin_pct < 0 ? "critical" : c.net_margin_pct < 8 ? "warning" : "good"} />
          <Scorecard l="Freight cost" v={money(c.freight)} />
          <Scorecard l="Returns" v={`${c.returns_pct}%`} />
          <Scorecard l="Storage util." v={`${c.storage_util}%`} />
          <Scorecard l="DSO" v={`${c.dso}d`} s={c.dso > 55 ? "warning" : undefined} sub={`${c.pay_on_time}% on-time`} />
          <Scorecard l="SLA" v={`${c.sla_actual}%`} s={c.sla_actual < 92 ? "critical" : "good"} sub={`target ${c.sla_target}%`} />
        </div>
        <div className="grid gap-4" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <div>
            <div className="eyebrow mb-2">Cost-to-serve breakdown</div>
            {Object.entries(c.activities).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
              <div key={k} className="flex justify-between text-[0.8125rem] py-1 border-b" style={{ borderColor: "var(--hairline)" }}>
                <span className="text-ink-2 capitalize">{k.replace(/_/g, " ")}</span><span className="tnum">{money(v)}</span>
              </div>
            ))}
          </div>
          <div>
            <div className="eyebrow mb-2">AI insights</div>
            {c.insights.map((t, i) => <div key={i} className="flex gap-2 text-[0.8125rem] text-ink-2 py-1"><span style={{ color: "var(--accent)" }}>›</span>{t}</div>)}
            <div className="eyebrow mt-3 mb-2">Contract &amp; forecast</div>
            <div className="text-[0.8125rem] text-ink-2 flex flex-col gap-1">
              <div>Pick fee <b className="text-ink">${String(c.contract.pick_fee)}</b> vs actual <b style={{ color: c.contract.pick_underpriced ? "var(--critical)" : "var(--good)" }}>${String(c.contract.actual_pick_cost)}</b></div>
              <div>Renewal in <b className="text-ink">{String(c.contract.renewal_months)} mo</b> · storage rate ${String(c.contract.storage_rate)}</div>
              <div>Next-qtr forecast <b className="text-ink">{c.forecast_next_qtr_orders.toLocaleString()} orders</b> ({c.vol_trend > 0 ? "+" : ""}{c.vol_trend}% trend)</div>
              <div>Inventory <b className="text-ink">{c.inventory_profile.skus} SKUs</b> · {c.inventory_profile.days_cover}d cover</div>
              {c.revenue_gap > 0 && <div style={{ color: "var(--warning)" }}>Repricing gap to target: <b>{money(c.revenue_gap)}/yr</b></div>}
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ═══ Revenue Leakage Center ═══ */
function LeakageCenter({ leak, onInvoice }: { leak: CiLeakage | null; onInvoice: (id: string) => void }) {
  const [inv, setInv] = useState<Record<string, string>>({});
  const gen = async (account: string, cause: string, id: string) => {
    const r = await api.ciInvoice(id, cause);
    setInv((s) => ({ ...s, [account + cause]: r.invoice_no }));
  };
  return (
    <Card className="mt-4">
      <CardHead title="Revenue Leakage Center" hint="detected unbilled / undercharged activity"
        right={leak ? <div className="flex gap-2"><Badge status="critical">{money(leak.annual_leakage)}/yr</Badge><Badge status="good">{money(leak.recoverable)} recoverable</Badge></div> : undefined} />
      <div className="p-[18px] grid gap-4" style={{ gridTemplateColumns: "1fr 1.4fr" }}>
        <div>
          <div className="eyebrow mb-2">By root cause · {leak?.affected_customers ?? 0} customers affected</div>
          {(leak?.by_cause ?? []).map((c) => (
            <div key={c.cause} className="py-1.5 border-b" style={{ borderColor: "var(--hairline)" }}>
              <div className="flex justify-between text-[0.8125rem]"><span className="text-ink font-medium">{c.cause}</span><span className="tnum text-critical">{money(c.amount)}</span></div>
              <div className="text-[0.6875rem] text-ink-3">{c.detail}</div>
            </div>
          ))}
        </div>
        <div>
          <div className="eyebrow mb-2">Recovery opportunities · one-click invoice</div>
          <div className="max-h-[320px] overflow-y-auto">
            <DataTable head={<><Th>Account</Th><Th>Cause</Th><Th num>Amount</Th><Th>Action</Th></>}>
              {(leak?.items ?? []).map((it, i) => {
                const id = it.account.toLowerCase().replace(/ /g, "-");
                const key = it.account + it.cause;
                return (
                  <tr key={i}>
                    <Td strong>{it.account}</Td>
                    <Td>{it.cause_label}</Td>
                    <Td num>{money(it.amount)}</Td>
                    <Td>{inv[key]
                      ? <Badge status="good">{inv[key]}</Badge>
                      : <Button variant="secondary" sm onClick={() => gen(it.account, it.cause, id)}>Generate invoice</Button>}</Td>
                  </tr>
                );
              })}
            </DataTable>
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ═══ Contract Intelligence ═══ */
function Contracts({ data }: { data: CiContracts | null }) {
  return (
    <Card>
      <CardHead title="Contract Intelligence" hint="contractual vs actual cost-to-serve"
        right={data ? <div className="flex gap-2"><Badge status="critical">{data.unprofitable_count} unprofitable</Badge><Badge status="warning">{data.renewals_90d} renew ≤90d</Badge></div> : undefined} />
      <div className="p-[18px] flex flex-col gap-2 max-h-[420px] overflow-y-auto">
        {(data?.contracts ?? []).slice(0, 10).map((c) => (
          <div key={c.account} className="rounded border p-2.5" style={{ borderColor: c.unprofitable ? "color-mix(in srgb,var(--critical) 40%,transparent)" : "var(--hairline)", background: "var(--panel-2)" }}>
            <div className="flex items-center justify-between gap-2">
              <div className="text-[0.875rem] font-semibold">{c.account} <span className="text-ink-3 text-[0.6875rem]">{c.region}</span></div>
              <div className="flex gap-1.5">
                {c.unprofitable && <Badge status="critical">unprofitable</Badge>}
                {c.pick_underpriced && <Badge status="warning">pick underpriced</Badge>}
                {c.renewal_soon && <Badge status="info">renews soon</Badge>}
              </div>
            </div>
            <div className="grid grid-cols-4 gap-x-3 gap-y-1 text-[0.6875rem] mt-1.5 text-ink-3">
              <div>Pick <b className="text-ink">${c.contractual_pick}</b> / act ${c.actual_pick}</div>
              <div>Storage ${c.terms.storage_rate}</div>
              <div>Fuel {c.terms.fuel_surcharge_pct}%</div>
              <div>Rebate {c.terms.rebate_pct}%</div>
              <div>SLA {c.terms.sla_target}%</div>
              <div>Penalty ${c.terms.penalty_per_breach}</div>
              <div>Renews {c.terms.renewal_months}mo</div>
              <div>Net <b style={{ color: c.net_margin_pct < 0 ? "var(--critical)" : "var(--ink)" }}>{c.net_margin_pct}%</b></div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ═══ AI Pricing Optimizer ═══ */
function Pricing({ data }: { data: CiPricing | null }) {
  const [done, setDone] = useState<Record<string, string>>({});
  const act = async (id: string, account: string, action: string) => {
    await api.ciDecide(`Reprice ${account}`, action);
    setDone((d) => ({ ...d, [id]: action }));
  };
  return (
    <Card>
      <CardHead title="AI Pricing Optimizer" hint="revised fees · churn risk · uplift"
        right={data ? <Badge status="good">{money(data.total_uplift)}/yr uplift</Badge> : undefined} />
      <div className="p-[18px] flex flex-col gap-2 max-h-[420px] overflow-y-auto">
        {(data?.recommendations ?? []).map((r) => (
          <div key={r.id} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
            <div className="flex items-center justify-between gap-2">
              <div className="text-[0.875rem] font-semibold">{r.account} <span className="text-ink-3 text-[0.6875rem]">net {r.net_margin_pct}%</span></div>
              <Badge status="good">{money(r.profit_uplift)}/yr</Badge>
            </div>
            <div className="flex gap-1.5 flex-wrap mt-1.5">
              {Object.entries(r.changes).map(([k, v]) => <Badge key={k} status="neutral">{k.replace(/_/g, " ")} {v}</Badge>)}
            </div>
            <div className="flex items-center justify-between mt-2 text-[0.6875rem] text-ink-3">
              <span>churn risk <b style={{ color: r.churn_risk_pct > 20 ? "var(--warning)" : "var(--good)" }}>{r.churn_risk_pct}%</b> · conf {r.confidence}% · {r.negotiation}</span>
              {done[r.id]
                ? <Badge status={done[r.id] === "REJECTED" ? "critical" : "good"}>{done[r.id].toLowerCase()}</Badge>
                : <div className="flex gap-1"><Button variant="primary" sm onClick={() => act(r.id, r.account, "APPROVED")}>Approve</Button><Button variant="ghost" sm onClick={() => act(r.id, r.account, "SCHEDULED")}>Schedule</Button></div>}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ═══ Customer Risk Scoring ═══ */
function riskCell(v: number) {
  const t = Math.min(1, v / 100);
  const hue = v >= 66 ? "var(--critical)" : v >= 33 ? "var(--warning)" : "var(--good)";
  return `color-mix(in srgb, ${hue} ${Math.round(30 + t * 55)}%, transparent)`;
}
function RiskScoring({ data }: { data: CiRisk | null }) {
  return (
    <Card className="mt-4">
      <CardHead title="Customer Risk Scoring" hint="profitability · payment · expiry · service · volume · concentration" />
      <div className="p-[18px] overflow-x-auto">
        <table className="border-collapse text-[0.75rem]" style={{ minWidth: 720 }}>
          <thead><tr>
            <th className="text-left px-3 py-2 text-[10px] uppercase text-ink-3">Account</th>
            {(data?.dimensions ?? []).map((d) => <th key={d} className="px-2 py-2 text-[10px] uppercase text-ink-3 text-center">{d.replace(/_/g, " ")}</th>)}
            <th className="px-2 py-2 text-[10px] uppercase text-ink-3 text-center">Overall</th>
          </tr></thead>
          <tbody>
            {(data?.rows ?? []).map((r) => (
              <tr key={r.id}>
                <td className="px-3 py-1.5 text-ink font-medium whitespace-nowrap">{r.account} <span className="text-ink-3 text-[0.625rem]">{r.region}</span></td>
                {(data?.dimensions ?? []).map((d) => (
                  <td key={d} className="px-2 py-1.5 text-center tnum" style={{ background: riskCell(r.scores[d]), color: "var(--ink)" }}>{r.scores[d]}</td>
                ))}
                <td className="px-2 py-1.5 text-center"><Badge status={bandColor[r.overall_band as keyof typeof bandColor]}>{r.overall}</Badge></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
