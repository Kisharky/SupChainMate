"use client";
/**
 * Fraud & Anomaly Detection — the enterprise trust console.
 * Duplicate invoices, double-brokering, payment/price anomalies, and carrier/
 * supplier identity risk. Detected anomalies escalate into the Decision Center.
 * Representative signals (labelled) over the existing audit + Commercial layers.
 */
import { useEffect, useState } from "react";
import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import {
  Card, CardHead, KpiCard, Badge, Button, Progress, DataTable, Th, Td, EmptyState, Alert,
} from "@/components/ui/primitives";
import { api, FraudResponse } from "@/lib/api";

const compact = (n: number) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);

export default function Fraud() {
  const [d, setD] = useState<FraudResponse | null>(null);
  const [err, setErr] = useState(false);
  const [sent, setSent] = useState<Record<string, boolean>>({});
  useEffect(() => { api.fraud().then(setD).catch(() => setErr(true)); }, []);

  const s = d?.summary;

  return (
    <AppShell title="Fraud & Risk">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-4">
        <div>
          <div className="eyebrow" style={{ color: "var(--accent)" }}>
            Trust & compliance {d?.source === "representative" ? "· representative signals" : ""}
          </div>
          <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Fraud &amp; Anomaly Detection</h1>
          <p className="text-ink-2 mt-1.5 text-[0.9375rem] max-w-2xl">
            Continuously scans invoices, payments, and load documents for duplicates, double-brokering, and identity
            risk — surfacing the money at risk before it leaves the door.
          </p>
        </div>
        <Link href="/decisions"><Badge status="info">Cases → Decision Center</Badge></Link>
      </div>

      {/* ---- Summary ---- */}
      <div className="grid gap-3 mb-6" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))" }}>
        <KpiCard label="Open Alerts" value={s?.open_alerts ?? "—"} status={s && s.open_alerts > 0 ? "warning" : "good"} seed={3} />
        <KpiCard label="High Severity" value={s?.high_severity ?? "—"} status={s && s.high_severity > 0 ? "critical" : "good"} seed={9} />
        <KpiCard label="Amount at Risk" prefix="$" value={s ? compact(s.amount_at_risk) : "—"} status="warning" seed={5} />
        <KpiCard label="Entities Flagged" value={s?.entities_flagged ?? "—"} status={s && s.entities_flagged > 0 ? "warning" : "good"} seed={7} />
        <KpiCard label="Duplicate Invoices" value={s?.duplicate_invoices ?? "—"} status={s && s.duplicate_invoices > 0 ? "warning" : "good"} seed={2} />
        <KpiCard label="Detection Accuracy" value={s?.detection_accuracy ?? "—"} unit="%" status="good" seed={8} />
      </div>

      {err && <Alert status="critical" title="API unreachable">Start the FastAPI backend to load the fraud console.</Alert>}

      {/* ---- Detection coverage ---- */}
      <Card className="mb-6">
        <CardHead title="Detection coverage" hint="active detectors" />
        <div className="p-[18px] grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))" }}>
          {!d ? <EmptyState kind="loading" /> : d.checks.map((c) => (
            <div key={c.name} className="rounded border p-3" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <span className="text-[0.8125rem] font-medium">{c.name}</span>
                <Badge status={c.status as any}>{c.coverage}%</Badge>
              </div>
              <Progress value={c.coverage} status={c.status as any} />
            </div>
          ))}
        </div>
      </Card>

      <div className="grid gap-4 items-start" style={{ gridTemplateColumns: "1.5fr 1fr" }}>
        {/* ---- Alert feed ---- */}
        <div>
          <h2 className="text-[1.25rem] font-semibold tracking-tight mb-3">Anomaly alerts</h2>
          {!d && !err && <Card><EmptyState kind="loading" /></Card>}
          <div className="flex flex-col gap-3">
            {(d?.alerts ?? []).map((a) => (
              <Card key={a.id} className="p-4">
                <div className="flex items-start gap-3">
                  <span className="grid h-9 w-9 place-items-center rounded-md text-[16px] flex-none"
                    style={{ background: "var(--panel-2)", border: "1px solid var(--hairline)", color: `var(--${a.severity_status})` }}>{a.icon}</span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge status={a.severity_status as any}>{a.severity}</Badge>
                      <span className="text-[0.75rem] text-ink-3">{a.type_label}</span>
                      <span className="text-[0.75rem] text-ink-3">· {a.entity}</span>
                      <span className="ml-auto font-mono text-[0.6875rem] text-ink-3">{a.id} · {a.hours_ago}h ago</span>
                    </div>
                    <p className="text-[0.875rem] text-ink mt-1.5">{a.detail}</p>
                    <div className="flex items-center gap-4 mt-2 text-[0.75rem] text-ink-3 flex-wrap">
                      {a.amount_at_risk > 0 && <span>At risk <b className="text-warning tnum">${a.amount_at_risk.toLocaleString()}</b></span>}
                      <span>Confidence <b className="text-ink tnum">{a.confidence}%</b></span>
                      <span>Status <b className="text-ink">{a.status}</b></span>
                    </div>
                    <div className="flex items-center gap-2 mt-3 flex-wrap">
                      <span className="text-[0.75rem] text-ink-2 flex-1 min-w-[160px]">
                        <span className="text-ink-3">Recommended:</span> {a.recommended_action}
                      </span>
                      {sent[a.id]
                        ? <Badge status="good">Sent to Decision Center</Badge>
                        : <Button sm variant="secondary" onClick={() => setSent((m) => ({ ...m, [a.id]: true }))}>Send to Decision Center</Button>}
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* ---- Entity risk register ---- */}
        <Card>
          <CardHead title="Entity risk register" hint="carriers & suppliers" />
          <DataTable head={<><Th>Entity</Th><Th num>Risk</Th><Th>Tier</Th></>}>
            {(d?.entities ?? []).map((e) => (
              <tr key={e.name}>
                <Td strong>{e.name}
                  <div className="text-[0.6875rem] text-ink-3 font-normal">{e.kind} · {e.top_factor}</div>
                </Td>
                <Td num>
                  <span className="inline-flex items-center gap-2 justify-end">
                    <span className="tnum font-semibold" style={{ color: `var(--${e.tier_status})` }}>{e.risk_score}</span>
                    <span className="h-1.5 w-10 rounded-full overflow-hidden inline-block" style={{ background: "var(--hairline)" }}>
                      <span className="block h-full rounded-full" style={{ width: `${e.risk_score}%`, background: `var(--${e.tier_status})` }} />
                    </span>
                  </span>
                </Td>
                <Td><Badge status={e.tier_status as any}>{e.tier}</Badge></Td>
              </tr>
            ))}
            {d && d.entities.length === 0 && !err && <tr><Td>No entities flagged.</Td><Td> </Td><Td> </Td></tr>}
          </DataTable>
          <div className="px-[18px] py-3 text-[0.6875rem] text-ink-3 border-t" style={{ borderColor: "var(--hairline)" }}>
            Scores blend document consistency, identity age, and payment behaviour. High-risk entities are held pending
            human verification via the <Link href="/decisions" className="underline">Decision Center</Link>.
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
