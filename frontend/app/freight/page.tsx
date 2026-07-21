"use client";
/**
 * Freight Operations — the brokerage back office: carrier vetting & onboarding,
 * load ↔ carrier matching, instant spot quoting, and inbound email triage.
 * Execution-oriented workflows that sit alongside the shipper-side intelligence.
 * Representative signals (labelled) over the existing layers; infra-heavy items
 * (voice, appointments, EDI) are shown as roadmap, not faked.
 */
import { useEffect, useState } from "react";
import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import {
  Card, CardHead, KpiCard, Badge, Button, Modal, DataTable, Th, Td, EmptyState, Alert,
} from "@/components/ui/primitives";
import { api, FreightResponse, CarrierRow, CarrierDetail, QuoteResult } from "@/lib/api";

const money = (n: number) => `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

export default function Freight() {
  const [d, setD] = useState<FreightResponse | null>(null);
  const [err, setErr] = useState(false);
  const [carrier, setCarrier] = useState<CarrierRow | null>(null);
  useEffect(() => { api.freight().then(setD).catch(() => setErr(true)); }, []);
  const s = d?.summary;

  return (
    <AppShell title="Freight Ops">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>
          Brokerage back office {d?.source === "representative" ? "· representative operations" : ""}
        </div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Freight Operations</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem] max-w-2xl">
          Vet and onboard carriers, match loads to capacity, quote spot lanes, and triage the inbox — the
          execution workflows a digital worker runs end-to-end.
        </p>
      </div>

      {/* ---- Summary ---- */}
      <div className="grid gap-3 mb-6" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))" }}>
        <KpiCard label="Carriers Onboarded" value={s?.carriers_onboarded ?? "—"} status="good" seed={3} />
        <KpiCard label="Pending Vetting" value={s?.pending_vetting ?? "—"} status={s && s.pending_vetting > 0 ? "warning" : "good"} seed={6} />
        <KpiCard label="High-Risk Carriers" value={s?.high_risk_carriers ?? "—"} status={s && s.high_risk_carriers > 0 ? "critical" : "good"} seed={9} />
        <KpiCard label="Open Loads" value={s?.open_loads ?? "—"} status="info" seed={2} />
        <KpiCard label="Inbox to Triage" value={s?.triage_queue ?? "—"} status="info" seed={5} />
        <KpiCard label="Open Claims" value={s?.open_claims ?? "—"} status={s && s.open_claims > 0 ? "warning" : "good"} seed={8} />
      </div>

      {err && <Alert status="critical" title="API unreachable">Start the FastAPI backend to load freight operations.</Alert>}

      {/* ---- Carrier vetting & onboarding ---- */}
      <Card className="mb-6">
        <CardHead title="Carrier vetting & onboarding" hint="FMCSA authority · insurance · fraud signals" />
        <DataTable head={<>
          <Th>Carrier</Th><Th>Authority</Th><Th>Insurance</Th><Th>Stage</Th>
          <Th num>Risk</Th><Th>Recommendation</Th><Th>{" "}</Th>
        </>}>
          {(d?.carriers ?? []).map((c) => (
            <tr key={c.id} className="hover:bg-[color-mix(in_srgb,var(--accent)_6%,transparent)]">
              <Td strong>{c.name}<div className="text-[0.6875rem] text-ink-3 font-normal font-mono">{c.mc_number} · {c.dot_number}</div></Td>
              <Td>
                <Badge status={c.authority_status === "active" ? "good" : "warning"}>{c.authority_status}</Badge>
                <span className="text-[0.6875rem] text-ink-3 ml-1.5">{c.authority_age_days}d</span>
              </Td>
              <Td>
                <Badge status={c.insurance_status === "valid" ? "good" : c.insurance_status === "expiring" ? "warning" : "critical"}>{c.insurance_status}</Badge>
              </Td>
              <Td><span className="text-[0.8125rem] capitalize">{c.stage}</span></Td>
              <Td num>
                <span className="tnum font-semibold" style={{ color: `var(--${c.risk_status})` }}>{c.risk_score}</span>
                {c.flag_count > 0 && <span className="text-[0.6875rem] text-ink-3 ml-1">· {c.flag_count}⚑</span>}
              </Td>
              <Td><span className="text-[0.75rem]" style={{ color: c.risk_severity === "high" ? "var(--critical)" : "var(--text-2)" }}>{c.recommendation}</span></Td>
              <Td><Button sm variant="secondary" onClick={() => setCarrier(c)}>Vet</Button></Td>
            </tr>
          ))}
          {!d && !err && <tr><Td>Loading…</Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td></tr>}
        </DataTable>
      </Card>

      <div className="grid gap-4 items-start mb-6" style={{ gridTemplateColumns: "1.4fr 1fr" }}>
        {/* ---- Load ↔ carrier matching ---- */}
        <Card>
          <CardHead title="Load ↔ carrier matching" hint="ranked by lane fit" />
          <div className="p-[18px] flex flex-col gap-3">
            {!d ? <EmptyState kind="loading" /> : d.loads.map((l) => (
              <div key={l.id} className="rounded border p-3" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <div>
                    <span className="font-semibold text-[0.875rem]">{l.origin} → {l.destination}</span>
                    <span className="text-[0.6875rem] text-ink-3 ml-2 font-mono">{l.id}</span>
                  </div>
                  <span className="text-[0.6875rem] text-ink-3">{l.equipment} · {l.miles} mi · {l.pickup}</span>
                </div>
                <div className="mt-2 flex flex-col gap-1">
                  {l.matches.map((m, i) => (
                    <div key={m.carrier_id} className="flex items-center gap-2 text-[0.75rem]">
                      <span className="w-4 text-ink-3">{i + 1}.</span>
                      <span className="text-ink font-medium flex-1 truncate">{m.carrier}</span>
                      <span className="text-ink-3">{m.on_time_pct}% OT · {m.trucks_available} trucks · {m.lane_loads} lane</span>
                      <Badge status={m.fit_score >= 70 ? "good" : m.fit_score >= 50 ? "info" : "warning"}>{m.fit_score}</Badge>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* ---- Instant spot quote ---- */}
        <QuoteCard />
      </div>

      {/* ---- Inbound email triage ---- */}
      <Card className="mb-6">
        <CardHead title="Inbound email triage" hint="classified · most recent first" />
        <DataTable head={<><Th>From / subject</Th><Th>Type</Th><Th num>Conf.</Th><Th>Suggested action</Th><Th num>When</Th></>}>
          {(d?.triage ?? []).map((e) => (
            <tr key={e.id}>
              <Td strong>{e.subject}<div className="text-[0.6875rem] text-ink-3 font-normal">{e.from}
                {Object.keys(e.extracted).length > 0 && <span className="ml-1">· {Object.entries(e.extracted).map(([k, v]) => `${k}: ${v}`).join(" · ")}</span>}
              </div></Td>
              <Td><Badge status={e.type_status as any}>{e.type_label}</Badge></Td>
              <Td num>{e.confidence}%</Td>
              <Td><span className="text-[0.8125rem] text-ink-2">{e.suggested_action}</span></Td>
              <Td num>{e.minutes_ago}m</Td>
            </tr>
          ))}
          {!d && !err && <tr><Td>Loading…</Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td></tr>}
        </DataTable>
      </Card>

      {/* ---- Roadmap ---- */}
      <Card>
        <CardHead title="On the roadmap" hint="infra-heavy — not simulated" />
        <div className="p-[18px] grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))" }}>
          {(d?.roadmap ?? []).map((r) => (
            <div key={r.name} className="rounded border p-3 opacity-80" style={{ borderColor: "var(--hairline)", borderStyle: "dashed" }}>
              <div className="text-[0.8125rem] font-semibold">{r.name}</div>
              <div className="text-[0.6875rem] text-ink-3 mt-1">{r.detail}</div>
            </div>
          ))}
        </div>
        <div className="px-[18px] pb-3 text-[0.6875rem] text-ink-3">
          Voice, appointment, CRM, factoring, and EDI workflows need real telephony/partner integrations — declared
          here rather than mocked. Accessorial recovery and freight audit already run in{" "}
          <Link href="/fraud" className="underline">Fraud &amp; Risk</Link> and{" "}
          <Link href="/documents" className="underline">Documents</Link>.
        </div>
      </Card>

      <CarrierModal carrier={carrier} onClose={() => setCarrier(null)} />
    </AppShell>
  );
}

/* ---- Carrier vetting modal ---- */
function CarrierModal({ carrier, onClose }: { carrier: CarrierRow | null; onClose: () => void }) {
  const [det, setDet] = useState<CarrierDetail | null>(null);
  useEffect(() => {
    setDet(null);
    if (carrier) api.freightCarrier(carrier.id).then(setDet).catch(() => setDet(null));
  }, [carrier]);
  if (!carrier) return null;

  return (
    <Modal open={!!carrier} onClose={onClose}
      title={`Vet · ${carrier.name}`} subtitle={`${carrier.mc_number} · ${carrier.dot_number}`}
      footer={<>
        <Button variant="danger" sm onClick={onClose}>Reject</Button>
        <Button variant="secondary" sm onClick={onClose}>Request docs</Button>
        <Button variant="primary" sm onClick={onClose}>{carrier.risk_severity === "high" ? "Override & approve" : "Approve carrier"}</Button>
      </>}>
      {!det ? <EmptyState kind="loading" /> : !det.ok ? <EmptyState kind="error" title="Couldn't load carrier" /> : (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <Badge status={det.risk_status as any}>risk {det.risk_score}</Badge>
            <span className="text-[0.8125rem]" style={{ color: carrier.risk_severity === "high" ? "var(--critical)" : "var(--text-2)" }}>{det.recommendation}</span>
          </div>
          <div>
            <div className="eyebrow mb-1.5">Verification checklist</div>
            <div className="flex flex-col gap-1.5">
              {det.checks.map((c) => (
                <div key={c.name} className="flex items-center gap-2.5 rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                  <span className="text-[15px] flex-none" style={{ color: c.ok ? "var(--good)" : "var(--critical)" }}>{c.ok ? "✓" : "✕"}</span>
                  <span className="text-[0.8125rem] flex-1">{c.name}</span>
                  <span className="text-[0.75rem] text-ink-3">{c.detail}</span>
                </div>
              ))}
            </div>
          </div>
          {det.flags.length > 0 && (
            <Alert status="warning" title={`${det.flags.length} fraud / identity signal(s)`}>
              <ul className="list-disc ml-4 mt-1">{det.flags.map((f) => <li key={f.code}>{f.label}</li>)}</ul>
            </Alert>
          )}
        </div>
      )}
    </Modal>
  );
}

/* ---- Instant spot quote ---- */
const EQUIP = ["Dry Van", "Reefer", "Flatbed", "Expedited"];
function QuoteCard() {
  const [origin, setOrigin] = useState("São Paulo, SP");
  const [dest, setDest] = useState("Rio de Janeiro, RJ");
  const [equip, setEquip] = useState("Dry Van");
  const [q, setQ] = useState<QuoteResult | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    try { setQ(await api.freightQuote(origin, dest, equip)); } catch { /* representative */ }
    finally { setBusy(false); }
  };

  return (
    <Card>
      <CardHead title="Instant spot quote" hint="transparent rate build" />
      <div className="p-[18px] flex flex-col gap-3">
        <div className="grid gap-2" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <label className="flex flex-col gap-1"><span className="text-[0.6875rem] text-ink-3">Origin</span>
            <input value={origin} onChange={(e) => setOrigin(e.target.value)}
              className="rounded-sm border bg-[var(--panel-2)] px-2.5 py-1.5 text-[0.8125rem] text-ink" style={{ borderColor: "var(--hairline-strong)" }} /></label>
          <label className="flex flex-col gap-1"><span className="text-[0.6875rem] text-ink-3">Destination</span>
            <input value={dest} onChange={(e) => setDest(e.target.value)}
              className="rounded-sm border bg-[var(--panel-2)] px-2.5 py-1.5 text-[0.8125rem] text-ink" style={{ borderColor: "var(--hairline-strong)" }} /></label>
        </div>
        <div className="flex items-end gap-2">
          <label className="flex flex-col gap-1 flex-1"><span className="text-[0.6875rem] text-ink-3">Equipment</span>
            <select value={equip} onChange={(e) => setEquip(e.target.value)}
              className="rounded-sm border bg-[var(--panel-2)] px-2.5 py-1.5 text-[0.8125rem] text-ink" style={{ borderColor: "var(--hairline-strong)" }}>
              {EQUIP.map((x) => <option key={x}>{x}</option>)}
            </select></label>
          <Button variant="primary" sm onClick={run} disabled={busy}>{busy ? "Quoting…" : "Get quote"}</Button>
        </div>

        {q && (
          <div className="rounded border p-3 mt-1" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
            <div className="flex items-baseline justify-between mb-2">
              <span className="text-[0.75rem] text-ink-3">{q.miles} mi · {q.transit_days}d transit</span>
              <span className="text-[1.5rem] font-bold tnum" style={{ color: "var(--good)" }}>{money(q.all_in_rate)}</span>
            </div>
            {q.breakdown.map((b) => (
              <div key={b.label} className="flex justify-between text-[0.75rem] py-0.5">
                <span className="text-ink-2">{b.label} <span className="text-ink-3">· {b.basis}</span></span>
                <span className="tnum text-ink">{money(b.amount)}</span>
              </div>
            ))}
            <div className="flex justify-between text-[0.75rem] pt-1.5 mt-1.5 border-t" style={{ borderColor: "var(--hairline)" }}>
              <span className="text-ink-3">Carrier cost</span><span className="tnum">{money(q.carrier_cost)}</span>
            </div>
            <div className="flex justify-between text-[0.75rem] font-semibold">
              <span style={{ color: "var(--accent)" }}>Margin ({q.margin_pct}%)</span>
              <span className="tnum" style={{ color: "var(--accent)" }}>{money(q.margin_usd)}</span>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
