"use client";
/**
 * Decision Center — the trust layer. Every recommendation carries its evidence,
 * confidence, and business impact, and every human decision (approve / reject /
 * modify / escalate) is written to an immutable audit trail. This is what makes
 * SupChainMate a decision platform, not a dashboard.
 */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, Button, Badge, DataTable, Th, Td } from "@/components/ui/primitives";
import { api, DecisionsResponse, Recommendation, AuditEntry, DecisionStatus } from "@/lib/api";

const confColor = (c: number) => (c >= 80 ? "good" : c >= 55 ? "warning" : "critical");
const statusBadge: Record<string, "good" | "warning" | "critical" | "info" | "neutral"> = {
  APPROVED: "good", MODIFIED: "info", REJECTED: "critical", ESCALATED: "warning", PENDING: "neutral",
};

export default function Decisions() {
  const [data, setData] = useState<DecisionsResponse | null>(null);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [modifyKey, setModifyKey] = useState<string | null>(null);
  const [note, setNote] = useState("");

  const load = () => {
    api.decisions().then(setData).catch(() => setData(null));
    api.audit().then((r) => setAudit(r.entries)).catch(() => setAudit([]));
  };
  useEffect(load, []);

  const act = async (rec: Recommendation, status: DecisionStatus) => {
    setBusy(rec.rec_key);
    try {
      await api.decide(rec.rec_key, status, status === "MODIFIED" ? note : "");
      setModifyKey(null); setNote("");
      load();
    } finally { setBusy(null); }
  };

  const k = data?.kpis;

  return (
    <AppShell title="Decision Center">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-4">
        <div>
          <div className="eyebrow" style={{ color: "var(--accent)" }}>Trust layer · human-in-the-loop · audited</div>
          <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Decision Center</h1>
          <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Every recommendation shows its evidence, confidence, and impact — and every decision is logged.</p>
        </div>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))" }}>
        {[
          { l: "Pending", v: k?.pending ?? "—", s: "warning" },
          { l: "Approved", v: k?.approved ?? "—", s: "good" },
          { l: "Rejected", v: k?.rejected ?? "—", s: "critical" },
          { l: "Approved Savings/yr", v: k?.approved_savings != null ? `$${Math.round(k.approved_savings).toLocaleString()}` : "—", s: "good" },
          { l: "Avg Confidence", v: k?.avg_confidence != null ? `${k.avg_confidence}%` : "—", s: "info" },
        ].map((m) => (
          <Card key={m.l} className="p-4 relative overflow-hidden">
            <span className="absolute left-0 top-0 bottom-0 w-[3px]" style={{ background: `var(--${m.s})` }} />
            <div className="eyebrow">{m.l}</div>
            <div className="text-[1.75rem] font-bold tnum mt-1">{m.v}</div>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 mt-4 items-start" style={{ gridTemplateColumns: "1.6fr 1fr" }}>
        {/* Pending decisions */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h2 className="text-[1.125rem] font-semibold">Pending decisions</h2>
            <span className="text-[0.75rem] text-ink-3">{data?.pending.length ?? 0} awaiting review</span>
          </div>
          {(data?.pending ?? []).map((rec) => (
            <Card key={rec.rec_key} className="overflow-hidden">
              <div className="p-[18px]">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="eyebrow" style={{ color: "var(--accent)" }}>{rec.category} · {rec.source}</div>
                    <div className="text-[1.0625rem] font-semibold mt-1">{rec.title}</div>
                  </div>
                  <Badge status={confColor(rec.confidence)}>{rec.confidence.toFixed(0)}% confidence</Badge>
                </div>
                <p className="text-[0.875rem] text-ink-2 mt-2">{rec.action}</p>

                {/* Business impact */}
                <div className="flex gap-2 flex-wrap mt-3">
                  {rec.impact?.cost_savings_yr ? <Badge status="good">↑ ${Math.round(rec.impact.cost_savings_yr).toLocaleString()}/yr</Badge> : null}
                  {rec.impact?.stockout_risk_pct != null ? <Badge status="warning">stockout {rec.impact.stockout_risk_pct}%</Badge> : null}
                  {rec.impact?.service_level_pct != null ? <Badge status="info">service {rec.impact.service_level_pct}%</Badge> : null}
                  {rec.impact?.other ? <Badge status="neutral">{rec.impact.other}</Badge> : null}
                </div>

                {/* Evidence */}
                <div className="mt-3 pt-3 border-t" style={{ borderColor: "var(--hairline)" }}>
                  <div className="eyebrow mb-1.5">Evidence · {rec.confidence_basis}</div>
                  <div className="flex flex-col gap-1">
                    {rec.drivers.map((d, i) => (
                      <div key={i} className="flex gap-2 text-[0.8125rem]">
                        <span style={{ color: "var(--accent)" }}>›</span>
                        <span className="text-ink-2"><b className="text-ink">{d.reason}:</b> {d.evidence}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Actions */}
                {modifyKey === rec.rec_key ? (
                  <div className="mt-3 flex gap-2 items-center">
                    <input autoFocus value={note} onChange={(e) => setNote(e.target.value)} placeholder="Modification note…"
                      className="flex-1 bg-[var(--bg-sunken)] border rounded-sm px-3 py-2 text-[0.8125rem] text-ink outline-none"
                      style={{ borderColor: "var(--hairline-strong)" }} />
                    <Button variant="primary" sm onClick={() => act(rec, "MODIFIED")} disabled={busy === rec.rec_key}>Save</Button>
                    <Button variant="ghost" sm onClick={() => { setModifyKey(null); setNote(""); }}>Cancel</Button>
                  </div>
                ) : (
                  <div className="mt-3 flex gap-2 flex-wrap">
                    <Button variant="primary" sm onClick={() => act(rec, "APPROVED")} disabled={busy === rec.rec_key}>✓ Approve</Button>
                    <Button variant="danger" sm onClick={() => act(rec, "REJECTED")} disabled={busy === rec.rec_key}>✕ Reject</Button>
                    <Button variant="secondary" sm onClick={() => { setModifyKey(rec.rec_key); setNote(""); }}>✎ Modify</Button>
                    <Button variant="ghost" sm onClick={() => act(rec, "ESCALATED")} disabled={busy === rec.rec_key}>↑ Escalate</Button>
                  </div>
                )}
              </div>
            </Card>
          ))}
          {data && data.pending.length === 0 && (
            <Card className="p-8 text-center text-ink-3 text-[0.9375rem]">All caught up — no pending decisions.</Card>
          )}
        </div>

        {/* History + audit */}
        <div className="flex flex-col gap-4">
          <Card>
            <CardHead title="Decision history" hint={`${data?.history.length ?? 0} decided`} />
            <div className="px-[18px] py-1.5 max-h-[280px] overflow-y-auto">
              {(data?.history ?? []).map((r) => (
                <div key={r.rec_key} className="flex gap-3 py-2.5 border-b last:border-0" style={{ borderColor: "var(--hairline)" }}>
                  <div className="min-w-0">
                    <div className="text-[0.8125rem] text-ink font-medium truncate">{r.title}</div>
                    <div className="text-[0.75rem] text-ink-3">{r.decided_by ?? "—"}{r.note ? ` · ${r.note}` : ""}</div>
                  </div>
                  <span className="ml-auto"><Badge status={statusBadge[r.status] ?? "neutral"}>{r.status}</Badge></span>
                </div>
              ))}
              {data && data.history.length === 0 && <p className="text-ink-3 text-[0.8125rem] py-3">No decisions yet.</p>}
            </div>
          </Card>
          <Card>
            <CardHead title="Audit trail" hint="immutable" />
            <div className="px-[18px] py-1.5 max-h-[280px] overflow-y-auto">
              {audit.slice(0, 30).map((e, i) => (
                <div key={i} className="flex gap-3 py-2 border-b last:border-0 text-[0.75rem]" style={{ borderColor: "var(--hairline)" }}>
                  <span className="font-mono text-ink-3 whitespace-nowrap">{e.ts?.slice(5, 16)}</span>
                  <span className="text-ink-2"><b className="text-ink">{e.actor}</b> · {e.event.replace(/_/g, " ")}{e.details ? ` — ${e.details}` : ""}</span>
                </div>
              ))}
              {audit.length === 0 && <p className="text-ink-3 text-[0.8125rem] py-3">No audit entries yet.</p>}
            </div>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
