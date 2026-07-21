"use client";
/**
 * AI Digital Workers — the agentic automation cockpit.
 * Each worker is a real Planner capability; the queue shows tasks completed with
 * zero human touch, awaiting approval, or escalated into the Decision Center.
 * Representative metrics (labelled) over the existing agents — no new AI engine.
 */
import { useEffect, useState } from "react";
import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import {
  Card, CardHead, KpiCard, Badge, Progress, DataTable, Th, Td, EmptyState, Alert,
} from "@/components/ui/primitives";
import { api, WorkersResponse } from "@/lib/api";

const compact = (n: number) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);

export default function Workforce() {
  const [d, setD] = useState<WorkersResponse | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => { api.workers().then(setD).catch(() => setErr(true)); }, []);

  const s = d?.summary;

  return (
    <AppShell title="Workforce">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-4">
        <div>
          <div className="eyebrow" style={{ color: "var(--accent)" }}>
            Agentic automation {d?.source === "representative" ? "· representative activity" : ""}
          </div>
          <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">AI Digital Workers</h1>
          <p className="text-ink-2 mt-1.5 text-[0.9375rem] max-w-2xl">
            Autonomous digital workers run the back office end-to-end — completing routine tasks with zero human touch and
            escalating only the exceptions that need a decision.
          </p>
        </div>
        <Link href="/decisions"><Badge status="info">Exceptions → Decision Center</Badge></Link>
      </div>

      {/* ---- Productivity summary ---- */}
      <div className="grid gap-3 mb-6" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))" }}>
        <KpiCard label="Active Workers" value={s ? `${s.active_workers}/${s.total_workers}` : "—"} status="good" seed={3} />
        <KpiCard label="Tasks Automated Today" value={s ? compact(s.tasks_automated_today) : "—"} status="info" seed={6} />
        <KpiCard label="Zero-Touch Rate" value={s?.zero_touch_pct ?? "—"} unit="%" status="good" seed={2} />
        <KpiCard label="Hours Saved / Week" value={s ? compact(s.hours_saved_week) : "—"} status="good" seed={8} />
        <KpiCard label="Awaiting Approval" value={s?.awaiting_approval ?? "—"} status={s && s.awaiting_approval > 0 ? "warning" : "good"} seed={4} />
        <KpiCard label="Escalated" value={s?.escalated ?? "—"} status={s && s.escalated > 0 ? "critical" : "good"} seed={9} />
      </div>

      {err && <Alert status="critical" title="API unreachable">Start the FastAPI backend to load the digital workforce.</Alert>}

      {/* ---- Worker roster ---- */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-[1.25rem] font-semibold tracking-tight">Worker roster</h2>
        <div className="text-[0.75rem] text-ink-3">discovered from registered capabilities</div>
      </div>

      {!d && !err && <Card><EmptyState kind="loading" /></Card>}

      <div className="grid gap-3 mb-8" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(300px,1fr))" }}>
        {(d?.workers ?? []).map((w) => (
          <Card key={w.id} className="p-4 flex flex-col gap-3">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="font-semibold text-[0.9375rem] truncate">{w.name}</div>
                <div className="text-[0.6875rem] text-ink-3 mt-0.5">{w.domain}</div>
              </div>
              <Badge status={w.status === "active" ? "good" : "neutral"}>{w.status === "active" ? "Active" : "Idle"}</Badge>
            </div>
            <p className="text-[0.8125rem] text-ink-2 leading-snug min-h-[2.4em]">{w.skill}</p>
            <div>
              <div className="flex justify-between text-[0.6875rem] text-ink-3 mb-1">
                <span>Zero-touch autonomy</span><span className="tnum" style={{ color: "var(--good)" }}>{w.zero_touch_pct}%</span>
              </div>
              <Progress value={w.zero_touch_pct} status="good" />
            </div>
            <div className="flex items-center gap-4 text-[0.75rem] text-ink-3">
              <span><b className="text-ink tnum">{w.tasks_today}</b> tasks today</span>
              <span className={w.exceptions > 0 ? "" : "opacity-60"}>
                <b className="tnum" style={{ color: w.exceptions > 0 ? "var(--warning)" : "var(--text-3)" }}>{w.exceptions}</b> exceptions
              </span>
            </div>
          </Card>
        ))}
      </div>

      {/* ---- Live task queue ---- */}
      <Card>
        <CardHead title="Live task queue" hint="most recent first"
          right={s && <span className="text-[0.75rem] text-ink-3">{s.awaiting_approval + s.escalated} need a human</span>} />
        <DataTable head={<>
          <Th>Task</Th><Th>Worker</Th><Th>Status</Th><Th num>Confidence</Th><Th num>Impact</Th><Th num>When</Th>
        </>}>
          {(d?.queue ?? []).map((t) => (
            <tr key={t.id} className="hover:bg-[color-mix(in_srgb,var(--accent)_6%,transparent)]">
              <Td strong>{t.task}<div className="text-[0.6875rem] text-ink-3 font-normal font-mono">{t.id} · {t.domain}</div></Td>
              <Td>{t.worker}</Td>
              <Td><Badge status={t.state_status as any}>{t.state_label}</Badge></Td>
              <Td num>{t.confidence}%</Td>
              <Td num>${compact(t.impact_usd)}</Td>
              <Td num>{t.minutes_ago}m ago</Td>
            </tr>
          ))}
          {d && d.queue.length === 0 && !err && <tr><Td>No tasks in the queue.</Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td></tr>}
        </DataTable>
        <div className="px-[18px] py-3 text-[0.6875rem] text-ink-3 border-t" style={{ borderColor: "var(--hairline)" }}>
          Auto-completed tasks post straight to the audit trail. <b style={{ color: "var(--accent)" }}>Awaiting approval</b> and{" "}
          <b style={{ color: "var(--critical)" }}>escalated</b> items route to the{" "}
          <Link href="/decisions" className="underline">Decision Center</Link> for a human call — the same approval + audit
          layer already in the platform.
        </div>
      </Card>
    </AppShell>
  );
}
