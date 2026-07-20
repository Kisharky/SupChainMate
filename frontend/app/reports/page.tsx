"use client";
/** Executive Reports — report library + exports. */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, Button, Badge, Sparkline } from "@/components/ui/primitives";
import { api, ReportItem } from "@/lib/api";

export default function Reports() {
  const [reports, setReports] = useState<ReportItem[]>([]);
  useEffect(() => { api.reports().then((r) => setReports(r.reports)).catch(() => setReports([])); }, []);

  return (
    <AppShell title="Executive Reports">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-4">
        <div>
          <div className="text-[0.75rem] uppercase tracking-[.16em] font-semibold" style={{ color: "var(--accent)" }}>Generated · exportable to PDF</div>
          <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Executive Reports</h1>
          <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Board-ready summaries, trend packs, and audit exports synthesised by the Executive agent.</p>
        </div>
        <Button variant="primary" sm>⇩ New report</Button>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))" }}>
        {reports.map((r, i) => (
          <Card key={r.id} className="overflow-hidden flex flex-col">
            <div className="h-[120px] relative border-b" style={{ borderColor: "var(--hairline)", background: "linear-gradient(135deg,var(--navy-700),var(--navy-850))" }}>
              <div className="absolute inset-0 opacity-80"><Sparkline seed={(i + 2) * 3} color="var(--accent)" w={260} h={120} trend="up" /></div>
              <div className="absolute top-3 left-3"><Badge status={r.status === "ready" ? "good" : "warning"}>{r.status}</Badge></div>
            </div>
            <div className="p-[18px]">
              <div className="text-[0.8125rem] text-ink font-semibold">{r.title}</div>
              <div className="text-[0.75rem] text-ink-3 my-1 mb-3">{r.subtitle}</div>
              <div className="flex gap-2.5">
                <Button variant={r.status === "ready" ? "primary" : "secondary"} sm>{r.status === "ready" ? "⇩ PDF" : "Generate"}</Button>
                <Button variant="ghost" sm>Preview</Button>
              </div>
            </div>
          </Card>
        ))}
        {reports.length === 0 && <p className="text-ink-3">Loading reports…</p>}
      </div>
    </AppShell>
  );
}
