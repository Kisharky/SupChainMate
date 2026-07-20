"use client";
/** Procurement — carrier allocation optimisation (live). */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, KpiCard, DataTable, Th, Td, Button } from "@/components/ui/primitives";
import { api, ProcurementResponse } from "@/lib/api";

export default function Procurement() {
  const [data, setData] = useState<ProcurementResponse | null>(null);
  useEffect(() => { api.procurement().then(setData).catch(() => setData(null)); }, []);
  const im = data?.impact ?? {};
  const savings = im["savings_total"];
  const otNow = im["on_time_current"]; const otRec = im["on_time_recommended"];

  return (
    <AppShell title="Procurement">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-4">
        <div>
          <div className="eyebrow" style={{ color: "var(--accent)" }}>Strategic sourcing · {data?.source === "live" ? "live allocation" : "loading"}</div>
          <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Procurement Intelligence</h1>
          <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Carrier allocation optimised on cost and on-time performance from the live scorecard.</p>
        </div>
        <Button variant="primary" sm>✓ Approve allocation shift</Button>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))" }}>
        <KpiCard label="Projected Savings" value={savings ? `${Math.round(savings / 1000)}` : "—"} unit="k" prefix="$" status="good" seed={3} />
        <KpiCard label="On-Time (now)" value={otNow ? otNow.toFixed(1) : "—"} unit="%" status="good" seed={7} />
        <KpiCard label="On-Time (optimised)" value={otRec ? otRec.toFixed(1) : "—"} unit="%" status="good" seed={9} delta={otNow && otRec ? Number((otRec - otNow).toFixed(2)) : undefined} />
        <KpiCard label="Volume Shift" value={im["total_shift_pts"] ? im["total_shift_pts"].toFixed(1) : "—"} unit="pts" status="info" seed={5} />
      </div>

      <Card className="mt-4">
        <CardHead title="Carrier allocation recommendation" hint="current → recommended share of volume" />
        <DataTable head={<><Th>Carrier</Th><Th num>Score</Th><Th num>On-time</Th><Th num>Current share</Th><Th num>Recommended</Th></>}>
          {(data?.carriers ?? []).map((c) => (
            <tr key={c.carrier}>
              <Td strong>{c.carrier}</Td>
              <Td num>{c.score.toFixed(1)}</Td>
              <Td num>{c.on_time == null ? "—" : `${c.on_time.toFixed(1)}%`}</Td>
              <Td num>{c.current_share == null ? "—" : `${c.current_share.toFixed(1)}%`}</Td>
              <Td num strong>{c.recommended_share == null ? "—" : `${c.recommended_share.toFixed(1)}%`}</Td>
            </tr>
          ))}
          {(!data?.carriers || data.carriers.length === 0) && <tr><Td>Loading…</Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td></tr>}
        </DataTable>
      </Card>
    </AppShell>
  );
}
