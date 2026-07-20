"use client";
/** Forecasting — Prophet demand forecast (live). */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, KpiCard } from "@/components/ui/primitives";
import { ForecastChart } from "@/components/ForecastChart";
import { api, ForecastResponse } from "@/lib/api";

export default function Forecasting() {
  const [data, setData] = useState<ForecastResponse | null>(null);
  useEffect(() => { api.forecast().then(setData).catch(() => setData(null)); }, []);
  const ins = data?.insights;
  const pct = ins?.demand_pct_change_vs_prior_week;

  return (
    <AppShell title="Forecasting">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>Prophet · {data?.source === "live" ? "live" : "loading"} · 14-day horizon</div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Demand Forecasting</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Order-volume forecast with a confidence band, computed from real order history.</p>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))" }}>
        <KpiCard label="Next 7d Demand" value={ins?.next_week_total?.toLocaleString() ?? "—"} status="info" seed={3} />
        <KpiCard label="WoW Change" value={pct == null ? "—" : pct.toFixed(1)} unit="%" delta={pct ? Number(pct.toFixed(1)) : undefined} status={pct && pct < 0 ? "warning" : "good"} seed={5} />
        <KpiCard label="Stockout Risk" value={ins?.stockout_risk_short ?? "—"} status="good" seed={7} />
        <KpiCard label="P90 Daily" value={ins?.historical_p90_daily?.toLocaleString() ?? "—"} status="good" seed={6} />
      </div>

      <Card className="mt-4">
        <CardHead title="Order volume — history & forecast" hint="emerald = actual · dashed = forecast · band = interval" />
        <div className="p-[18px]">
          {data ? <ForecastChart history={data.history} forecast={data.forecast} /> : <div className="h-[220px] grid place-items-center text-ink-3 text-[0.8125rem]">Loading forecast…</div>}
          {ins?.stockout_risk_detail && <p className="text-[0.8125rem] text-ink-2 mt-3">{ins.stockout_risk_detail}</p>}
        </div>
      </Card>
    </AppShell>
  );
}
