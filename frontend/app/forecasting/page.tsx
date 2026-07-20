"use client";
/** Forecasting — Prophet demand forecast (live). */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, KpiCard } from "@/components/ui/primitives";
import { ForecastChart } from "@/components/ForecastChart";
import { api, ForecastResponse, BacktestResponse } from "@/lib/api";

export default function Forecasting() {
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [bt, setBt] = useState<BacktestResponse | null>(null);
  useEffect(() => {
    api.forecast().then(setData).catch(() => setData(null));
    api.backtest().then(setBt).catch(() => setBt(null));
  }, []);
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

      {/* Backtest — measurable accuracy */}
      <Card className="mt-4">
        <CardHead title="Forecast accuracy — backtest"
          hint={bt ? `${bt.granularity} · ${bt.holdout_weeks}-week holdout` : "running…"} />
        <div className="p-[18px]">
          <div className="grid gap-3 mb-4" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))" }}>
            {[
              { l: "Accuracy", v: bt?.accuracy != null ? `${bt.accuracy}%` : "—", hint: "100 − MAPE" },
              { l: "MAPE", v: bt?.mape != null ? `${bt.mape}%` : "—", hint: "mean abs % error" },
              { l: "MAE", v: bt?.mae ?? "—", hint: "mean abs error" },
              { l: "RMSE", v: bt?.rmse ?? "—", hint: "root mean sq error" },
              { l: "Bias", v: bt?.bias != null ? (bt.bias > 0 ? `+${bt.bias}` : bt.bias) : "—", hint: "over/under forecast" },
            ].map((m) => (
              <div key={m.l} className="rounded border p-3" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                <div className="eyebrow">{m.l}</div>
                <div className="text-[1.5rem] font-bold tnum mt-1">{m.v}</div>
                <div className="text-[0.6875rem] text-ink-3 mt-0.5">{m.hint}</div>
              </div>
            ))}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[0.8125rem] min-w-[520px]">
              <thead><tr>
                <th className="text-left px-3 py-2 text-[10px] uppercase tracking-wider text-ink-3 border-b" style={{ borderColor: "var(--hairline)" }}>Week</th>
                <th className="text-right px-3 py-2 text-[10px] uppercase tracking-wider text-ink-3 border-b" style={{ borderColor: "var(--hairline)" }}>Actual</th>
                <th className="text-right px-3 py-2 text-[10px] uppercase tracking-wider text-ink-3 border-b" style={{ borderColor: "var(--hairline)" }}>Predicted</th>
                <th className="text-right px-3 py-2 text-[10px] uppercase tracking-wider text-ink-3 border-b" style={{ borderColor: "var(--hairline)" }}>Error %</th>
              </tr></thead>
              <tbody>
                {(bt?.points ?? []).map((p) => {
                  const errPct = p.actual ? ((p.predicted - p.actual) / p.actual) * 100 : 0;
                  return (
                    <tr key={p.ds}>
                      <td className="px-3 py-2 border-b text-ink-2 font-mono" style={{ borderColor: "var(--hairline)" }}>{p.ds}</td>
                      <td className="px-3 py-2 border-b text-right tnum text-ink" style={{ borderColor: "var(--hairline)" }}>{p.actual.toLocaleString()}</td>
                      <td className="px-3 py-2 border-b text-right tnum text-ink-2" style={{ borderColor: "var(--hairline)" }}>{p.predicted.toLocaleString()}</td>
                      <td className="px-3 py-2 border-b text-right tnum" style={{ borderColor: "var(--hairline)", color: Math.abs(errPct) > 25 ? "var(--critical)" : Math.abs(errPct) > 10 ? "var(--warning)" : "var(--good)" }}>
                        {errPct > 0 ? "+" : ""}{errPct.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
                {!bt?.points?.length && <tr><td colSpan={4} className="px-3 py-4 text-ink-3 text-center">Running backtest…</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </Card>
    </AppShell>
  );
}
