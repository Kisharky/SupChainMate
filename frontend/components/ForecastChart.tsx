"use client";
/** SVG demand chart: history line + forecast median with confidence band. */
import type { ForecastPoint } from "@/lib/api";

export function ForecastChart({ history, forecast, w = 720, h = 220 }:
  { history: ForecastPoint[]; forecast: ForecastPoint[]; w?: number; h?: number }) {
  const pad = { l: 8, r: 8, t: 12, b: 18 };
  const hist = history.map((p) => p.y ?? 0);
  const fc = forecast.map((p) => p.yhat ?? 0);
  const up = forecast.map((p) => p.upper ?? 0);
  const lo = forecast.map((p) => p.lower ?? 0);
  const all = [...hist, ...up, ...lo].filter((v) => v > 0);
  if (all.length === 0) return <div className="text-ink-3 text-[0.8125rem]">No forecast data.</div>;
  const max = Math.max(...all) * 1.08;
  const min = Math.min(...all) * 0.9;
  const n = history.length + forecast.length;
  const X = (i: number) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const Y = (v: number) => pad.t + (1 - (v - min) / (max - min)) * (h - pad.t - pad.b);

  const histPath = hist.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v)}`).join(" ");
  const fcStart = history.length - 1;
  const fcPath = fc.map((v, i) => `${i ? "L" : "M"}${X(fcStart + 1 + i)},${Y(v)}`).join(" ");
  const bandTop = up.map((v, i) => `${i ? "L" : "M"}${X(fcStart + 1 + i)},${Y(v)}`).join(" ");
  const bandBot = lo.map((v, i) => `L${X(fcStart + 1 + i)},${Y(v)}`).reverse().join(" ");
  const band = `${bandTop} ${lo.map((v, i) => `L${X(fcStart + 1 + i)},${Y(v)}`).reverse().join(" ")} Z`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none" aria-label="Demand forecast">
      {[0, 1, 2, 3].map((g) => {
        const y = pad.t + (g / 3) * (h - pad.t - pad.b);
        return <line key={g} x1={0} y1={y} x2={w} y2={y} stroke="var(--hairline)" strokeWidth={1} />;
      })}
      <path d={band} fill="var(--info)" opacity={0.14} />
      <line x1={X(fcStart)} y1={pad.t} x2={X(fcStart)} y2={h - pad.b} stroke="var(--hairline-strong)" strokeDasharray="3 3" />
      <path d={histPath} fill="none" stroke="var(--accent)" strokeWidth={2.2} strokeLinejoin="round" />
      <path d={fcPath} fill="none" stroke="var(--info)" strokeWidth={2.2} strokeDasharray="5 4" strokeLinejoin="round" />
    </svg>
  );
}
