/**
 * Shared UI primitives — the design-system components as React.
 * Button · Card · Badge · KpiCard · Alert · Sparkline · DataTable.
 * All styling flows from the CSS tokens in app/globals.css.
 */
import { ReactNode } from "react";
import type { KpiStatus } from "@/lib/api";

const STATUS_COLOR: Record<KpiStatus, string> = {
  good: "var(--good)", warning: "var(--warning)",
  critical: "var(--critical)", info: "var(--info)",
};
const STATUS_BG: Record<KpiStatus, string> = {
  good: "var(--good-bg)", warning: "var(--warning-bg)",
  critical: "var(--critical-bg)", info: "var(--info-bg)",
};

/* ---- Button ---- */
type BtnVariant = "primary" | "secondary" | "ghost" | "danger";
export function Button(
  { variant = "secondary", sm, children, ...rest }:
  { variant?: BtnVariant; sm?: boolean; children: ReactNode } &
  React.ButtonHTMLAttributes<HTMLButtonElement>,
) {
  const base = "inline-flex items-center justify-center gap-2 font-semibold whitespace-nowrap rounded-sm border transition " +
    "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 active:translate-y-px disabled:opacity-45";
  const size = sm ? "px-2.5 py-1.5 text-[0.75rem]" : "px-3.5 py-2 text-[0.8125rem]";
  const styles: Record<BtnVariant, string> = {
    primary: "bg-emerald-500 text-[var(--accent-ink)] border-emerald-500 hover:bg-emerald-400 shadow-[0_4px_14px_var(--emerald-glow)]",
    secondary: "bg-[var(--panel-2)] text-ink border-[var(--hairline-strong)] hover:brightness-110",
    ghost: "bg-transparent text-ink-2 border-transparent hover:bg-white/5 hover:text-ink",
    danger: "bg-transparent text-critical border-[color-mix(in_srgb,var(--critical)_45%,transparent)] hover:bg-[var(--critical-bg)]",
  };
  return <button className={`${base} ${size} ${styles[variant]}`} style={{ outlineColor: "var(--focus)" }} {...rest}>{children}</button>;
}

/* ---- Card ---- */
export function Card({ children, className = "", ...rest }: { children: ReactNode; className?: string } & React.HTMLAttributes<HTMLDivElement>) {
  return <div className={`rounded-lg border shadow-card bg-panel ${className}`} style={{ borderColor: "var(--hairline)" }} {...rest}>{children}</div>;
}
export function CardHead({ title, hint, right }: { title: string; hint?: string; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 px-[18px] py-3.5 border-b" style={{ borderColor: "var(--hairline)" }}>
      <div className="text-[0.9375rem] font-semibold">{title}</div>
      {hint && <div className="text-[0.75rem] text-ink-3">{hint}</div>}
      {right}
    </div>
  );
}

/* ---- Badge ---- */
export function Badge({ status = "info", children }: { status?: KpiStatus | "neutral"; children: ReactNode }) {
  const isNeutral = status === "neutral";
  const color = isNeutral ? "var(--text-2)" : STATUS_COLOR[status as KpiStatus];
  const bg = isNeutral ? "color-mix(in srgb, var(--text) 8%, transparent)" : STATUS_BG[status as KpiStatus];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border px-2 py-[3px] text-[0.75rem] font-semibold"
      style={{ color, background: bg, borderColor: `color-mix(in srgb, ${color} 30%, transparent)` }}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: "currentColor" }} />
      {children}
    </span>
  );
}

/* ---- Sparkline (deterministic, SSR-safe) ---- */
export function Sparkline({ seed = 4, color = "var(--good)", trend = "up", w = 100, h = 40 }:
  { seed?: number; color?: string; trend?: "up" | "down" | "flat"; w?: number; h?: number }) {
  let s = seed;
  const rnd = () => (s = (s * 9301 + 49297) % 233280) / 233280;
  const n = 20; const pts: number[] = []; let y = 0.5;
  for (let i = 0; i < n; i++) {
    if (trend === "up") y = 0.68 - (i / n) * 0.32 + (rnd() - 0.5) * 0.12;
    else if (trend === "down") y = 0.32 + (i / n) * 0.32 + (rnd() - 0.5) * 0.12;
    else { y += (rnd() - 0.5) * 0.14; }
    pts.push(Math.max(0.12, Math.min(0.88, y)));
  }
  const d = pts.map((p, i) => `${i ? "L" : "M"}${(i / (n - 1)) * w},${h - p * h}`).join(" ");
  const area = `${d} L${w},${h} L0,${h} Z`;
  const gid = `sp${seed}${trend}`;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden>
      <defs><linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stopColor={color} stopOpacity="0.33" /><stop offset="1" stopColor={color} stopOpacity="0" />
      </linearGradient></defs>
      <path d={area} fill={`url(#${gid})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

/* ---- KPI card ---- */
export function KpiCard({ label, value, unit, prefix, delta, status = "good", seed = 4 }:
  { label: string; value: number | string; unit?: string; prefix?: string; delta?: number; status?: KpiStatus; seed?: number }) {
  const trend = delta == null || delta === 0 ? "flat" : delta > 0 ? "up" : "down";
  const deltaColor = trend === "up" ? "var(--good)" : trend === "down" ? "var(--critical)" : "var(--text-3)";
  const arrow = trend === "up" ? "▲" : trend === "down" ? "▼" : "—";
  return (
    <Card className="relative overflow-hidden">
      <span className="absolute left-0 top-0 bottom-0 w-[3px]" style={{ background: STATUS_COLOR[status] }} />
      <div className="p-4 flex flex-col gap-2.5">
        <div className="eyebrow">{label}</div>
        <div className="text-[2.25rem] font-bold leading-none tracking-tight tnum">
          {prefix}{value}{unit && <span className="text-[1.125rem] text-ink-3 font-semibold ml-0.5">{unit}</span>}
        </div>
        {delta != null && (
          <div className="text-[0.8125rem] font-semibold" style={{ color: deltaColor }}>
            {arrow} {Math.abs(delta)}{unit === "%" ? " pts" : ""}
          </div>
        )}
        <div className="absolute right-0 bottom-0 opacity-90">
          <Sparkline seed={seed} color={STATUS_COLOR[status]} trend={trend === "flat" ? "up" : trend} />
        </div>
      </div>
    </Card>
  );
}

/* ---- Alert ---- */
export function Alert({ status = "info", title, children }: { status?: KpiStatus; title: string; children?: ReactNode }) {
  const ico = { good: "●", warning: "▲", critical: "◆", info: "◔" }[status];
  return (
    <div className="flex gap-3 rounded border p-3 items-start bg-[var(--panel-2)]"
      style={{ borderColor: "var(--hairline)", borderLeftWidth: 3, borderLeftColor: STATUS_COLOR[status] }}>
      <span className="text-[15px] leading-snug" style={{ color: STATUS_COLOR[status] }}>{ico}</span>
      <div className="min-w-0">
        <div className="font-semibold text-[0.9375rem]">{title}</div>
        {children && <div className="text-ink-2 text-[0.8125rem] mt-0.5">{children}</div>}
      </div>
    </div>
  );
}

/* ---- Data table ---- */
export function DataTable({ head, children }: { head: ReactNode; children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-lg border" style={{ borderColor: "var(--hairline)" }}>
      <table className="w-full border-collapse text-[0.8125rem] min-w-[640px]">
        <thead><tr className="text-left">{head}</tr></thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}
export function Th({ children, num }: { children: ReactNode; num?: boolean }) {
  return <th className={`px-3.5 py-2.5 text-[10.5px] uppercase tracking-wider font-semibold text-ink-3 bg-[var(--panel-2)] border-b ${num ? "text-right" : ""}`}
    style={{ borderColor: "var(--hairline)" }}>{children}</th>;
}
export function Td({ children, num, strong }: { children: ReactNode; num?: boolean; strong?: boolean }) {
  return <td className={`px-3.5 py-2.5 border-b ${num ? "text-right tnum" : ""} ${strong ? "text-ink font-semibold" : "text-ink-2"}`}
    style={{ borderColor: "var(--hairline)" }}>{children}</td>;
}
