"use client";
/**
 * Disruption & Risk Radar — signal convergence over the supply network.
 * A composite Supply Chain Risk Index, toggleable disruption layers, a dual map
 * engine (flat Leaflet + 3D globe), and convergence alerts that fire only when
 * several independent signals line up. Exceptions route to the Decision Center.
 * Representative signals (labelled) over the existing layers; fully offline.
 */
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, Badge, Button, Progress, Modal, EmptyState, Alert } from "@/components/ui/primitives";
import { api, RadarResponse, RadarNodeDetail, MapResponse } from "@/lib/api";

const MapLoading = () => <div style={{ height: 460 }} className="grid place-items-center"><EmptyState kind="loading" /></div>;
const RadarMap = dynamic(() => import("@/components/RadarMap"), { ssr: false, loading: MapLoading });
const RadarGlobe = dynamic(() => import("@/components/RadarGlobe"), { ssr: false, loading: MapLoading });

export default function Radar() {
  const [d, setD] = useState<RadarResponse | null>(null);
  const [map, setMap] = useState<MapResponse | null>(null);
  const [err, setErr] = useState(false);
  const [view, setView] = useState<"flat" | "globe">("flat");
  const [enabled, setEnabled] = useState<string[] | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [sent, setSent] = useState<Record<string, boolean>>({});

  useEffect(() => { api.radar().then(setD).catch(() => setErr(true)); }, []);
  useEffect(() => { api.logisticsMap().then(setMap).catch(() => setMap(null)); }, []);
  useEffect(() => { if (d && enabled === null) setEnabled(d.layers.map((l) => l.id)); }, [d, enabled]);

  const on = enabled ?? [];
  const toggle = (id: string) => setEnabled((e) => (e ?? []).includes(id) ? (e ?? []).filter((x) => x !== id) : [...(e ?? []), id]);
  const idx = d?.index;

  const bandColor = (status?: string) => `var(--${status ?? "info"})`;
  const alerts = useMemo(() => d?.alerts ?? [], [d]);

  return (
    <AppShell title="Risk Radar">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>
          Situational awareness {d?.source === "representative" ? "· representative signals" : ""}
        </div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Disruption &amp; Risk Radar</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem] max-w-2xl">
          Weather, ports, suppliers, labour, customs, and geopolitics on one map — a lane lights up only when several
          independent signals converge, so the board shows decisions, not noise.
        </p>
      </div>

      {err && <Alert status="critical" title="API unreachable">Start the FastAPI backend to load the radar.</Alert>}

      {/* ---- Index + brief ---- */}
      <div className="grid gap-4 mb-4 items-stretch" style={{ gridTemplateColumns: "minmax(240px,320px) 1fr" }}>
        <Card className="p-5 relative overflow-hidden">
          <span className="absolute left-0 top-0 bottom-0 w-[3px]" style={{ background: bandColor(idx?.status) }} />
          <div className="eyebrow">Supply Chain Risk Index</div>
          <div className="flex items-end gap-3 mt-1">
            <div className="text-[3.5rem] font-bold leading-none tnum">{idx?.score ?? "—"}</div>
            {idx && <Badge status={idx.status as any}>{idx.band}</Badge>}
          </div>
          <div className="flex gap-4 mt-3 text-[0.8125rem]">
            <span className="text-ink-2"><b className="text-ink tnum">{idx?.converging_alerts ?? 0}</b> converging</span>
            <span className="text-ink-2"><b className="tnum" style={{ color: "var(--critical)" }}>{idx?.critical_alerts ?? 0}</b> critical</span>
          </div>
        </Card>
        <Card className="p-5 flex flex-col justify-center">
          <div className="eyebrow mb-1.5">Live disruption brief</div>
          <p className="text-[0.9375rem] text-ink leading-relaxed">{d?.brief ?? "Loading the network…"}</p>
          <div className="flex flex-wrap gap-1.5 mt-3">
            {(idx?.by_region ?? []).map((r) => (
              <span key={r.region} className="inline-flex items-center gap-1.5 rounded-full border px-2 py-[3px] text-[0.6875rem]"
                style={{ borderColor: "var(--hairline)", color: bandColor(undefined) }}>
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: r.score >= 75 ? "var(--critical)" : r.score >= 50 ? "var(--warning)" : "var(--info)" }} />
                {r.region} <b className="tnum text-ink-2">{r.score}</b>
              </span>
            ))}
          </div>
        </Card>
      </div>

      {/* ---- Map + layers ---- */}
      <Card className="mb-4">
        <CardHead title="Network disruption map" hint={`${on.length}/${d?.layers.length ?? 0} layers`}
          right={
            <div className="flex rounded-sm border overflow-hidden" style={{ borderColor: "var(--hairline-strong)" }}>
              {(["flat", "globe"] as const).map((v) => (
                <button key={v} onClick={() => setView(v)}
                  className="px-3 py-1 text-[0.75rem] font-semibold capitalize"
                  style={view === v ? { background: "var(--accent)", color: "var(--accent-ink)" } : { color: "var(--text-2)" }}>
                  {v === "flat" ? "Flat map" : "3D globe"}
                </button>
              ))}
            </div>
          } />
        {/* layer toggles */}
        <div className="flex flex-wrap gap-1.5 px-[18px] py-3 border-b" style={{ borderColor: "var(--hairline)" }}>
          {(d?.layers ?? []).map((l) => {
            const active = on.includes(l.id);
            return (
              <button key={l.id} onClick={() => toggle(l.id)}
                className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[0.75rem] transition"
                style={{ borderColor: active ? l.color : "var(--hairline)", color: active ? "var(--text)" : "var(--text-3)",
                         background: active ? `color-mix(in srgb, ${l.color} 14%, transparent)` : "transparent" }}>
                <span>{l.icon}</span>{l.name}
                <span className="tnum" style={{ color: l.color }}>{l.active_events}</span>
              </button>
            );
          })}
        </div>
        <div className="p-2">
          {!d ? <div style={{ height: 460 }} className="grid place-items-center"><EmptyState kind="loading" /></div>
            : view === "flat"
              ? <RadarMap nodes={d.nodes} lanes={d.lanes} enabled={on} tiles={map?.tiles_url} attribution={map?.attribution} onSelect={setSel} />
              : <RadarGlobe nodes={d.nodes} lanes={d.lanes} enabled={on} onSelect={setSel} />}
        </div>
        <div className="px-[18px] py-2.5 text-[0.6875rem] text-ink-3 border-t flex flex-wrap gap-x-4 gap-y-1" style={{ borderColor: "var(--hairline)" }}>
          <span><span style={{ color: "var(--critical)" }}>●</span> Severe</span>
          <span><span style={{ color: "var(--warning)" }}>●</span> High</span>
          <span><span style={{ color: "var(--info)" }}>●</span> Elevated</span>
          <span><span style={{ color: "var(--good)" }}>●</span> Low</span>
          <span className="ml-auto">Click a node for its convergence breakdown · toggle layers to filter signals</span>
        </div>
      </Card>

      {/* ---- Convergence alerts + category breakdown ---- */}
      <div className="grid gap-4 items-start" style={{ gridTemplateColumns: "1.5fr 1fr" }}>
        <div>
          <h2 className="text-[1.25rem] font-semibold tracking-tight mb-3">Convergence alerts</h2>
          <div className="flex flex-col gap-3">
            {alerts.map((a) => (
              <Card key={a.id} className="p-4">
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge status={a.status as any}>{a.band}</Badge>
                  {a.critical && <Badge status="critical">critical convergence</Badge>}
                  <span className="font-semibold text-[0.9375rem]">{a.name}</span>
                  <span className="text-[0.75rem] text-ink-3">· {a.region}</span>
                  <span className="ml-auto text-[0.75rem] text-ink-3"><b className="text-ink tnum">{a.convergence}</b> signals · score <b className="text-ink tnum">{a.composite_score}</b></span>
                </div>
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {a.categories.map((c) => <span key={c} className="rounded border px-1.5 py-0.5 text-[0.6875rem] text-ink-2" style={{ borderColor: "var(--hairline)" }}>{c}</span>)}
                </div>
                <p className="text-[0.8125rem] text-ink-2 mt-2">{a.why}</p>
                <div className="flex items-center gap-2 mt-3 flex-wrap">
                  <span className="text-[0.75rem] text-ink-2 flex-1 min-w-[180px]"><span className="text-ink-3">Recommended:</span> {a.recommended_action}</span>
                  <Button sm variant="ghost" onClick={() => setSel(a.ref_id)}>Inspect node</Button>
                  {sent[a.id]
                    ? <Badge status="good">Sent to Decision Center</Badge>
                    : <Button sm variant="secondary" onClick={() => setSent((m) => ({ ...m, [a.id]: true }))}>Send to Decision Center</Button>}
                </div>
              </Card>
            ))}
            {d && alerts.length === 0 && <Card><EmptyState title="No converging signals" hint="The network is stable." /></Card>}
            {!d && !err && <Card><EmptyState kind="loading" /></Card>}
          </div>
        </div>

        <Card>
          <CardHead title="Risk by category" hint="peak signal per layer" />
          <div className="p-[18px] flex flex-col gap-2.5">
            {(idx?.by_category ?? []).map((c) => (
              <div key={c.id}>
                <div className="flex justify-between text-[0.75rem] mb-1">
                  <span className="text-ink-2">{c.name}</span>
                  <span className="tnum text-ink-3">{c.severity} · {c.band}</span>
                </div>
                <Progress value={c.severity} status={c.severity >= 75 ? "critical" : c.severity >= 50 ? "warning" : c.severity >= 25 ? "info" : "good"} />
              </div>
            ))}
            {!idx && <EmptyState kind="loading" />}
          </div>
          <div className="px-[18px] py-3 text-[0.6875rem] text-ink-3 border-t" style={{ borderColor: "var(--hairline)" }}>
            Convergence alerts and node inspections route to the{" "}
            <Link href="/decisions" className="underline">Decision Center</Link> for a human call.
          </div>
        </Card>
      </div>

      <NodeModal nodeId={sel} onClose={() => setSel(null)} />
    </AppShell>
  );
}

function NodeModal({ nodeId, onClose }: { nodeId: string | null; onClose: () => void }) {
  const [det, setDet] = useState<RadarNodeDetail | null>(null);
  useEffect(() => {
    setDet(null);
    if (nodeId) api.radarNode(nodeId).then(setDet).catch(() => setDet(null));
  }, [nodeId]);
  if (!nodeId) return null;

  return (
    <Modal open={!!nodeId} onClose={onClose}
      title={det ? `${det.name}` : "Node"} subtitle={det ? `${det.type} · ${det.region}` : undefined}
      footer={<Button variant="primary" sm onClick={onClose}>Send to Decision Center</Button>}>
      {!det ? <EmptyState kind="loading" /> : !det.ok ? <EmptyState kind="error" title="Couldn't load node" /> : (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <Badge status={det.status as any}>risk {det.risk_score} · {det.band}</Badge>
            <span className="text-[0.8125rem] text-ink-2">{det.convergence} converging signal(s)</span>
          </div>
          <div>
            <div className="eyebrow mb-1.5">Signal breakdown</div>
            <div className="flex flex-col gap-2">
              {det.signals.map((s) => (
                <div key={s.layer}>
                  <div className="flex justify-between text-[0.75rem] mb-1">
                    <span className={s.active ? "text-ink" : "text-ink-3"}>{s.layer_name}{s.active && <span className="text-[0.625rem] ml-1" style={{ color: "var(--accent)" }}>active</span>}</span>
                    <span className="tnum text-ink-3">{s.severity}</span>
                  </div>
                  <Progress value={s.severity} status={s.severity >= 75 ? "critical" : s.severity >= 50 ? "warning" : s.severity >= 25 ? "info" : "good"} />
                </div>
              ))}
            </div>
          </div>
          <Alert status={det.convergence >= 3 ? "critical" : "warning"} title="Why this is flagged">{det.why}</Alert>
          <div className="text-[0.8125rem] text-ink-2"><span className="text-ink-3">Recommended:</span> {det.recommended_action}</div>
          {det.lanes.length > 0 && (
            <div>
              <div className="eyebrow mb-1.5">Lanes touched</div>
              <div className="flex flex-wrap gap-1.5">
                {det.lanes.map((l, i) => (
                  <span key={i} className="inline-flex items-center gap-1.5 rounded border px-2 py-1 text-[0.75rem]" style={{ borderColor: "var(--hairline)" }}>
                    → {l.to} <Badge status={l.risk_score >= 75 ? "critical" : l.risk_score >= 50 ? "warning" : "info"}>{l.risk_score}</Badge>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}
