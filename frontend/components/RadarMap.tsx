"use client";
/**
 * RadarMap — flat disruption map (Leaflet). Network nodes coloured by risk band,
 * lanes as risk-coloured arcs. Layer toggles control which nodes are "lit"
 * (they have an active signal in an enabled layer); the rest dim to context.
 * Loaded via next/dynamic (ssr:false).
 */
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import type { RadarNode, RadarLane } from "@/lib/api";

const COLOR: Record<string, string> = { good: "#10B981", warning: "#F59E0B", critical: "#EF4444", info: "#38BDF8" };
const ACTIVE = 40;

export default function RadarMap({ nodes, lanes, enabled, tiles, attribution, onSelect }: {
  nodes: RadarNode[]; lanes: RadarLane[]; enabled: string[];
  tiles?: string | null; attribution?: string; onSelect: (id: string) => void;
}) {
  const lit = (n: RadarNode) => Object.entries(n.signals).some(([k, v]) => v >= ACTIVE && enabled.includes(k));
  const litIds = new Set(nodes.filter(lit).map((n) => n.id));

  return (
    <MapContainer center={[20, 10]} zoom={2} minZoom={2} worldCopyJump
      style={{ height: 460, width: "100%", background: "var(--bg-sunken)" }}
      scrollWheelZoom={false} attributionControl={Boolean(tiles)}>
      {tiles && <TileLayer url={tiles} attribution={attribution} />}
      {lanes.map((l) => {
        const on = litIds.has(l.from_id) || litIds.has(l.to_id);
        return (
          <Polyline key={l.id} positions={[[l.from_lat, l.from_lon], [l.to_lat, l.to_lon]]}
            pathOptions={{ color: on ? COLOR[l.status] : "#334155", weight: on ? 2 : 1, opacity: on ? 0.7 : 0.22 }}>
            <Tooltip>{l.from} → {l.to} · risk {l.risk_score}</Tooltip>
          </Polyline>
        );
      })}
      {nodes.map((n) => {
        const on = litIds.has(n.id);
        const r = on ? 5 + Math.round(n.risk_score / 12) : 4;
        const c = on ? COLOR[n.status] : "#475569";
        return (
          <CircleMarker key={n.id} center={[n.lat, n.lon]} radius={r}
            pathOptions={{ color: c, fillColor: c, fillOpacity: on ? 0.85 : 0.3, weight: 1 }}
            eventHandlers={{ click: () => onSelect(n.id) }}>
            <Tooltip>{n.name} · risk {n.risk_score} · {n.convergence} signal(s)</Tooltip>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
