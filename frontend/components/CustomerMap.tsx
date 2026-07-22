"use client";
/** CustomerMap — compact Leaflet route map for a customer's shipment lane.
 * Points coloured by on-time/late status; loaded via next/dynamic (ssr:false). */
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";

type Pt = { name: string; lat: number; lon: number; status: string };
const COLOR: Record<string, string> = { on_time: "#10B981", late: "#EF4444" };

export default function CustomerMap({ points, tiles, attribution }: {
  points: Pt[]; tiles?: string | null; attribution?: string;
}) {
  const line = points.map((p) => [p.lat, p.lon] as [number, number]);
  const center: [number, number] = points.length
    ? [points.reduce((s, p) => s + p.lat, 0) / points.length, points.reduce((s, p) => s + p.lon, 0) / points.length]
    : [-22, -45];
  return (
    <MapContainer center={center} zoom={5} style={{ height: 260, width: "100%", background: "var(--bg-sunken)" }}
      scrollWheelZoom={false} attributionControl={Boolean(tiles)}>
      {tiles && <TileLayer url={tiles} attribution={attribution} />}
      {line.length > 1 && <Polyline positions={line} pathOptions={{ color: "#38BDF8", weight: 2, opacity: 0.6, dashArray: "5 6" }} />}
      {points.map((p, i) => (
        <CircleMarker key={i} center={[p.lat, p.lon]} radius={7}
          pathOptions={{ color: COLOR[p.status] ?? "#38BDF8", fillColor: COLOR[p.status] ?? "#38BDF8", fillOpacity: 0.85, weight: 1 }}>
          <Tooltip>{p.name} · {p.status.replace("_", " ")}</Tooltip>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
