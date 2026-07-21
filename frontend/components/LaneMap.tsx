"use client";
/**
 * LaneMap — real basemap (MapTiler tiles when a key is configured) with the
 * network's hubs and inter-hub routes drawn from live backend geo data.
 * Loaded via next/dynamic (ssr:false) because Leaflet needs the DOM.
 */
import { useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import type { MapResponse } from "@/lib/api";

const COLOR = { good: "#10B981", warning: "#F59E0B", critical: "#EF4444", info: "#38BDF8" };

function Fit({ data }: { data: MapResponse }) {
  const map = useMap();
  useEffect(() => {
    if (data.points.length) {
      const b = data.points.map((p) => [p.lat, p.lon] as [number, number]);
      map.fitBounds(b, { padding: [40, 40] });
    }
  }, [data, map]);
  return null;
}

export default function LaneMap({ data, tour }: { data: MapResponse; tour?: { lat: number; lon: number }[] }) {
  const hasTiles = Boolean(data.tiles_url);
  const tourLine = tour && tour.length > 1
    ? [...tour.map((t) => [t.lat, t.lon] as [number, number]), [tour[0].lat, tour[0].lon] as [number, number]]
    : null;
  return (
    <MapContainer center={data.center} zoom={data.zoom} style={{ height: 340, width: "100%", background: "var(--bg-sunken)" }}
      scrollWheelZoom={false} attributionControl={hasTiles}>
      {hasTiles && <TileLayer url={data.tiles_url!} attribution={data.attribution} />}
      {tourLine && (
        <Polyline positions={tourLine} pathOptions={{ color: "#10B981", weight: 3, opacity: 0.95 }}>
          <Tooltip>Optimised delivery tour</Tooltip>
        </Polyline>
      )}
      {!tourLine && data.routes.map((r, i) => (
        <Polyline key={i} positions={[[r.from.lat, r.from.lon], [r.to.lat, r.to.lon]]}
          pathOptions={{ color: COLOR[r.status], weight: 2, opacity: 0.85, dashArray: r.status === "warning" ? "6 6" : undefined }}>
          <Tooltip>{r.from.name} → {r.to.name} · {Math.round(r.distance_km)} km</Tooltip>
        </Polyline>
      ))}
      {data.points.map((p, i) => (
        <CircleMarker key={i} center={[p.lat, p.lon]} radius={i === 0 ? 9 : 6}
          pathOptions={{ color: i === 0 ? COLOR.info : COLOR.good, fillColor: i === 0 ? COLOR.info : COLOR.good, fillOpacity: 0.7, weight: 2 }}>
          <Tooltip>{p.name} · {p.size.toLocaleString()} locations</Tooltip>
        </CircleMarker>
      ))}
      <Fit data={data} />
    </MapContainer>
  );
}
