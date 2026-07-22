"use client";
/**
 * RadarGlobe — 3D disruption globe (globe.gl / three). Risk nodes as points,
 * lanes as animated arcs, coloured by risk band. No external textures (a plain
 * dark globe + graticules), so it renders fully offline. Loaded via next/dynamic
 * (ssr:false) because it needs the DOM / WebGL.
 */
import { useEffect, useRef } from "react";
import type { RadarNode, RadarLane } from "@/lib/api";

const COLOR: Record<string, string> = { good: "#10B981", warning: "#F59E0B", critical: "#EF4444", info: "#38BDF8" };
const ACTIVE = 40;

export default function RadarGlobe({ nodes, lanes, enabled, onSelect }: {
  nodes: RadarNode[]; lanes: RadarLane[]; enabled: string[]; onSelect: (id: string) => void;
}) {
  const elRef = useRef<HTMLDivElement>(null);
  const globeRef = useRef<any>(null);
  const selRef = useRef(onSelect);
  selRef.current = onSelect;

  // Instantiate once.
  useEffect(() => {
    let disposed = false;
    const el = elRef.current;
    (async () => {
      const Globe = (await import("globe.gl")).default;
      if (disposed || !el) return;
      const globe = new Globe(el)
        .backgroundColor("rgba(0,0,0,0)")
        .showAtmosphere(true).atmosphereColor("#38BDF8").atmosphereAltitude(0.16)
        .showGraticules(true)
        .width(el.clientWidth).height(460)
        .pointLat("lat").pointLng("lon").pointColor((d: any) => d._color)
        .pointAltitude((d: any) => 0.01 + d.risk_score / 900).pointRadius((d: any) => 0.28 + d.risk_score / 130)
        .pointLabel((d: any) => `${d.name} · risk ${d.risk_score} · ${d.convergence} signal(s)`)
        .onPointClick((d: any) => selRef.current(d.id))
        .arcStartLat("from_lat").arcStartLng("from_lon").arcEndLat("to_lat").arcEndLng("to_lon")
        .arcColor((d: any) => [d._color, d._color]).arcStroke(0.5)
        .arcDashLength(0.4).arcDashGap(0.18).arcDashAnimateTime((d: any) => 2600 - d.risk_score * 12)
        .arcLabel((d: any) => `${d.from} → ${d.to} · risk ${d.risk_score}`);
      try {
        const mat = globe.globeMaterial();
        mat.color?.set?.("#0b1220");
        if (mat.emissive?.set) { mat.emissive.set("#0a1424"); mat.emissiveIntensity = 0.6; }
      } catch { /* material tweak is cosmetic */ }
      const controls = globe.controls();
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.55;
      controls.enableZoom = true;
      globe.pointOfView({ lat: 15, lng: -30, altitude: 2.4 });
      globeRef.current = globe;

      const onResize = () => globe.width(el.clientWidth);
      window.addEventListener("resize", onResize);
      (globe as any)._onResize = onResize;
    })();
    return () => {
      disposed = true;
      const g = globeRef.current;
      if (g) {
        if ((g as any)._onResize) window.removeEventListener("resize", (g as any)._onResize);
        try { g._destructor?.(); } catch { /* noop */ }
      }
      if (el) el.innerHTML = "";
      globeRef.current = null;
    };
  }, []);

  // Push data on change (respecting the enabled-layer filter).
  useEffect(() => {
    const globe = globeRef.current;
    if (!globe) return;
    const lit = (n: RadarNode) => Object.entries(n.signals).some(([k, v]) => v >= ACTIVE && enabled.includes(k));
    const litNodes = nodes.filter(lit).map((n) => ({ ...n, _color: COLOR[n.status] }));
    const litIds = new Set(litNodes.map((n) => n.id));
    const litLanes = lanes
      .filter((l) => litIds.has(l.from_id) || litIds.has(l.to_id))
      .map((l) => ({ ...l, _color: COLOR[l.status] }));
    globe.pointsData(litNodes).arcsData(litLanes);
  }, [nodes, lanes, enabled]);

  return <div ref={elRef} style={{ height: 460, width: "100%" }} />;
}
