"""
views/map_view.py
Leaflet disruption-radar map (folium) with MapTiler dark tiles.

Falls back cleanly: no MapTiler key → OSM tiles; folium missing → the
caller keeps the plotly map. Points are sampled for browser performance.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

import config
from modules import geo

_log = config.get_logger(__name__)

MAX_POINTS = 1200
_LEVEL_COLORS = {"Critical": "#FF003C", "Warning": "#FBC02D", "Safe": "#00D4FF"}


def leaflet_available() -> bool:
    try:
        import folium  # noqa: F401
        import streamlit_folium  # noqa: F401
        return True
    except ImportError:
        return False


def render_radar(geo_df: pd.DataFrame, height: int = 480) -> bool:
    """Render the risk map with Leaflet. Returns False if unavailable."""
    if not leaflet_available():
        return False
    import folium
    from streamlit_folium import st_folium

    pts = geo_df.dropna(subset=["lat", "lon"])
    if not len(pts):
        return False
    if len(pts) > MAX_POINTS:
        pts = pts.sample(MAX_POINTS, random_state=42)

    tiles = geo.maptiler_tiles_url()
    if tiles:
        base = dict(tiles=tiles, attr=geo.maptiler_attribution())
        basemap_label = "MAPTILER"
    else:
        base = dict(tiles="OpenStreetMap", attr=None)
        basemap_label = "OPENSTREETMAP (set MAPTILER_API_KEY for the dark basemap)"

    m = folium.Map(location=[float(pts["lat"].mean()), float(pts["lon"].mean())],
                   zoom_start=4, control_scale=True, **base)
    for _, r in pts.iterrows():
        level = str(r.get("combined_level", "Safe"))
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=3 + float(r.get("combined_risk", 0)) / 25,
            color=_LEVEL_COLORS.get(level, "#00D4FF"),
            fill=True, fill_opacity=0.6, weight=1,
            tooltip=(f"{level} · combined risk "
                     f"{float(r.get('combined_risk', 0)):.0f}/100"),
        ).add_to(m)

    if "cluster" in pts.columns:
        for cid, grp in pts.groupby("cluster"):
            folium.Marker(
                location=[float(grp["lat"].mean()), float(grp["lon"].mean())],
                tooltip=f"Hub / zone {cid} · {len(grp)} customers (sampled)",
                icon=folium.Icon(color="black", icon="warehouse", prefix="fa"),
            ).add_to(m)

    st_folium(m, height=height, use_container_width=True, returned_objects=[])
    st.caption(f"LEAFLET · BASEMAP: {basemap_label} · "
               f"{len(pts):,} points shown{' (sampled)' if len(geo_df) > MAX_POINTS else ''}")
    return True
