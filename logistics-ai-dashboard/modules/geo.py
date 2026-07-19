"""
modules/geo.py
Vendor-neutral geo services for the control tower:

  - Basemap   : MapTiler raster tiles (key) → carto-darkmatter fallback
  - Geocoding : Nominatim / OpenStreetMap (keyless, cached, rate-polite)
  - Routing   : HERE Matrix API (key) → OSRM public server (keyless) → None
  - Weather   : OpenWeatherMap (key) → Open-Meteo current (keyless)

Every provider is an adapter behind a common return shape and every call
degrades gracefully — the app never breaks because a key is missing or a
provider is down.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

import config
from modules import store

_log = config.get_logger(__name__)
_TIMEOUT = 15

MAPTILER_STYLE = "streets-v2-dark"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_OSRM_URL = "https://router.project-osrm.org/table/v1/driving/"
_HERE_MATRIX_URL = "https://matrix.router.hereapi.com/v8/matrix"
_OWM_URL = "https://api.openweathermap.org/data/2.5/weather"
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_USER_AGENT = "SupChainMate/5.x (open-source supply chain control tower)"

_last_nominatim_call = 0.0


# ── Basemap ────────────────────────────────────────────────────────────────────

def maptiler_key() -> Optional[str]:
    return config.get_env("MAPTILER_API_KEY")


def maptiler_tiles_url(style: str = MAPTILER_STYLE) -> Optional[str]:
    """XYZ raster tile URL for Leaflet/plotly, or None without a key."""
    key = maptiler_key()
    if not key:
        return None
    return f"https://api.maptiler.com/maps/{style}/{{z}}/{{x}}/{{y}}.png?key={key}"


def maptiler_attribution() -> str:
    return ("&copy; <a href='https://www.maptiler.com/'>MapTiler</a> "
            "&copy; <a href='https://www.openstreetmap.org/'>OpenStreetMap</a> contributors")


# ── Geocoding (Nominatim) ──────────────────────────────────────────────────────

def geocode(query: str) -> tuple[Optional[dict], str]:
    """
    Address → {lat, lon, display_name}. Keyless via Nominatim; results are
    cached in SQLite and calls are throttled to respect the usage policy.
    """
    q = str(query).strip()
    if not q:
        return None, "Enter an address or place name."
    cache_key = f"geocode::{q.lower()}"
    cached = store.load_setting(cache_key)
    if cached:
        return cached, "cached"

    global _last_nominatim_call
    wait = 1.1 - (time.monotonic() - _last_nominatim_call)
    if wait > 0:
        time.sleep(wait)
    try:
        _last_nominatim_call = time.monotonic()
        r = requests.get(_NOMINATIM_URL,
                         params={"q": q, "format": "json", "limit": 1},
                         headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT)
        r.raise_for_status()
        hits = r.json()
    except requests.exceptions.RequestException as e:
        return None, f"Geocoding unavailable: {type(e).__name__}"
    except ValueError:
        return None, "Geocoder returned unexpected data."
    if not hits:
        return None, f"No match for '{q}'."
    hit = {"lat": float(hits[0]["lat"]), "lon": float(hits[0]["lon"]),
           "display_name": str(hits[0].get("display_name", q))}
    store.save_setting(cache_key, hit)
    return hit, "nominatim"


# ── Road routing (HERE → OSRM) ─────────────────────────────────────────────────

def _osrm_matrix(coords: list[tuple[float, float]]) -> tuple[Optional[dict], str]:
    """Many-to-many road matrix from the OSRM public server (keyless)."""
    path = ";".join(f"{lon:.5f},{lat:.5f}" for lat, lon in coords)
    try:
        r = requests.get(_OSRM_URL + path,
                         params={"annotations": "distance,duration"},
                         headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT * 2)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        return None, f"OSRM unavailable: {type(e).__name__}"
    except ValueError:
        return None, "OSRM returned unexpected data."
    if data.get("code") != "Ok":
        return None, f"OSRM error: {data.get('code')}"
    return ({"distances_km": [[(d or 0) / 1000 for d in row]
                              for row in data.get("distances", [])],
             "durations_min": [[(d or 0) / 60 for d in row]
                               for row in data.get("durations", [])]},
            "OSRM public server (keyless)")


def _here_matrix(coords: list[tuple[float, float]],
                 api_key: str) -> tuple[Optional[dict], str]:
    """HERE Matrix API v8 with live traffic (synchronous, small matrices)."""
    body = {"origins": [{"lat": lat, "lng": lon} for lat, lon in coords],
            "regionDefinition": {"type": "world"},
            "matrixAttributes": ["distances", "travelTimes"]}
    try:
        r = requests.post(_HERE_MATRIX_URL, params={"apiKey": api_key, "async": "false"},
                          json=body, timeout=_TIMEOUT * 2)
        if r.status_code in (401, 403):
            return None, f"HERE rejected the key ({r.status_code})."
        r.raise_for_status()
        m = r.json().get("matrix", {})
    except requests.exceptions.RequestException as e:
        return None, f"HERE unavailable: {type(e).__name__}"
    except ValueError:
        return None, "HERE returned unexpected data."
    n = len(coords)
    dist, ttime = m.get("distances"), m.get("travelTimes")
    if not dist or len(dist) != n * n:
        return None, "HERE matrix incomplete."
    return ({"distances_km": [[dist[i * n + j] / 1000 for j in range(n)] for i in range(n)],
             "durations_min": [[ttime[i * n + j] / 60 for j in range(n)] for i in range(n)]},
            "HERE Matrix API (live traffic)")


def road_matrix(coords: list[tuple[float, float]]) -> tuple[Optional[dict], str]:
    """Best available road matrix: HERE with a key, else OSRM, else None."""
    if len(coords) < 2:
        return None, "Need at least two locations."
    here_key = config.get_env("HERE_API_KEY")
    if here_key:
        result, source = _here_matrix(coords, here_key)
        if result is not None:
            return result, source
        _log.warning("HERE matrix failed (%s) — falling back to OSRM", source)
    return _osrm_matrix(coords)


# ── Weather (OpenWeatherMap → Open-Meteo) ──────────────────────────────────────

def current_weather(lat: float, lon: float) -> tuple[Optional[dict], str]:
    """Current conditions at a point: {temp_c, wind_kmh, precip_mm, desc}."""
    owm_key = config.get_env("OPENWEATHER_API_KEY")
    if owm_key:
        try:
            r = requests.get(_OWM_URL, params={"lat": lat, "lon": lon,
                                               "appid": owm_key, "units": "metric"},
                             timeout=_TIMEOUT)
            if r.status_code != 401:
                r.raise_for_status()
                d = r.json()
                return ({"temp_c": float(d["main"]["temp"]),
                         "wind_kmh": float(d.get("wind", {}).get("speed", 0)) * 3.6,
                         "precip_mm": float(d.get("rain", {}).get("1h", 0)),
                         "desc": (d.get("weather") or [{}])[0].get("description", "")},
                        "OpenWeatherMap")
            _log.warning("OpenWeatherMap key rejected — falling back to Open-Meteo")
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            _log.warning("OpenWeatherMap failed (%s) — falling back to Open-Meteo", e)
    try:
        r = requests.get(_OPEN_METEO_URL,
                         params={"latitude": round(lat, 3), "longitude": round(lon, 3),
                                 "current": "temperature_2m,precipitation,wind_speed_10m"},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        c = r.json().get("current", {})
        return ({"temp_c": float(c.get("temperature_2m", 0)),
                 "wind_kmh": float(c.get("wind_speed_10m", 0)),
                 "precip_mm": float(c.get("precipitation", 0)),
                 "desc": ""}, "Open-Meteo (keyless)")
    except (requests.exceptions.RequestException, ValueError) as e:
        return None, f"Weather unavailable: {type(e).__name__}"


def weather_disruption_note(wx: dict) -> Optional[str]:
    """Plain-language delivery-risk note from current conditions."""
    if wx.get("precip_mm", 0) >= 5:
        return f"heavy precipitation ({wx['precip_mm']:.0f} mm) — expect delivery delays"
    if wx.get("wind_kmh", 0) >= 60:
        return f"high winds ({wx['wind_kmh']:.0f} km/h) — line-haul risk"
    if wx.get("temp_c", 20) <= -10:
        return f"extreme cold ({wx['temp_c']:.0f}°C) — vehicle and handling risk"
    return None
