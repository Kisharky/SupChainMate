"""
Geo-stack tests: MapTiler URL building, Nominatim geocoding (mocked +
cached), HERE/OSRM road matrices (mocked, incl. fallback), and the
weather adapter chain (OpenWeatherMap → Open-Meteo).
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules import geo, store


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "geo.db"))
    monkeypatch.setattr(geo, "_last_nominatim_call", 0.0)


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status
    def json(self):
        return self._payload
    def raise_for_status(self):
        if self.status_code >= 400:
            raise geo.requests.exceptions.HTTPError(str(self.status_code))


# ── Basemap ───────────────────────────────────────────────────────────────────

def test_maptiler_url(monkeypatch):
    monkeypatch.setattr(config, "get_env",
                        lambda n: "k123" if n == "MAPTILER_API_KEY" else None)
    url = geo.maptiler_tiles_url()
    assert url == ("https://api.maptiler.com/maps/streets-v2-dark/"
                   "{z}/{x}/{y}.png?key=k123")
    monkeypatch.setattr(config, "get_env", lambda n: None)
    assert geo.maptiler_tiles_url() is None


# ── Geocoding ─────────────────────────────────────────────────────────────────

def test_geocode_and_cache(monkeypatch):
    payload = [{"lat": "-23.56", "lon": "-46.65", "display_name": "São Paulo, Brazil"}]
    calls = {"n": 0}
    def fake_get(*a, **k):
        calls["n"] += 1
        return _FakeResp(payload)
    with patch.object(geo.requests, "get", side_effect=fake_get):
        hit, src = geo.geocode("São Paulo")
        assert src == "nominatim" and hit["lat"] == pytest.approx(-23.56)
        hit2, src2 = geo.geocode("São Paulo")   # served from SQLite cache
        assert src2 == "cached" and calls["n"] == 1
    assert geo.geocode("")[0] is None


def test_geocode_no_match():
    with patch.object(geo.requests, "get", return_value=_FakeResp([])):
        hit, msg = geo.geocode("xyzzy-nowhere")
    assert hit is None and "No match" in msg


# ── Road routing ──────────────────────────────────────────────────────────────

_COORDS = [(-23.5, -46.6), (-22.9, -43.2)]


def test_osrm_matrix_parse(monkeypatch):
    monkeypatch.setattr(config, "get_env", lambda n: None)  # no HERE key
    payload = {"code": "Ok",
               "distances": [[0, 430000], [432000, 0]],
               "durations": [[0, 18000], [18100, 0]]}
    with patch.object(geo.requests, "get", return_value=_FakeResp(payload)):
        matrix, src = geo.road_matrix(_COORDS)
    assert "OSRM" in src
    assert matrix["distances_km"][0][1] == pytest.approx(430.0)
    assert matrix["durations_min"][0][1] == pytest.approx(300.0)


def test_here_matrix_and_fallback(monkeypatch):
    monkeypatch.setattr(config, "get_env",
                        lambda n: "hk" if n == "HERE_API_KEY" else None)
    here_payload = {"matrix": {"distances": [0, 430000, 432000, 0],
                               "travelTimes": [0, 15000, 15100, 0]}}
    with patch.object(geo.requests, "post", return_value=_FakeResp(here_payload)):
        matrix, src = geo.road_matrix(_COORDS)
    assert "HERE" in src and matrix["distances_km"][0][1] == pytest.approx(430.0)

    # HERE rejects the key → falls back to OSRM
    osrm_payload = {"code": "Ok", "distances": [[0, 1000], [1000, 0]],
                    "durations": [[0, 60], [60, 0]]}
    with patch.object(geo.requests, "post", return_value=_FakeResp({}, status=401)), \
         patch.object(geo.requests, "get", return_value=_FakeResp(osrm_payload)):
        matrix, src = geo.road_matrix(_COORDS)
    assert "OSRM" in src and matrix["distances_km"][0][1] == pytest.approx(1.0)
    assert geo.road_matrix([_COORDS[0]])[0] is None


# ── Weather ───────────────────────────────────────────────────────────────────

def test_weather_owm_then_openmeteo(monkeypatch):
    monkeypatch.setattr(config, "get_env",
                        lambda n: "wk" if n == "OPENWEATHER_API_KEY" else None)
    owm = {"main": {"temp": 22.5}, "wind": {"speed": 5.0},
           "rain": {"1h": 1.2}, "weather": [{"description": "light rain"}]}
    with patch.object(geo.requests, "get", return_value=_FakeResp(owm)):
        wx, src = geo.current_weather(-23.5, -46.6)
    assert src == "OpenWeatherMap"
    assert wx["temp_c"] == 22.5 and wx["wind_kmh"] == pytest.approx(18.0)

    monkeypatch.setattr(config, "get_env", lambda n: None)  # keyless → Open-Meteo
    om = {"current": {"temperature_2m": 18.0, "precipitation": 0.0,
                      "wind_speed_10m": 12.0}}
    with patch.object(geo.requests, "get", return_value=_FakeResp(om)):
        wx, src = geo.current_weather(-23.5, -46.6)
    assert "Open-Meteo" in src and wx["temp_c"] == 18.0


def test_weather_disruption_notes():
    assert "precipitation" in geo.weather_disruption_note(
        {"precip_mm": 9.0, "wind_kmh": 10, "temp_c": 20})
    assert "winds" in geo.weather_disruption_note(
        {"precip_mm": 0, "wind_kmh": 70, "temp_c": 20})
    assert geo.weather_disruption_note(
        {"precip_mm": 0, "wind_kmh": 10, "temp_c": 20}) is None
