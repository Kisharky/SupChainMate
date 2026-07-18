"""
Core-module tests: decision engine, forecasting, optimisation, network
scoring, and configuration. Run with:  python -m pytest tests/ -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules import decisions, forecast, network, optimization


# ── Decision engine ───────────────────────────────────────────────────────────

def _profile(**overrides):
    base = dict(avg_daily_demand=100.0, std_daily_demand=20.0,
                avg_lead_time_days=7.0, std_lead_time_days=2.0,
                annual_demand=36500.0, horizon_forecast=700.0, horizon_days=7)
    base.update(overrides)
    return decisions.DemandProfile(**base)


def test_z_score_table_and_interpolation():
    assert decisions.z_score(0.95) == pytest.approx(1.645)
    assert decisions.z_score(0.99) == pytest.approx(2.326)
    # interpolated value sits between its neighbours
    z = decisions.z_score(0.925)
    assert decisions.z_score(0.90) < z < decisions.z_score(0.95)


def test_safety_stock_combined_variance_formula():
    out = decisions.run_decision_engine(_profile(), service_level=0.95)
    z = decisions.z_score(0.95)
    expected_ss = z * np.sqrt(7 * 20.0**2 + 100.0**2 * 2.0**2)
    assert out.safety_stock == pytest.approx(expected_ss, rel=1e-3)
    # ROP = mu_d * mu_LT + SS
    assert out.reorder_point == pytest.approx(100.0 * 7 + out.safety_stock, rel=1e-3)


def test_eoq_formula():
    out = decisions.run_decision_engine(
        _profile(), service_level=0.95,
        unit_cost=15.0, holding_rate=0.25, ordering_cost=200.0)
    expected_eoq = np.sqrt(2 * 36500.0 * 200.0 / (15.0 * 0.25))
    assert out.eoq == pytest.approx(expected_eoq, rel=1e-3)
    assert out.order_frequency_days == pytest.approx(out.eoq / 100.0, rel=1e-2)


def test_engine_monotonicity():
    """Higher service level and higher variability must both raise safety stock."""
    lo = decisions.run_decision_engine(_profile(), service_level=0.90)
    hi = decisions.run_decision_engine(_profile(), service_level=0.99)
    assert hi.safety_stock > lo.safety_stock
    calm = decisions.run_decision_engine(_profile(std_daily_demand=5.0))
    wild = decisions.run_decision_engine(_profile(std_daily_demand=50.0))
    assert wild.safety_stock > calm.safety_stock


def test_engine_recommendations_and_stockout_flag():
    surge = decisions.run_decision_engine(_profile(horizon_forecast=2000.0))
    cats = [r["category"] for r in surge.recommendations]
    assert "STOCKOUT RISK" in cats and surge.recommendations[0]["category"] == "STOCKOUT RISK"
    assert all(set(r) >= {"priority", "category", "action", "impact"}
               for r in surge.recommendations)


def test_retail_profile_builder():
    p = decisions.build_demand_profile_from_retail_inputs(70.0, 14.0, "High")
    assert p.avg_daily_demand == pytest.approx(10.0)
    assert p.std_daily_demand == pytest.approx(10.0 * 0.15, rel=1e-2)
    assert p.std_lead_time_days == pytest.approx(2.1, rel=1e-2)


def test_execution_plan_structure():
    out = decisions.run_decision_engine(_profile())
    plan = decisions.build_execution_plan(_profile(), out, 15.0, 200.0)
    assert {"Priority", "Category", "Action", "Impact", "Owner", "Target Date"} <= set(plan.columns)
    assert (plan["Priority"].values[:-1] <= plan["Priority"].values[1:]).all()


# ── Forecasting (non-Prophet paths) ───────────────────────────────────────────

def test_daily_demand_aggregation():
    orders = pd.DataFrame({"order_purchase_timestamp": pd.to_datetime(
        ["2025-01-01 09:00", "2025-01-01 15:00", "2025-01-02 10:00"])})
    daily = forecast.daily_demand(orders)
    assert list(daily.columns) == ["ds", "y", "external_signal"]
    # y = raw daily counts scaled by (1 + 0.5 x external_signal)
    expected = np.array([2.0, 1.0]) * (1.0 + 0.5 * daily["external_signal"].to_numpy())
    assert np.allclose(daily["y"].to_numpy(), expected)


def test_load_orders_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        forecast.load_orders("data/does_not_exist.csv")


def test_forecast_insights():
    daily = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=60, freq="D"),
                          "y": 100.0})
    fc = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=67, freq="D"),
                       "yhat": 110.0})
    ins = forecast.forecast_insights(fc, daily, horizon_days=7)
    assert ins["next_week_total"] == pytest.approx(770, rel=1e-2)


# ── Optimisation ──────────────────────────────────────────────────────────────

def test_network_summary():
    n = 50
    purchase = pd.date_range("2025-01-01", periods=n, freq="D")
    delivered = purchase + pd.Timedelta(days=5)
    estimated = purchase + pd.to_timedelta([7] * (n - 10) + [3] * 10, unit="D")
    orders = pd.DataFrame({
        "order_purchase_timestamp": purchase,
        "order_delivered_customer_date": delivered,
        "order_estimated_delivery_date": estimated,
    })
    s = optimization.network_summary(orders)
    assert s is not None and s["n_delivered_observed"] == n
    assert s["avg_lead_days"] == pytest.approx(5.0)
    assert s["on_time_pct"] == pytest.approx((n - 10) / n * 100)
    # no delivered dates → graceful None
    assert optimization.network_summary(
        orders.assign(order_delivered_customer_date=pd.NaT)) is None


# ── Network scoring ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def geo_points():
    rng = np.random.default_rng(4)
    # two dense clusters + a handful of isolated outliers
    a = rng.normal([-23.5, -46.6], 0.2, (120, 2))
    b = rng.normal([-12.9, -38.5], 0.2, (120, 2))
    outliers = rng.uniform([-30, -70], [0, -35], (6, 2))
    pts = np.vstack([a, b, outliers])
    return pd.DataFrame({"lat": pts[:, 0], "lon": pts[:, 1]})


def test_haversine_known_distance():
    # São Paulo → Rio de Janeiro ≈ 360 km
    d = network.haversine_km(-23.55, -46.63, -22.91, -43.17)
    assert 330 < d < 390
    assert network.haversine_km(0, 0, 0, 0) == 0


def test_clustering_and_centroid_metrics(geo_points):
    clustered = network.run_clustering(geo_points, n_clusters=3)
    assert clustered["cluster"].nunique() == 3
    stats = network.cluster_centroid_distances(clustered)
    assert {"customers", "avg_dist_km", "max_dist_km", "efficiency_score"} <= set(stats.columns)
    assert (stats["efficiency_score"] >= 0).all() and (stats["efficiency_score"] <= 100).all()
    assert stats["customers"].sum() == len(geo_points)


def test_isolation_forest_scores_range_and_outliers(geo_points):
    clustered = network.run_clustering(geo_points, n_clusters=3)
    scored = network.isolation_forest_risk_scores(clustered)
    assert scored["risk_score"].between(0, 100).all()
    # isolated points should score riskier than dense-cluster points on average
    dense_avg = scored.iloc[:240]["risk_score"].mean()
    outlier_avg = scored.iloc[240:]["risk_score"].mean()
    assert outlier_avg > dense_avg


# ── Configuration ─────────────────────────────────────────────────────────────

def test_config_get_env(monkeypatch):
    monkeypatch.setenv("SCM_TEST_VAR", "hello")
    assert config.get_env("SCM_TEST_VAR") == "hello"
    assert config.get_env("SCM_MISSING_VAR_XYZ") is None


def test_config_paths_exist():
    assert os.path.isdir(config.DATA_DIR)
    assert os.path.exists(config.DEMO_ORDERS)
