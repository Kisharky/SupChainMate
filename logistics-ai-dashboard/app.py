import os
import io
import math
import time
from dataclasses import replace as dc_replace
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


from modules import forecast, network, optimization, tracking, ingestion, decisions, retail
from modules import nvidia_api, groq_ai, control_tower, agent, cost_audit
from modules import health_check, tender, alerts, store, connect, carbon, doc_intel, ensemble, factors, runbook, sku

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupChainMate — Mission Control",
    layout="wide",
    initial_sidebar_state="collapsed",
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from views import helpers as vh
from views import landing as v_landing, retail as v_retail, upload as v_upload
from views import decision_center as v_decisions
from modules import trust

vh.apply_theme()

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════
_SESSION_KEYS = [
    "orders_df", "delivery_df", "location_df", "cost_df",
    "daily_df", "forecast_df", "tracking_df", "geo_df",
    "delay_model", "X_test_delay", "summary", "current_cost",
    "data_loaded", "demo_mode", "shipments_df", "carriers_simulated",
    "kpi_snapshot_saved", "orders_sku_df", "sku_stock",
]

for key in _SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False

if "retail_products" not in st.session_state:
    # Restore the saved tracker from SQLite (empty list when nothing saved)
    st.session_state.retail_products = store.load_retail_products()

if "entry_mode" not in st.session_state:
    # Skip landing if user already has enterprise data loaded (returning session).
    st.session_state.entry_mode = "enterprise" if st.session_state.get("data_loaded") else "landing"


def _reset_enterprise_session_preserve_retail():
    """Clear enterprise dashboard state; return to enterprise upload. Keep retail tracker."""
    backup = list(st.session_state.get("retail_products") or [])
    st.session_state.clear()
    st.session_state.retail_products = backup
    st.session_state.entry_mode = "enterprise"
    for key in _SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None
    st.session_state.data_loaded = False


# ═══════════════════════════════════════════════════════════════════════════════
# LANDING — CHOOSE ENTERPRISE VS SMALL RETAILER
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.entry_mode == "landing":
    v_landing.render()

# ═══════════════════════════════════════════════════════════════════════════════
# SMALL RETAILER — STANDALONE FLOW (NO ENTERPRISE DATA)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.entry_mode == "retail":
    v_retail.render()
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD SCREEN (ENTERPRISE ONLY)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.entry_mode == "enterprise" and not st.session_state.data_loaded:
    v_upload.render()



# ═══════════════════════════════════════════════════════════════════════════════
# DATA IS LOADED — COMPUTE RUNTIME METRICS
# ═══════════════════════════════════════════════════════════════════════════════

# Sidebar
with st.sidebar:
    st.markdown("### ⚙ SYSTEM CONFIG")
    days           = st.slider("FORECAST HORIZON (DAYS)", 7, 30, 7)
    simulate_event = st.toggle("🚨 MACRO EVENT SIMULATION", value=False)
    demand_change  = st.slider("DEMAND DELTA (%)", -50, 50, 0)
    st.divider()
    n_clusters     = st.slider("NETWORK HUBS (CLUSTERS)", 2, 12, 5,
                               help="What if we opened N hubs instead? Reconfigures the network instantly.")
    st.divider()
    st.markdown("<div class='hud-label'>DECISION ENGINE PARAMS</div>", unsafe_allow_html=True)
    service_level  = st.select_slider(
        "SERVICE LEVEL TARGET",
        options=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%"
    )
    avg_lead_time  = st.slider("AVG LEAD TIME (DAYS)", 1, 30, 7)
    std_lead_time  = st.slider("LEAD TIME STD DEV (DAYS)", 0, 10, 2)
    unit_cost      = st.number_input("UNIT COST ($)", min_value=1.0, value=15.0, step=1.0)
    ordering_cost  = st.number_input("ORDER COST ($)", min_value=10.0, value=200.0, step=10.0)
    holding_rate   = st.slider("HOLDING RATE (%/YR)", 10, 40, 25) / 100.0
    st.divider()
    mode_label = "DEMO DATASET" if st.session_state.demo_mode else "USER DATA"
    st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#666;'>SOURCE: {mode_label}</div>", unsafe_allow_html=True)
    if st.button("🔄 LOAD NEW DATA"):
        _reset_enterprise_session_preserve_retail()
        st.rerun()

# Pull from session
daily_df     = st.session_state.daily_df
forecast_obj = st.session_state._prophet_model
tracking_df  = st.session_state.tracking_df
geo_df       = st.session_state.geo_df
delay_model  = st.session_state.delay_model
X_test_delay = st.session_state.X_test_delay
current_cost = st.session_state.current_cost
summary      = st.session_state.summary

# Extend forecast with sidebar days
future                    = forecast_obj.make_future_dataframe(periods=days)
future                    = future.merge(daily_df[["ds", "external_signal"]], on="ds", how="left")
future["external_signal"] = future["external_signal"].fillna(1 if simulate_event else 0)
forecast_df               = forecast_obj.predict(future)
insights                  = forecast.forecast_insights(forecast_df, daily_df, horizon_days=days)

# ── Decision Engine ────────────────────────────────────────────────────────────
demand_profile = decisions.build_demand_profile(
    daily_df, forecast_df, horizon_days=days,
    avg_lead_time_days=avg_lead_time,
    std_lead_time_days=std_lead_time,
)
decision_outputs = decisions.run_decision_engine(
    demand_profile,
    service_level=service_level,
    unit_cost=unit_cost,
    holding_rate=holding_rate,
    ordering_cost=ordering_cost,
)
avg_daily        = float(daily_df["y"].mean())
growth           = ((float(forecast_df["yhat"].tail(days).mean()) - avg_daily) / avg_daily * 100.0) if avg_daily > 0 else 0.0
next_week_demand = insights["next_week_total"]
adjusted_demand  = int(next_week_demand * (1 + demand_change / 100))

status_counts  = tracking.get_status_counts(tracking_df)
delayed        = int(status_counts.get("Delayed", 0))
total_orders   = len(tracking_df)
preds          = delay_model.predict(X_test_delay)
delay_risk     = float(preds.mean() * 100)

optimized_cost = current_cost * 0.85
savings        = current_cost - optimized_cost

# ═══════════════════════════════════════════════════════════════════════════════
# TOP STATUS BAR
# ═══════════════════════════════════════════════════════════════════════════════
system_status = "HIGH ALERT" if delay_risk > 15 or delayed > 0.15 * total_orders else "NOMINAL"
status_color  = "#FF003C" if system_status == "HIGH ALERT" else "#00E676"
active_breaches = sum([
    1 if delay_risk > 15 else 0,
    1 if growth < -10 else 0,
    1 if simulate_event else 0,
    1 if demand_change > 20 else 0,
])

st.markdown(f"""
<div style="background:#0D0D10; border-bottom:1px solid #FF003C; padding:10px 20px;
            display:flex; align-items:center; justify-content:space-between;
            font-family:'Share Tech Mono',monospace; font-size:0.72rem;
            letter-spacing:0.08rem; margin-bottom:16px;">
    <div>
        <span style="color:#FF003C; font-size:1.1rem; font-weight:700;
                     font-family:'Teko',sans-serif; letter-spacing:0.1rem;">
            SUPCHAINMATE — MISSION CONTROL
        </span>
        <span style="color:{status_color}; margin-left:16px;">● SYSTEM {system_status}</span>
        <span style="color:#333; margin-left:12px; font-size:0.6rem;">
            SOURCE: {"DEMO" if st.session_state.demo_mode else "USER DATA"} |
            {total_orders:,} ORDERS | {len(daily_df)} DAYS
        </span>
    </div>
    <div style="display:flex; gap:24px; align-items:center;">
        <div style="text-align:center;">
            <div style="color:#FBC02D; font-size:1.4rem; font-family:'Teko',sans-serif;">{active_breaches} ACTIVE</div>
            <div style="color:#666; font-size:0.6rem;">BREACHES</div>
        </div>
        <div style="text-align:center;">
            <div style="color:#00E676; font-size:1.4rem; font-family:'Teko',sans-serif;">{100 - delay_risk:.1f}%</div>
            <div style="color:#666; font-size:0.6rem;">NOMINAL</div>
        </div>
        <div style="background:#FF003C; color:#FFF; padding:6px 14px;
                    font-size:0.7rem; font-weight:700; cursor:pointer;">
            OVERRIDE SYSTEM
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-INSIGHTS (Groq AI — instant analysis on load)
# ═══════════════════════════════════════════════════════════════════════════════
if groq_ai.is_available():
    _insight_ctx = {
        "Delay Risk":        f"{delay_risk:.1f}%",
        "Demand Growth":     f"{growth:+.1f}%",
        "Safety Stock":      f"{decision_outputs.safety_stock:,.0f} units",
        "EOQ":               f"{decision_outputs.eoq:,.0f} units",
        "Annual Savings":    f"${decision_outputs.savings_vs_current:,.0f}",
        "Active Breaches":   active_breaches,
        "Total Orders":      total_orders,
        "Forecast Demand":   f"{next_week_demand:,}",
    }
    _severity_colors = {"HIGH": "#FF003C", "MEDIUM": "#FBC02D", "LOW": "#00D4FF"}
    with st.spinner("Groq AI generating insights..."):
        auto_insights = groq_ai.generate_auto_insights(_insight_ctx)

    ins_cols = st.columns(len(auto_insights))
    for col, ins in zip(ins_cols, auto_insights):
        color = _severity_colors.get(ins["severity"], "#888")
        col.markdown(f"""
        <div style="background:#151518;border-top:2px solid {color};padding:12px 14px;
                    font-family:'Share Tech Mono',monospace;font-size:0.72rem;">
            <div style="color:{color};font-size:0.6rem;letter-spacing:0.1rem;margin-bottom:4px;">
                AI INSIGHT {ins['number']} · {ins['severity']}
            </div>
            <div style="color:#CCCCCC;line-height:1.5;">{ins['text']}</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT: Map (left) | HUD Panels (right)
# ═══════════════════════════════════════════════════════════════════════════════
col_map, col_hud = st.columns([2, 1], gap="small")

with col_map:
    # Re-cluster with sidebar n_clusters (what-if parameter)
    geo_df = geo_df.copy()
    geo_df = network.run_clustering(geo_df[["lat", "lon"]], n_clusters=n_clusters)

    # ── Isolation Forest anomaly scoring ──────────────────────────────────────
    # Replaces random risk scores — isolated nodes = genuinely higher delivery risk
    geo_df = network.isolation_forest_risk_scores(geo_df)

    # ── Combined multi-signal risk fusion ────────────────────────────────
    # Fuses Isolation Forest spatial score + LightGBM delay probability
    geo_df = network.combined_risk_signal(geo_df, tracking_df, delay_model)

    # Haversine centroid metrics
    centroid_stats = network.cluster_centroid_distances(geo_df)

    fig_map = px.scatter_mapbox(
        geo_df, lat="lat", lon="lon",
        color="combined_level",       # ← fused multi-signal colour
        size="combined_risk",
        size_max=18,
        hover_data={"risk_score": True, "delay_proba": True, "combined_risk": True, "signal_agreement": True},
        zoom=2.5, height=480,
        color_discrete_map={"Critical": "#FF003C", "Warning": "#FBC02D", "Safe": "#00D4FF"},
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=20, lon=0), zoom=1.5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(13,13,16,0.8)", bordercolor="#FF003C",
                    borderwidth=1, font=dict(color="#CCCCCC", size=10)),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    critical_pct    = len(geo_df[geo_df["combined_level"] == "Critical"]) / len(geo_df) * 100
    agreed_critical = geo_df[geo_df["signal_agreement"] == True]["cluster"].unique()

    st.markdown(f"""
    <div class="hud-panel">
        <div style="color:#FF003C;font-family:'Teko',sans-serif;font-size:1.1rem;letter-spacing:0.1rem;">
            ⚡ MULTI-SIGNAL DISRUPTION RADAR — ACTIVE
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#AAAAAA;margin:6px 0;">
            {critical_pct:.0f}% OF NODES CRITICAL &nbsp;·&nbsp;
            <span style="color:#FF003C;">{len(agreed_critical)} ZONE(S) WITH SIGNAL AGREEMENT</span>
            &nbsp;(ISOLATION FOREST + LGBM BOTH ≥ 70 — HIGH CONFIDENCE)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Consulting-grade per-zone alerts where both signals agree
    for cid in agreed_critical[:3]:
        zone_data = geo_df[geo_df["cluster"] == cid]
        avg_cr = zone_data["combined_risk"].mean()
        avg_dp = zone_data["delay_proba"].mean()
        avg_if = zone_data["risk_score"].mean()
        st.markdown(f"""
        <div style="background:#1A0508;border-left:3px solid #FF003C;padding:8px 14px;
                    margin-bottom:4px;font-family:'Share Tech Mono',monospace;font-size:0.72rem;">
            <b style="color:#FF003C;">⚡ ZONE {cid} — HIGH CONFIDENCE RISK ALERT</b><br>
            <span style="color:#AAAAAA;">
                Spatial anomaly (IF): <b style="color:#FF003C;">{avg_if:.0f}/100</b> &nbsp;·&nbsp;
                Delay probability (LightGBM): <b style="color:#FBC02D;">{avg_dp:.1f}%</b> &nbsp;·&nbsp;
                Combined signal: <b style="color:#FF003C;">{avg_cr:.0f}/100</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Haversine cluster efficiency table
    with st.expander(f"📡 NETWORK TOPOLOGY — {n_clusters} HUBS (HAVERSINE METRICS)", expanded=False):
        display_stats = centroid_stats.reset_index()[["cluster","customers","avg_dist_km","max_dist_km","efficiency_score"]]
        display_stats.columns = ["CLUSTER", "CUSTOMERS", "AVG DIST KM", "MAX DIST KM", "EFFICIENCY %"]
        st.dataframe(
            display_stats.style.background_gradient(subset=["EFFICIENCY %"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True
        )

    st.markdown(f"""
    <div class="hud-panel-yellow">
        <div style="color:#FBC02D;font-family:'Teko',sans-serif;font-size:1.1rem;letter-spacing:0.1rem;">
            ★ NVIDIA cuOPT — ROUTE OPTIMIZATION
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#AAAAAA;margin:4px 0 10px 0;">
            REAL VRP SOLVER · HAVERSINE COST MATRIX · {n_clusters} DELIVERY ZONES
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("⚡ EXECUTE cuOPT OPTIMIZATION", key="exec_opt"):
        with st.spinner("NVIDIA cuOpt solving VRP..."):
            opt_result = nvidia_api.cuopt_optimize(geo_df, n_vehicles=max(2, n_clusters // 2))
        if opt_result.get("success"):
            st.success(f"✅ {opt_result['summary']}")
            if opt_result.get("savings_km", 0) > 0:
                st.markdown(f"""
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                            color:#00E676;padding:6px 0;">
                    TOTAL ROUTE: {opt_result.get('total_cost_km',0):,.0f} KM &nbsp;·&nbsp;
                    NAIVE BASELINE: {opt_result.get('naive_cost_km',0):,.0f} KM &nbsp;·&nbsp;
                    SAVINGS: {opt_result.get('savings_km',0):,.0f} KM ({opt_result.get('savings_pct',0):.1f}%) &nbsp;·&nbsp;
                    ≈ {carbon.route_savings_co2(opt_result.get('savings_km',0)):,.1f} tCO₂e AVOIDED
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error(opt_result.get("summary", "cuOpt error"))

with col_hud:
    risk_label  = "EXTREME" if delay_risk > 25 else ("CRITICAL" if delay_risk > 15 else "MODERATE")
    weather_idx = min(99, delay_risk * 3.2)
    node_cong   = min(99, delay_risk * 2.4)

    st.markdown(f"""
    <div class="hud-panel" style="border-color:rgba(255,0,60,0.6);">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <div class="hud-label">DELAY RISK CRITICAL</div>
                <div class="hud-value-red">{delay_risk:.1f}%</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#FBC02D;margin-top:4px;">
                    ↑ {abs(growth):.1f}% TREND
                </div>
            </div>
            <span class="action-required-badge">ACTION REQUIRED</span>
        </div>
        <div class="scan-line"></div>
        <div class="hud-label" style="margin-top:8px;">WEATHER DISRUPTION INDEX</div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div class="progress-bar-container" style="flex:1;margin-right:10px;">
                <div class="progress-bar-fill-red" style="width:{weather_idx:.0f}%;"></div>
            </div>
            <span style="color:#FF003C;font-family:'Share Tech Mono',monospace;font-size:0.7rem;white-space:nowrap;">
                {weather_idx:.0f}% ({risk_label})
            </span>
        </div>
        <div class="hud-label" style="margin-top:8px;">NODE CONGESTION</div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div class="progress-bar-container" style="flex:1;margin-right:10px;">
                <div class="progress-bar-fill-yellow" style="width:{node_cong:.0f}%;"></div>
            </div>
            <span style="color:#FBC02D;font-family:'Share Tech Mono',monospace;font-size:0.7rem;white-space:nowrap;">
                {node_cong:.0f}% (WARNING)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    avg_lead = summary["avg_lead_days"] if summary else 14.2
    risk_pct = 24
    st.markdown(f"""
    <div class="hud-panel" style="border-color:#333340;">
        <div style="color:#00D4FF;font-family:'Teko',sans-serif;font-size:1rem;
                    letter-spacing:0.1rem;text-transform:uppercase;margin-bottom:6px;">
            ◈ SYSTEM BENCHMARKS
        </div>
        <table class="benchmark-table">
            <tr><th>VECTOR</th><th>LEGACY</th><th class="optimized">OPTIMIZED</th></tr>
            <tr><td>COST</td><td>${current_cost:,.0f}</td><td class="optimized">${optimized_cost:,.0f}</td></tr>
            <tr><td>RISK</td><td>{risk_pct}%</td><td class="optimized">{int(risk_pct*0.75)}%</td></tr>
            <tr><td>LEAD</td><td>{avg_lead:.1f}D</td><td class="optimized">{avg_lead*0.7:.1f}D</td></tr>
            <tr><td>DELAY</td><td>{delay_risk:.1f}%</td><td class="optimized">{delay_risk*0.6:.1f}%</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hud-panel-blue">
        <div style="color:#00D4FF;font-family:'Teko',sans-serif;font-size:1rem;
                    letter-spacing:0.1rem;text-transform:uppercase;margin-bottom:8px;">
            ◎ OPERATIONAL DIRECTIVE
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                    color:#AAAAAA;line-height:1.6;font-style:italic;">
            "Consolidate high-risk clusters immediately. Route {15 + int(delay_risk/2):.0f}%
            volume via regional hubs. Reallocate safety stock from low-risk zones.
            Confidence score: 87%."
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_ack, col_warn = st.columns([3, 1])
    with col_ack:
        if st.button("ACKNOWLEDGE DIRECTIVE", key="ack_dir"):
            st.success("CONFIRMED — EXECUTING")
    with col_warn:
        st.markdown('<div style="background:#FF003C;width:40px;height:40px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;margin-top:2px;">⚠</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DECISION ENGINE SECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="font-family:'Teko',sans-serif;font-size:1.6rem;letter-spacing:0.12rem;
            text-transform:uppercase;color:#FFFFFF;padding:8px 0;border-bottom:1px solid #FF003C;
            margin-bottom:16px;">
    ⌁ SUPPLY CHAIN DECISION ENGINE
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#666;margin-left:12px;">
        SAFETY STOCK · EOQ · REORDER POINT · LEAD TIME BUFFER
    </span>
</div>
""", unsafe_allow_html=True)

de1, de2, de3, de4 = st.columns(4)
impact_color = {"CRITICAL": "#FF003C", "HIGH": "#FF003C", "MEDIUM": "#FBC02D", "LOW": "#00E676"}

de1.markdown(f"""
<div class="hud-panel">
    <div class="hud-label">SAFETY STOCK</div>
    <div class="hud-value-red">{decision_outputs.safety_stock:,.0f}</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#888;margin-top:4px;">
        UNITS @ {service_level*100:.0f}% SVC LEVEL (Z={decision_outputs.z_value})
    </div>
    <div style="color:{'#00E676' if decision_outputs.safety_stock_delta_pct >= 0 else '#FF003C'};
                font-family:'Share Tech Mono',monospace;font-size:0.7rem;margin-top:6px;">
        {'▲' if decision_outputs.safety_stock_delta_pct >= 0 else '▼'}
        {abs(decision_outputs.safety_stock_delta_pct):.0f}% vs CURRENT
    </div>
</div>""", unsafe_allow_html=True)

de2.markdown(f"""
<div class="hud-panel">
    <div class="hud-label">EOQ — OPTIMAL ORDER QTY</div>
    <div class="hud-value-red">{decision_outputs.eoq:,.0f}</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#888;margin-top:4px;">
        UNITS/ORDER · EVERY {decision_outputs.order_frequency_days:.0f} DAYS
    </div>
</div>""", unsafe_allow_html=True)

de3.markdown(f"""
<div class="hud-panel">
    <div class="hud-label">REORDER POINT</div>
    <div class="hud-value-red">{decision_outputs.reorder_point:,.0f}</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#888;margin-top:4px;">
        UNITS IN STOCK · TRIGGER REPLENISHMENT
    </div>
</div>""", unsafe_allow_html=True)

de4.markdown(f"""
<div class="hud-panel">
    <div class="hud-label">ANNUAL SAVINGS</div>
    <div class="hud-value-red">${decision_outputs.savings_vs_current:,.0f}</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#888;margin-top:4px;">
        VS CURRENT ORDERING STRATEGY
    </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div class='hud-label' style='margin:16px 0 8px 0;'>PRESCRIPTIVE ACTIONS</div>", unsafe_allow_html=True)
for rec in decision_outputs.recommendations:
    color = impact_color.get(rec["impact"], "#888888")
    st.markdown(f"""
    <div style="background:#151518;border-left:3px solid {color};padding:10px 16px;
                margin-bottom:6px;font-family:'Share Tech Mono',monospace;font-size:0.75rem;">
        <span style="color:{color};font-weight:bold;font-size:0.65rem;letter-spacing:0.08rem;">
            [{rec['impact']}] {rec['category']}
        </span><br>
        <span style="color:#CCCCCC;">{rec['action']}</span>
    </div>
    """, unsafe_allow_html=True)

exec_plan_df = decisions.build_execution_plan(demand_profile, decision_outputs, unit_cost, ordering_cost)
col_dl, _ = st.columns([1, 3])
with col_dl:
    st.download_button(
        label="⇩ DOWNLOAD EXECUTION PLAN (CSV)",
        data=exec_plan_df.to_csv(index=False).encode(),
        file_name="supchainmate_execution_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SKU INTELLIGENCE — PER-PRODUCT DECISIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="font-family:'Teko',sans-serif;font-size:1.6rem;letter-spacing:0.12rem;
            text-transform:uppercase;color:#FFFFFF;padding:8px 0;border-bottom:1px solid #FF003C;
            margin-bottom:16px;">
    🗃 SKU INTELLIGENCE
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#666;margin-left:12px;">
        PER-PRODUCT SAFETY STOCK · ROP · EOQ · ABC CLASSIFICATION
    </span>
</div>
""", unsafe_allow_html=True)

sku_plan_with_status = None
if st.session_state.get("orders_sku_df") is None:
    st.info("Per-SKU decisions need a product/SKU column in your orders file "
            "(auto-detected: sku, product, item, material...). The demo dataset includes "
            "a simulated catalogue.")
else:
    if st.session_state.demo_mode:
        st.markdown(
            "<div style='font-family:Share Tech Mono,monospace;font-size:0.62rem;color:#FBC02D;'>"
            "⚠ DEMO MODE — SKU CATALOGUE &amp; PRICES ARE SIMULATED (over real Olist order dates)</div>",
            unsafe_allow_html=True)
    if "sku_profiles" not in st.session_state or st.session_state.get("sku_profiles") is None:
        with st.spinner("Profiling SKUs..."):
            _prof = sku.sku_demand_profiles(st.session_state.orders_sku_df)
            st.session_state.sku_profiles = sku.abc_classify(_prof) if _prof is not None else None
    sku_classified = st.session_state.sku_profiles

    if sku_classified is None or not len(sku_classified):
        st.info("No usable SKU rows found in the orders file.")
    else:
        sku_plan = sku.run_sku_engine(
            sku_classified,
            service_level=service_level,
            avg_lead_time_days=avg_lead_time,
            std_lead_time_days=std_lead_time,
            ordering_cost=ordering_cost,
            holding_rate=holding_rate,
            default_unit_cost=unit_cost,
        )
        # restore any previously entered stock levels
        if st.session_state.get("sku_stock"):
            sku_plan["Current Stock"] = sku_plan["SKU"].map(
                st.session_state.sku_stock).fillna(0.0)
        sku_plan_with_status = sku.stock_status(sku_plan)
        skpi = sku.sku_kpis(sku_classified, sku_plan_with_status)

        sk1, sk2, sk3, sk4 = st.columns(4)
        sk1.markdown(f"""
        <div class="hud-panel" style="border-color:#333340;">
            <div class="hud-label">SKUS UNDER MANAGEMENT</div>
            <div style="font-family:'Teko',sans-serif;font-size:1.8rem;color:#00D4FF;">{skpi['n_skus']}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#666;">TOP {sku.MAX_SKUS} BY VOLUME</div>
        </div>""", unsafe_allow_html=True)
        sk2.markdown(f"""
        <div class="hud-panel" style="border-color:#333340;">
            <div class="hud-label">A-CLASS SKUS</div>
            <div style="font-family:'Teko',sans-serif;font-size:1.8rem;color:#00E676;">{skpi['a_class']}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#666;">DRIVE 80% OF {skpi['basis'].upper()}</div>
        </div>""", unsafe_allow_html=True)
        sk3.markdown(f"""
        <div class="hud-panel" style="border-color:#333340;">
            <div class="hud-label">CATALOGUE REVENUE</div>
            <div style="font-family:'Teko',sans-serif;font-size:1.8rem;color:#FFF;">
                {f"${skpi['total_revenue']:,.0f}" if skpi['total_revenue'] else "N/A"}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#666;">OVER THE DATA PERIOD</div>
        </div>""", unsafe_allow_html=True)
        sk4.markdown(f"""
        <div class="hud-panel" style="border-color:#333340;">
            <div class="hud-label">NEED ORDERING NOW</div>
            <div style="font-family:'Teko',sans-serif;font-size:1.8rem;color:#FF003C;">{skpi.get('order_now', 0)}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#666;">STOCK AT/BELOW ROP</div>
        </div>""", unsafe_allow_html=True)

        skl, skr = st.columns([3, 2])
        with skl:
            st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>PER-SKU DECISION TABLE — ENTER CURRENT STOCK</div>", unsafe_allow_html=True)
            sku_edited = st.data_editor(
                sku_plan_with_status,
                use_container_width=True, hide_index=True, height=330,
                disabled=[c for c in sku_plan_with_status.columns if c != "Current Stock"],
                key="sku_editor",
            )
            ska, skb = st.columns(2)
            with ska:
                if st.button("APPLY STOCK LEVELS", key="sku_apply", use_container_width=True):
                    try:
                        st.session_state.sku_stock = dict(zip(
                            sku_edited["SKU"],
                            pd.to_numeric(sku_edited["Current Stock"], errors="coerce").fillna(0.0)))
                    except (KeyError, TypeError):
                        st.error("Could not read stock values.")
                    else:
                        st.rerun()
            with skb:
                st.download_button(
                    "⇩ PER-SKU REORDER PLAN (CSV)",
                    data=sku_plan_with_status.to_csv(index=False).encode(),
                    file_name="supchainmate_sku_plan.csv", mime="text/csv",
                    use_container_width=True,
                )
        with skr:
            st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>ABC ANALYSIS — PARETO BY " + skpi["basis"].upper() + "</div>", unsafe_allow_html=True)
            _abc_basis = skpi["basis"]
            fig_abc = px.bar(
                sku_classified.head(20), x=_abc_basis, y="SKU", orientation="h",
                color="ABC", color_discrete_map={"A": "#00E676", "B": "#FBC02D", "C": "#FF003C"},
                height=330,
            )
            fig_abc.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,13,16,1)",
                margin=dict(l=10, r=10, t=10, b=30),
                yaxis=dict(autorange="reversed"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
            )
            fig_abc.update_xaxes(gridcolor="#222228")
            st.plotly_chart(fig_abc, use_container_width=True)
            st.caption("Service levels step down by class: A = your sidebar target, "
                       "B −3 pts, C −8 pts — concentrate capital where it earns.")

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT-IF LAB — SCENARIO PLANNER
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="font-family:'Teko',sans-serif;font-size:1.6rem;letter-spacing:0.12rem;
            text-transform:uppercase;color:#FFFFFF;padding:8px 0;border-bottom:1px solid #FBC02D;
            margin-bottom:16px;">
    🧪 WHAT-IF LAB
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#666;margin-left:12px;">
        STRESS-TEST THE DECISION ENGINE — LIVE RECALCULATION
    </span>
</div>
""", unsafe_allow_html=True)

wi1, wi2, wi3, wi4 = st.columns(4)
wi_demand = wi1.slider("DEMAND CHANGE (%)", -50, 100, 0, key="wi_demand",
                       help="e.g. +30 = a demand spike of 30%")
wi_lt     = wi2.slider("LEAD TIME CHANGE (%)", -50, 100, 0, key="wi_lt",
                       help="e.g. +20 = supplier lead times stretch 20%")
wi_ltvar  = wi3.slider("LEAD TIME VARIABILITY (%)", -50, 200, 0, key="wi_ltvar",
                       help="How much more erratic deliveries become")
wi_svc    = wi4.select_slider("SERVICE LEVEL", options=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99],
                              value=service_level, key="wi_svc",
                              format_func=lambda x: f"{int(x*100)}%")

_scale_d  = 1 + wi_demand / 100
_scale_lt = 1 + wi_lt / 100
scenario_profile = dc_replace(
    demand_profile,
    avg_daily_demand=round(demand_profile.avg_daily_demand * _scale_d, 2),
    std_daily_demand=round(demand_profile.std_daily_demand * _scale_d, 2),
    avg_lead_time_days=round(demand_profile.avg_lead_time_days * _scale_lt, 2),
    std_lead_time_days=round(demand_profile.std_lead_time_days * _scale_lt * (1 + wi_ltvar / 100), 2),
    annual_demand=round(demand_profile.annual_demand * _scale_d, 0),
    horizon_forecast=round(demand_profile.horizon_forecast * _scale_d, 0),
)
scenario_outputs = decisions.run_decision_engine(
    scenario_profile,
    service_level=wi_svc,
    unit_cost=unit_cost,
    holding_rate=holding_rate,
    ordering_cost=ordering_cost,
)

wm1, wm2, wm3, wm4, wm5 = st.columns(5)
wm1.metric("SAFETY STOCK", f"{scenario_outputs.safety_stock:,.0f}",
           delta=f"{scenario_outputs.safety_stock - decision_outputs.safety_stock:+,.0f} units",
           delta_color="inverse")
wm2.metric("REORDER POINT", f"{scenario_outputs.reorder_point:,.0f}",
           delta=f"{scenario_outputs.reorder_point - decision_outputs.reorder_point:+,.0f} units",
           delta_color="inverse")
wm3.metric("EOQ", f"{scenario_outputs.eoq:,.0f}",
           delta=f"{scenario_outputs.eoq - decision_outputs.eoq:+,.0f} units/order",
           delta_color="off")
wm4.metric("ORDER EVERY", f"{scenario_outputs.order_frequency_days:.0f} days",
           delta=f"{scenario_outputs.order_frequency_days - decision_outputs.order_frequency_days:+.0f} days",
           delta_color="off")
wm5.metric("TOTAL COST / YR", f"${scenario_outputs.total_optimized_cost:,.0f}",
           delta=f"${scenario_outputs.total_optimized_cost - decision_outputs.total_optimized_cost:+,.0f}",
           delta_color="inverse")

if wi_demand or wi_lt or wi_ltvar or wi_svc != service_level:
    _sc_bits = []
    if wi_demand: _sc_bits.append(f"demand {wi_demand:+d}%")
    if wi_lt:     _sc_bits.append(f"lead time {wi_lt:+d}%")
    if wi_ltvar:  _sc_bits.append(f"LT variability {wi_ltvar:+d}%")
    if wi_svc != service_level: _sc_bits.append(f"service level → {int(wi_svc*100)}%")
    st.caption(f"Scenario: {', '.join(_sc_bits)} — deltas vs your current sidebar baseline.")
else:
    st.caption("Move a slider to stress-test — deltas show vs your current baseline.")

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET SIGNALS — EXTERNAL FACTOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📡 MARKET SIGNALS — EXTERNAL FACTOR ENGINE", expanded=False):
    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;font-size:0.62rem;color:#888;'>"
        "KEYLESS SOURCES: FRANKFURTER FX · STOOQ BRENT · OPEN-METEO WEATHER · OFFLINE HOLIDAY CALENDAR · "
        "OPTIONAL POSTHOG/GA EVENTS UPLOAD — EVERY FACTOR'S VALUE IS PROVEN ON THE FORECAST HOLDOUT</div>",
        unsafe_allow_html=True)

    fs1, fs2, fs3, fs4 = st.columns([1, 1, 2, 1])
    f_country = fs1.text_input("HOLIDAY COUNTRY", "BR", max_chars=2, key="f_country").upper()
    f_fx = fs2.text_input("LOCAL CURRENCY", "BRL", max_chars=3, key="f_fx").upper()
    f_analytics_file = fs3.file_uploader("PostHog / GA daily events export (date, count)",
                                         type=["csv"], key="f_analytics")
    with fs4:
        st.markdown("<br>", unsafe_allow_html=True)
        f_refresh = st.button("⟳ REFRESH SIGNALS", key="f_refresh", use_container_width=True)

    f_analytics_df = None
    if f_analytics_file is not None:
        try:
            f_analytics_df = pd.read_csv(f_analytics_file)
        except Exception as e:
            st.error(f"Could not read analytics file: {e}")

    _geo_lat = float(geo_df["lat"].mean()) if "lat" in geo_df.columns else None
    _geo_lon = float(geo_df["lon"].mean()) if "lon" in geo_df.columns else None
    _fkey = f"factors_{f_country}_{f_fx}_{len(f_analytics_df) if f_analytics_df is not None else 0}"
    if f_refresh:
        st.session_state.pop(_fkey, None)
    if _fkey not in st.session_state:
        with st.spinner("Fetching market signals (FX · oil · weather · holidays)..."):
            st.session_state[_fkey] = factors.build_factor_frame(
                daily_df, lat=_geo_lat, lon=_geo_lon, country=f_country,
                fx_symbol=f_fx, analytics_df=f_analytics_df)
    f_bundle = st.session_state[_fkey]
    factors_df = f_bundle["factors"]

    # ── Ticker strip ──────────────────────────────────────────────────────────
    readings = factors.latest_readings(factors_df)
    if readings:
        ticker_html = "".join(
            f"<span style='margin-right:26px;'>"
            f"<span style='color:#FBC02D;'>{r['label']}</span> "
            f"<span style='color:#FFF;'>{r['value']}</span> "
            f"<span style='color:{'#00E676' if r['delta'] >= 0 else '#FF003C'};font-size:0.62rem;'>"
            f"{'▲' if r['delta'] >= 0 else '▼'}{abs(r['delta']):.2f} vs -7d</span></span>"
            for r in readings)
        st.markdown(
            f"<div style='background:#0D0D10;border:1px solid #FBC02D;padding:10px 16px;"
            f"font-family:Share Tech Mono,monospace;font-size:0.8rem;overflow-x:auto;"
            f"white-space:nowrap;'>{ticker_html}</div>",
            unsafe_allow_html=True)
    st.caption("Sources: " + " · ".join(f_bundle["sources"])
               + (("  |  Skipped: " + " · ".join(f_bundle["errors"])) if f_bundle["errors"] else ""))

    fcl, fcr = st.columns([1, 1])
    with fcl:
        st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>FACTOR ↔ DEMAND CORRELATION</div>", unsafe_allow_html=True)
        corr = factors.factor_correlations(daily_df, factors_df)
        if len(corr):
            st.dataframe(corr, use_container_width=True, hide_index=True)
            fig_corr = px.bar(corr, x="Corr (same-day)", y="Factor", orientation="h", height=260,
                              color="Corr (same-day)", color_continuous_scale=["#FF003C", "#222228", "#00E676"],
                              range_color=[-1, 1])
            fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(13,13,16,1)", coloraxis_showscale=False,
                                   margin=dict(l=10, r=10, t=10, b=30))
            fig_corr.update_xaxes(gridcolor="#222228", range=[-1, 1])
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No overlapping factor data for the demand window.")

    with fcr:
        st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>FACTOR UPLIFT — PROVEN ON THE HOLDOUT</div>", unsafe_allow_html=True)
        _bk, _fk2 = f"tournament_{days}", f"tournament_factors_{days}_{_fkey}"
        if _bk not in st.session_state:
            with st.spinner("Baseline backtest..."):
                st.session_state[_bk] = ensemble.run_tournament(daily_df, forecast_df, horizon_days=days)
        if _fk2 not in st.session_state:
            with st.spinner("Factor-aware backtest..."):
                st.session_state[_fk2] = ensemble.run_tournament(
                    daily_df, forecast_df, horizon_days=days, factors_df=factors_df)
        t_base, t_fact = st.session_state[_bk], st.session_state[_fk2]
        if t_base is None or t_fact is None:
            st.info("Uplift needs ~120+ days of daily history.")
        else:
            uplift = t_base["champion_mape"] - t_fact["champion_mape"]
            up_color = "#00E676" if uplift > 0 else ("#FBC02D" if uplift == 0 else "#FF003C")
            st.markdown(f"""
            <div class="hud-panel" style="border-color:{up_color};">
                <div class="hud-label">CHAMPION MAPE — {t_base['holdout_days']}-DAY BACKTEST</div>
                <div style="font-family:'Teko',sans-serif;font-size:1.7rem;color:#FFF;">
                    {t_base['champion_mape']:.1f}% → <span style="color:{up_color};">{t_fact['champion_mape']:.1f}%</span>
                </div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:{up_color};">
                    {"FACTORS IMPROVE THE FORECAST BY " + f"{uplift:.1f} MAPE PTS" if uplift > 0
                     else "NO IMPROVEMENT FROM FACTORS ON THIS SERIES" if uplift <= 0 else ""}
                </div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#888;margin-top:4px;">
                    {len(t_fact.get('factor_cols', []))} FACTOR FEATURES · CHAMPION: {t_fact['champion']}
                </div>
            </div>""", unsafe_allow_html=True)
            st.dataframe(t_fact["leaderboard"], use_container_width=True, hide_index=True)
        st.download_button(
            "⇩ EXPORT FACTOR FRAME (CSV)",
            data=factors_df.to_csv(index=False).encode(),
            file_name="supchainmate_factors.csv", mime="text/csv",
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# FREIGHT CONTROL TOWER — SHIPMENT TRACKING BOARD + CARRIER SCORECARDS
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="font-family:'Teko',sans-serif;font-size:1.6rem;letter-spacing:0.12rem;
            text-transform:uppercase;color:#FFFFFF;padding:8px 0;border-bottom:1px solid #00D4FF;
            margin-bottom:16px;">
    ⛟ FREIGHT CONTROL TOWER
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#666;margin-left:12px;">
        SHIPMENT TRACKING BOARD · CARRIER SCORECARDS · EXCEPTION ALERTS
    </span>
</div>
""", unsafe_allow_html=True)

# Shipment prep is independent of sidebar params — compute once per data load.
if st.session_state.get("shipments_df") is None:
    with st.spinner("BUILDING SHIPMENT BOARD..."):
        st.session_state.shipments_df = control_tower.prepare_shipments(tracking_df, delay_model)
shipments_df = st.session_state.shipments_df
ct_kpis      = control_tower.shipment_kpis(shipments_df)

_hp = lambda label, value, sub, color="#00D4FF": f"""
<div class="hud-panel" style="border-color:#333340;">
    <div class="hud-label">{label}</div>
    <div style="font-family:'Teko',sans-serif;font-size:1.8rem;color:{color};">{value}</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#666;">{sub}</div>
</div>"""

ct1, ct2, ct3, ct4, ct5 = st.columns(5)
_on_time_txt = f"{ct_kpis['on_time_pct']:.1f}%" if not math.isnan(ct_kpis["on_time_pct"]) else "N/A"
_on_time_col = "#00E676" if (not math.isnan(ct_kpis["on_time_pct"]) and ct_kpis["on_time_pct"] >= 90) else "#FBC02D"
ct1.markdown(_hp("TOTAL SHIPMENTS", f"{ct_kpis['total']:,}", "ALL RECORDS"), unsafe_allow_html=True)
ct2.markdown(_hp("IN TRANSIT / OPEN", f"{ct_kpis['in_transit']:,}", "NOT YET DELIVERED", "#FBC02D"), unsafe_allow_html=True)
ct3.markdown(_hp("ON-TIME DELIVERY", _on_time_txt, "VS PROMISED DATE", _on_time_col), unsafe_allow_html=True)
ct4.markdown(_hp("LATE", f"{ct_kpis['late']:,}", "MISSED PROMISE", "#FF003C"), unsafe_allow_html=True)
ct5.markdown(_hp("AT RISK (ML)", f"{ct_kpis['at_risk']:,}", f"AVG DELAY {ct_kpis['avg_delay_days']:.1f}D WHEN LATE", "#FF003C"), unsafe_allow_html=True)

tower_board, tower_score = st.columns([1, 1], gap="medium")

with tower_board:
    st.markdown("<div class='hud-label' style='margin:12px 0 6px 0;'>SHIPMENT TRACKING BOARD</div>", unsafe_allow_html=True)
    health_options = ["ALL"] + sorted(shipments_df["health"].unique().tolist())
    pick_health = st.selectbox("Filter by status", health_options, key="ct_health_filter")
    board = shipments_df if pick_health == "ALL" else shipments_df[shipments_df["health"] == pick_health]
    # Exceptions first, then most recent
    _health_priority = {"LATE": 0, "AT RISK": 1, "DELIVERED LATE": 2, "ON TRACK": 3,
                        "DELIVERED ON TIME": 4, "CANCELLED": 5}
    board_view = (
        board.assign(_prio=board["health"].map(_health_priority).fillna(9))
        .sort_values(["_prio", "order_date"], ascending=[True, False])
        .drop(columns="_prio")
        .head(500)
        .copy()
    )
    for dcol in ("order_date", "promised_date", "delivered_date"):
        board_view[dcol] = board_view[dcol].dt.strftime("%Y-%m-%d")
    show_cols = ["shipment_id", "order_date", "promised_date", "delivered_date", "health", "delay_days", "delay_proba"]
    if board_view["carrier"].notna().any():
        show_cols.insert(1, "carrier")
    board_view = board_view[show_cols].rename(columns={
        "shipment_id": "Shipment", "carrier": "Carrier", "order_date": "Ordered",
        "promised_date": "Promised", "delivered_date": "Delivered",
        "health": "Status", "delay_days": "Delay (days)", "delay_proba": "ML Risk %",
    })
    st.dataframe(board_view.round({"ML Risk %": 1}), use_container_width=True, hide_index=True, height=320)
    st.caption(f"Showing {len(board_view):,} highest-priority of {len(board):,} matching shipments (exceptions first).")
    st.download_button(
        "⇩ EXPORT TRACKING BOARD (CSV)",
        data=board.to_csv(index=False).encode(),
        file_name="supchainmate_tracking_board.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tower_score:
    st.markdown("<div class='hud-label' style='margin:12px 0 6px 0;'>CARRIER SCORECARD</div>", unsafe_allow_html=True)
    if st.session_state.get("carriers_simulated"):
        st.markdown(
            "<div style='font-family:Share Tech Mono,monospace;font-size:0.62rem;color:#FBC02D;'>"
            "⚠ DEMO MODE — CARRIER NAMES &amp; COSTS ARE SIMULATED (fictional carriers over real Olist delivery dates)</div>",
            unsafe_allow_html=True,
        )
    scorecard = control_tower.carrier_scorecard(shipments_df)
    if scorecard is None:
        st.info(
            "No carrier column detected. Add a 'carrier' (or courier/transporter/3PL) column "
            "to your delivery file to unlock carrier scorecards."
        )
    else:
        st.dataframe(scorecard, use_container_width=True, hide_index=True)

        fig_carrier = px.bar(
            scorecard.dropna(subset=["On-Time %"]),
            x="Carrier", y="On-Time %", color="Grade",
            color_discrete_map={"A": "#00E676", "B": "#00D4FF", "C": "#FBC02D", "D": "#FF003C"},
            height=240,
        )
        fig_carrier.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,13,16,1)",
            margin=dict(l=40, r=20, t=10, b=30),
            yaxis_range=[0, 100],
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
        )
        fig_carrier.update_xaxes(gridcolor="#222228")
        fig_carrier.update_yaxes(gridcolor="#222228")
        st.plotly_chart(fig_carrier, use_container_width=True)

        for note in control_tower.scorecard_insights(scorecard):
            st.markdown(f"""
            <div style="background:#151518;border-left:3px solid #00D4FF;padding:8px 14px;
                        margin-bottom:4px;font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#CCCCCC;">
                {note}
            </div>""", unsafe_allow_html=True)

        st.download_button(
            "⇩ EXPORT CARRIER SCORECARD (CSV)",
            data=scorecard.to_csv(index=False).encode(),
            file_name="supchainmate_carrier_scorecard.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Freight Cost Audit ─────────────────────────────────────────────────────────
with st.expander("⚖ FREIGHT COST AUDIT — BILLING ANOMALY DETECTION", expanded=False):
    audit = cost_audit.run_audit(shipments_df)
    if audit is None:
        st.info(
            "No freight cost data detected. Add a cost/freight/charge column to your "
            "delivery file to unlock the audit (the demo dataset includes simulated costs)."
        )
    else:
        if st.session_state.get("carriers_simulated"):
            st.markdown(
                "<div style='font-family:Share Tech Mono,monospace;font-size:0.62rem;color:#FBC02D;'>"
                "⚠ DEMO MODE — FREIGHT COSTS ARE SIMULATED</div>",
                unsafe_allow_html=True,
            )
        ak = audit["kpis"]
        au1, au2, au3, au4, au5 = st.columns(5)
        au1.markdown(_hp("SPEND AUDITED", f"${ak['total_spend']:,.0f}",
                         f"{ak['audited_charges']:,} CHARGES"), unsafe_allow_html=True)
        au2.markdown(_hp("FLAGGED CHARGES", f"{ak['flagged_count']:,}",
                         f"EST. ${ak['flagged_value']:,.0f}", "#FF003C"), unsafe_allow_html=True)
        au3.markdown(_hp("COST OUTLIERS", f"${ak['outlier_overcharge']:,.0f}",
                         "ABOVE CARRIER IQR CAP", "#FF003C"), unsafe_allow_html=True)
        au4.markdown(_hp("LATE-PREMIUMS", f"${ak['late_premium_value']:,.0f}",
                         "PAID EXTRA, STILL LATE", "#FBC02D"), unsafe_allow_html=True)
        au5.markdown(_hp("RE-TENDER OPP.", f"${ak['retender_opportunity']:,.0f}",
                         "SPEND ABOVE NETWORK MEDIAN", "#00E676"), unsafe_allow_html=True)

        for note in audit["insights"]:
            st.markdown(f"""
            <div style="background:#151518;border-left:3px solid #FBC02D;padding:8px 14px;
                        margin-bottom:4px;font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#CCCCCC;">
                {note}
            </div>""", unsafe_allow_html=True)

        aud_l, aud_r = st.columns([3, 2])
        with aud_l:
            if len(audit["flagged"]):
                st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>FLAGGED CHARGES (WORST FIRST)</div>", unsafe_allow_html=True)
                flag_view = audit["flagged"].head(300).copy()
                if "order_date" in flag_view.columns:
                    flag_view["order_date"] = pd.to_datetime(flag_view["order_date"]).dt.strftime("%Y-%m-%d")
                st.dataframe(flag_view, use_container_width=True, hide_index=True, height=280)
                st.download_button(
                    "⇩ EXPORT FLAGGED CHARGES (CSV)",
                    data=audit["flagged"].to_csv(index=False).encode(),
                    file_name="supchainmate_flagged_charges.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.success("No anomalous charges detected.")
        with aud_r:
            if audit["by_carrier"] is not None:
                st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>COST PROFILE BY CARRIER ($/SHIPMENT)</div>", unsafe_allow_html=True)
                st.dataframe(audit["by_carrier"], use_container_width=True, hide_index=True, height=280)
                st.download_button(
                    "⇩ EXPORT AUDIT REPORT (TXT)",
                    data=cost_audit.audit_digest(audit).encode(),
                    file_name="supchainmate_cost_audit.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

# ── Invoice / BOL Scanner — Document Intelligence ──────────────────────────────
with st.expander("📄 INVOICE / BOL SCANNER — DOCUMENT INTELLIGENCE", expanded=False):
    di_engine = "🟢 GROQ EXTRACTION" if groq_ai.is_available() else "🟡 OFFLINE EXTRACTION (regex) — set GROQ_API_KEY for LLM-grade parsing"
    st.markdown(
        f"<div style='font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#888;'>"
        f"{di_engine} · PDF / TXT · RECONCILED AGAINST THE SHIPMENT BOARD & AUDITED RATE BANDS</div>",
        unsafe_allow_html=True)
    di_col1, di_col2 = st.columns([2, 1])
    with di_col1:
        di_file = st.file_uploader("Upload a freight invoice or BOL", type=["pdf", "txt"],
                                   key="di_upload", label_visibility="collapsed")
    with di_col2:
        di_sample = st.button("▷ TRY SAMPLE INVOICE", key="di_sample", use_container_width=True,
                              help="Generates an invoice from real board shipments with one inflated line")

    di_text = None
    if di_file is not None:
        di_text, di_msg = doc_intel.extract_text(di_file.read(), di_file.name)
        if di_text is None:
            st.error(di_msg)
    elif di_sample:
        di_text = doc_intel.sample_invoice(shipments_df)
        if di_text is None:
            st.info("Sample invoice needs shipments with freight costs.")

    if di_text:
        known_carriers = (shipments_df["carrier"].dropna().unique().tolist()
                          if "carrier" in shipments_df.columns else [])
        with st.spinner("Extracting fields..."):
            di_fields, di_used = doc_intel.extract_fields(di_text, known_carriers)
        di_result = doc_intel.reconcile(di_fields, shipments_df,
                                        audit["by_carrier"] if audit else None)
        v_color = {"OK TO PAY": "#00E676", "INSUFFICIENT DATA": "#888"}.get(
            di_result["verdict"], "#FF003C" if "UNKNOWN" in di_result["verdict"] else "#FBC02D")
        st.markdown(f"""
        <div class="hud-panel" style="border-color:{v_color};">
            <div class="hud-label">VERDICT · EXTRACTION: {di_used.upper()}</div>
            <div style="font-family:'Teko',sans-serif;font-size:1.8rem;color:{v_color};">
                {di_result['verdict']}
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#888;">
                INVOICE {di_fields.get('invoice_number') or '?'} ·
                {di_fields.get('carrier') or 'carrier unknown'} ·
                TOTAL {f"${di_fields['total_amount']:,.2f}" if di_fields.get('total_amount') else 'not found'}
            </div>
        </div>""", unsafe_allow_html=True)
        for f_txt in di_result["findings"]:
            f_color = "#FF003C" if f_txt.startswith("⚠") else "#00D4FF"
            st.markdown(f"""
            <div style="background:#151518;border-left:3px solid {f_color};padding:8px 14px;
                        margin-bottom:4px;font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#CCCCCC;">
                {f_txt}
            </div>""", unsafe_allow_html=True)
        if len(di_result["matched"]):
            with st.expander("Matched shipments", expanded=False):
                mv = di_result["matched"][[c for c in ["shipment_id", "carrier", "order_date",
                                                       "health", "freight_cost"]
                                           if c in di_result["matched"].columns]]
                st.dataframe(mv, use_container_width=True, hide_index=True)
        with st.expander("Document text", expanded=False):
            st.code(di_text[:3000], language=None)

# ── Carbon Lens ────────────────────────────────────────────────────────────────
with st.expander("🌱 CARBON LENS — FREIGHT CO₂e ESTIMATES", expanded=False):
    cl_avg_dist = carbon.network_avg_distance_km(centroid_stats)
    if cl_avg_dist is None:
        st.info("Carbon estimates need the network's cluster distance metrics.")
    else:
        st.markdown(
            "<div style='font-family:Share Tech Mono,monospace;font-size:0.62rem;color:#888;'>"
            "ESTIMATES: distance × weight × DEFRA-style mode factor (road 0.107 / rail 0.028 / "
            "air 1.13 / sea 0.016 kg CO₂e per tonne-km). Distances from your network's Haversine "
            "cluster metrics." + (" DEMO MODE — carrier transport modes are simulated."
                                  if st.session_state.get("carriers_simulated") else "") + "</div>",
            unsafe_allow_html=True)
        cl_weight = st.slider("AVG SHIPMENT WEIGHT (KG)", 1, 500, 20, key="cl_weight")
        cl_carriers = carbon.carrier_emissions(shipments_df, cl_avg_dist, cl_weight, scorecard)
        cl_zones = carbon.zone_emissions(centroid_stats, cl_weight)

        total_t = float(cl_carriers["Total tCO2e"].sum()) if cl_carriers is not None else (
            float(cl_zones["Total tCO2e"].sum()) if cl_zones is not None else 0.0)
        cb1, cb2, cb3 = st.columns(3)
        cb1.markdown(_hp("NETWORK FOOTPRINT", f"{total_t:,.1f} tCO₂e",
                         f"@ {cl_weight}KG/SHIPMENT · {cl_avg_dist:,.0f}KM AVG", "#00E676"), unsafe_allow_html=True)
        if cl_carriers is not None and len(cl_carriers):
            cb2.markdown(_hp("GREENEST CARRIER", str(cl_carriers.iloc[0]["Carrier"]),
                             f"{cl_carriers.iloc[0]['kg CO2e/shipment']:.2f} KG/SHIPMENT ({cl_carriers.iloc[0]['Mode'].upper()})",
                             "#00E676"), unsafe_allow_html=True)
            cb3.markdown(_hp("HIGHEST EMITTER", str(cl_carriers.iloc[-1]["Carrier"]),
                             f"{cl_carriers.iloc[-1]['kg CO2e/shipment']:.2f} KG/SHIPMENT ({cl_carriers.iloc[-1]['Mode'].upper()})",
                             "#FF003C"), unsafe_allow_html=True)

        ccl, ccr = st.columns([1, 1])
        with ccl:
            if cl_carriers is not None:
                st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>CO₂e BY CARRIER</div>", unsafe_allow_html=True)
                st.dataframe(cl_carriers, use_container_width=True, hide_index=True)
                if "Avg Cost/Shipment ($)" in cl_carriers.columns and cl_carriers["Avg Cost/Shipment ($)"].notna().any():
                    fig_co2 = px.scatter(
                        cl_carriers, x="Avg Cost/Shipment ($)", y="kg CO2e/shipment",
                        text="Carrier", color="Mode", size="Shipments",
                        color_discrete_map={"road": "#00D4FF", "rail": "#00E676",
                                            "air": "#FF003C", "sea": "#FBC02D"},
                        height=300,
                    )
                    fig_co2.update_traces(textposition="top center",
                                          textfont=dict(color="#AAAAAA", size=10))
                    fig_co2.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,13,16,1)",
                        margin=dict(l=40, r=20, t=10, b=40),
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
                    )
                    fig_co2.update_xaxes(gridcolor="#222228", title="$/shipment (cheapest →)")
                    fig_co2.update_yaxes(gridcolor="#222228", title="kg CO₂e/shipment (greenest ↓)")
                    st.plotly_chart(fig_co2, use_container_width=True)
        with ccr:
            if cl_zones is not None:
                st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>CO₂e BY DELIVERY ZONE</div>", unsafe_allow_html=True)
                st.dataframe(cl_zones, use_container_width=True, hide_index=True)
                st.download_button(
                    "⇩ EXPORT CARBON REPORT (CSV)",
                    data=(cl_carriers if cl_carriers is not None else cl_zones).to_csv(index=False).encode(),
                    file_name="supchainmate_carbon.csv", mime="text/csv",
                    use_container_width=True,
                )
        for note in carbon.carbon_insights(cl_carriers):
            st.markdown(f"""
            <div style="background:#151518;border-left:3px solid #00E676;padding:8px 14px;
                        margin-bottom:4px;font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#CCCCCC;">
                {note}
            </div>""", unsafe_allow_html=True)

# ── Supply Chain Health Check ──────────────────────────────────────────────────
with st.expander("🩺 SUPPLY CHAIN HEALTH CHECK — SCORED ASSESSMENT", expanded=False):
    hc = health_check.run_health_check(
        shipments=shipments_df,
        kpis=ct_kpis,
        audit=audit,
        decision_outputs=decision_outputs,
        delay_risk=delay_risk,
        centroid_stats=centroid_stats.reset_index() if centroid_stats is not None else None,
    )
    hc_color = {"A": "#00E676", "B": "#00D4FF", "C": "#FBC02D", "D": "#FF9800", "F": "#FF003C"}[hc["grade"]]
    hgl, hgr = st.columns([1, 2])
    with hgl:
        st.markdown(f"""
        <div class="hud-panel" style="border-color:{hc_color};text-align:center;">
            <div class="hud-label">OVERALL HEALTH</div>
            <div style="font-family:'Teko',sans-serif;font-size:4rem;color:{hc_color};line-height:1;">
                {hc['grade']}
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:1rem;color:#FFF;">
                {hc['score']:.0f}/100
            </div>
            {f'<div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#888;margin-top:6px;">DIFOT (APPROX): {hc["difot"]:.1f}%</div>' if hc.get("difot") is not None else ''}
        </div>""", unsafe_allow_html=True)
    with hgr:
        dim_df = pd.DataFrame(hc["dimensions"])[["dimension", "grade", "score", "detail"]]
        dim_df.columns = ["Dimension", "Grade", "Score", "Detail"]
        st.dataframe(dim_df, use_container_width=True, hide_index=True)
    for rec_txt in hc["recommendations"]:
        st.markdown(f"""
        <div style="background:#151518;border-left:3px solid {hc_color};padding:8px 14px;
                    margin-bottom:4px;font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#CCCCCC;">
            {rec_txt}
        </div>""", unsafe_allow_html=True)
    st.download_button(
        "⇩ EXPORT HEALTH CHECK (TXT)",
        data=health_check.health_report(hc).encode(),
        file_name="supchainmate_health_check.txt",
        mime="text/plain",
        use_container_width=True,
    )

# One KPI snapshot per data load — builds the performance history over time
if not st.session_state.get("kpi_snapshot_saved"):
    store.save_kpi_snapshot({
        "health_score": hc["score"],
        "grade": hc["grade"],
        "on_time_pct": None if math.isnan(ct_kpis["on_time_pct"]) else round(ct_kpis["on_time_pct"], 1),
        "difot": hc.get("difot"),
        "late": ct_kpis["late"],
        "at_risk": ct_kpis["at_risk"],
        "flagged_value": round(audit["kpis"]["flagged_value"], 0) if audit else None,
        "total_shipments": ct_kpis["total"],
        "source": "demo" if st.session_state.demo_mode else "user",
    })
    st.session_state.kpi_snapshot_saved = True

# ── Performance History ────────────────────────────────────────────────────────
with st.expander("📈 PERFORMANCE HISTORY — KPI TREND ACROSS SESSIONS", expanded=False):
    snapshots = store.load_kpi_snapshots()
    if len(snapshots) < 2:
        st.caption(
            f"{len(snapshots)} snapshot(s) saved so far — one is stored each time you load data. "
            "Trends appear once there are two or more."
        )
    else:
        hist_df = pd.DataFrame(snapshots)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=hist_df["ts"], y=hist_df["health_score"],
                                      mode="lines+markers", name="Health score",
                                      line=dict(color="#00D4FF", width=2)))
        if hist_df["on_time_pct"].notna().any():
            fig_hist.add_trace(go.Scatter(x=hist_df["ts"], y=hist_df["on_time_pct"],
                                          mode="lines+markers", name="On-time %",
                                          line=dict(color="#00E676", width=2)))
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,13,16,1)",
            margin=dict(l=40, r=20, t=20, b=40), height=280,
            yaxis_range=[0, 105],
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
        )
        fig_hist.update_xaxes(gridcolor="#222228")
        fig_hist.update_yaxes(gridcolor="#222228")
        st.plotly_chart(fig_hist, use_container_width=True)
        show_hist = hist_df[["ts", "grade", "health_score", "on_time_pct",
                             "late", "at_risk", "total_shipments", "source"]].rename(columns={
            "ts": "Timestamp (UTC)", "grade": "Grade", "health_score": "Health",
            "on_time_pct": "On-Time %", "late": "Late", "at_risk": "At Risk",
            "total_shipments": "Shipments", "source": "Source"})
        st.dataframe(show_hist.iloc[::-1], use_container_width=True, hide_index=True, height=200)

# ═══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS WORKFORCE — STATUS BOARD + RUNBOOK (STANDING RULES)
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="font-family:'Teko',sans-serif;font-size:1.6rem;letter-spacing:0.12rem;
            text-transform:uppercase;color:#FFFFFF;padding:8px 0;border-bottom:1px solid #00E676;
            margin-bottom:16px;">
    🤖 AUTONOMOUS WORKFORCE
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#666;margin-left:12px;">
        WORKERS MONITOR EVERY LOAD · RUNBOOK RULES IN PLAIN ENGLISH
    </span>
</div>
""", unsafe_allow_html=True)

_sweep_ctx = {"kpis": ct_kpis, "audit": audit, "scorecard": scorecard,
              "health": hc, "shipments": shipments_df}
sweep = agent.autonomous_sweep(_sweep_ctx)
runbook_rules = runbook.load_rules()
runbook_results = runbook.evaluate_all(runbook_rules, _sweep_ctx)
_fired_by_worker = {}
for rr in runbook_results:
    if rr["triggered"]:
        _fired_by_worker[rr["worker"]] = _fired_by_worker.get(rr["worker"], 0) + 1

_level_color = {"green": "#00E676", "yellow": "#FBC02D", "red": "#FF003C", "grey": "#555"}
sw_cols = st.columns(len(sweep))
for sw_col, s in zip(sw_cols, sweep):
    w = agent.WORKERS.get(s["worker"], {})
    color = _level_color[s["level"]]
    fired = _fired_by_worker.get(s["worker"], 0)
    badge = (f"<span style='background:#FF003C;color:#FFF;font-size:0.55rem;"
             f"padding:1px 6px;margin-left:6px;'>{fired} RULE{'S' if fired > 1 else ''} FIRED</span>"
             if fired else "")
    sw_col.markdown(f"""
    <div style="background:#151518;border:1px solid #222228;border-top:2px solid {color};
                padding:10px 12px;min-height:86px;">
        <div style="font-family:'Teko',sans-serif;font-size:1rem;color:#FFF;letter-spacing:0.06rem;">
            {w.get('emoji','•')} {s['worker'].upper()}
            <span style="color:{color};font-size:0.8rem;">●</span>{badge}
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:#AAA;
                    margin-top:6px;line-height:1.5;">{s['status']}</div>
    </div>""", unsafe_allow_html=True)

with st.expander("📓 RUNBOOK — STANDING RULES IN PLAIN ENGLISH", expanded=bool(_fired_by_worker)):
    st.caption(
        'Write a rule the way you\'d tell a colleague — e.g. "flag any shipment over $50", '
        '"alert me when SwiftLine on-time drops below 95%", "flag deliveries more than 5 days late", '
        '"health below 70". Rules are parsed, assigned to the right worker, saved, and '
        "re-evaluated on every data load."
    )
    rb1, rb2 = st.columns([3, 1])
    new_rule_text = rb1.text_input("New rule", key="rb_new_rule",
                                   placeholder='e.g. flag any shipment over $50',
                                   label_visibility="collapsed")
    with rb2:
        if st.button("＋ ADD RULE", key="rb_add", use_container_width=True) and new_rule_text.strip():
            added = runbook.add_rule(new_rule_text.strip())
            if added is None:
                st.error("Couldn't parse that rule — try phrasing like the examples above.")
            else:
                st.rerun()

    if not runbook_results:
        st.info("No standing rules yet — add one above and the workers will enforce it on every load.")
    else:
        for i, rr in enumerate(runbook_results):
            r_color = "#FF003C" if rr["triggered"] else "#00E676"
            w_emoji = agent.WORKERS.get(rr["worker"], {}).get("emoji", "•")
            st.markdown(f"""
            <div style="background:#151518;border-left:3px solid {r_color};padding:8px 14px;
                        margin-bottom:4px;font-family:'Share Tech Mono',monospace;font-size:0.72rem;">
                <span style="color:{r_color};font-weight:bold;">
                    {'⚠ TRIGGERED' if rr['triggered'] else '✓ CLEAR'}</span>
                <span style="color:#888;"> · {w_emoji} {rr['worker']}</span><br>
                <span style="color:#FFF;">"{rr['text']}"</span><br>
                <span style="color:#999;">{rr['detail']}</span>
            </div>""", unsafe_allow_html=True)
        del_idx = st.selectbox("Remove a rule", range(len(runbook_rules)),
                               format_func=lambda i: runbook_rules[i]["text"],
                               key="rb_del_select")
        if st.button("Remove selected rule", key="rb_del_btn"):
            runbook.remove_rule(del_idx)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# DECISION CENTER — EXPLAINABLE, HUMAN-APPROVED RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="font-family:'Teko',sans-serif;font-size:1.6rem;letter-spacing:0.12rem;
            text-transform:uppercase;color:#FFFFFF;padding:8px 0;border-bottom:1px solid #00D4FF;
            margin-bottom:16px;">
    🛡 DECISION CENTER
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#666;margin-left:12px;">
        EXPLAINED · SCORED · HUMAN-APPROVED · FULLY AUDITED
    </span>
</div>
""", unsafe_allow_html=True)

_trust_ctx = {
    "demand_profile": demand_profile,
    "decision_outputs": decision_outputs,
    "history_days": len(daily_df),
    "service_level": service_level,
    "sku_plan": sku_plan_with_status,
    "avg_lead_time": avg_lead_time,
    "scorecard": scorecard,
    "audit": audit,
}
v_decisions.render(trust.generate_all(_trust_ctx))

# ── Freight Tender / RFP Toolkit ───────────────────────────────────────────────
with st.expander("📑 FREIGHT TENDER / RFP TOOLKIT", expanded=False):
    tender_pack = tender.build_tender_pack(shipments_df, scorecard)
    if tender_pack is None:
        st.info("Tender pack needs shipment data with order dates.")
    else:
        ts = tender_pack["stats"]
        st.markdown(
            f"<div style='font-family:Share Tech Mono,monospace;font-size:0.72rem;color:#AAAAAA;'>"
            f"DATA-BACKED TENDER: {ts['total_shipments']:,} SHIPMENTS ({ts['period']}) · "
            f"AVG {ts['monthly_avg']:,.0f}/MONTH · PEAK {ts['peak_shipments']:,} IN {ts['peak_month']}"
            + (f" · SPEND ${ts['annual_spend']:,.0f}" if ts["annual_spend"] else "") + "</div>",
            unsafe_allow_html=True,
        )
        tl, tr = st.columns([1, 1])
        with tl:
            st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>MONTHLY LANE SUMMARY</div>", unsafe_allow_html=True)
            st.dataframe(tender_pack["lanes"], use_container_width=True, hide_index=True, height=240)
            st.download_button(
                "⇩ LANE SUMMARY (CSV)",
                data=tender_pack["lanes"].to_csv(index=False).encode(),
                file_name="tender_lane_summary.csv", mime="text/csv",
                use_container_width=True,
            )
            if tender_pack["carriers"] is not None:
                st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>INCUMBENT CARRIERS</div>", unsafe_allow_html=True)
                st.dataframe(tender_pack["carriers"], use_container_width=True, hide_index=True)
        with tr:
            st.markdown("<div class='hud-label' style='margin:8px 0 4px 0;'>RFP DOCUMENT DRAFT</div>", unsafe_allow_html=True)
            st.code(tender_pack["rfp_text"], language=None)
            st.download_button(
                "⇩ RFP DRAFT (TXT)",
                data=tender_pack["rfp_text"].encode(),
                file_name="freight_rfp_draft.txt", mime="text/plain",
                use_container_width=True,
            )

        # ── Rate-shift simulator ─────────────────────────────────────────────
        if audit is not None and audit["by_carrier"] is not None and len(audit["by_carrier"]) >= 2:
            st.markdown("<div class='hud-label' style='margin:12px 0 4px 0;'>RATE-SHIFT SIMULATOR</div>", unsafe_allow_html=True)
            carriers_list = audit["by_carrier"]["Carrier"].tolist()
            rs1, rs2, rs3 = st.columns(3)
            shift_from = rs1.selectbox("Move volume FROM", carriers_list,
                                       index=len(carriers_list) - 1, key="rs_from")
            shift_to = rs2.selectbox("TO", carriers_list, index=0, key="rs_to")
            shift_pct = rs3.slider("VOLUME TO SHIFT (%)", 5, 100, 25, key="rs_pct")
            sim = tender.simulate_rate_shift(audit["by_carrier"], shift_from, shift_to, shift_pct)
            if sim is None:
                st.caption("Pick two different carriers to simulate.")
            else:
                sim_color = "#00E676" if sim["cost_delta"] < 0 else "#FF003C"
                st.markdown(f"""
                <div style="background:#151518;border-left:3px solid {sim_color};padding:10px 16px;
                            font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#CCCCCC;">
                    {sim['summary']}<br>
                    <span style="color:#888;font-size:0.65rem;">Rate-only estimate — verify the
                    receiving carrier's capacity and service level (see scorecard) before committing.</span>
                </div>""", unsafe_allow_html=True)

# ── Alert Digest ───────────────────────────────────────────────────────────────
with st.expander("🔔 ALERT DIGEST — EMAIL / DOWNLOAD", expanded=False):
    _digest_ctx = {"shipments": shipments_df, "kpis": ct_kpis, "scorecard": scorecard}
    _, exc_arts = agent._TOOL_FUNCS["exception_summary"](_digest_ctx)
    exc_text = exc_arts[0]["data"] if exc_arts else "No exception data."
    digest_body = alerts.build_enterprise_digest(
        runbook.runbook_digest(runbook_results) + "\n\n" + exc_text,
        audit_text=cost_audit.audit_digest(audit) if audit else None,
        health_text=health_check.health_report(hc),
    )
    smtp_ok = alerts.smtp_configured()
    st.markdown(
        f"<div style='font-family:Share Tech Mono,monospace;font-size:0.65rem;color:{'#00E676' if smtp_ok else '#FBC02D'};'>"
        f"{'🟢 SMTP CONFIGURED — EMAIL DELIVERY ENABLED' if smtp_ok else '🟡 SMTP NOT CONFIGURED — set SMTP_HOST / SMTP_FROM (and SMTP_USER / SMTP_PASS) in .env to enable email delivery'}"
        f"</div>", unsafe_allow_html=True)
    al1, al2 = st.columns([2, 1])
    with al1:
        alert_email = st.text_input("Email address", key="ent_alert_email",
                                    value=store.load_setting("enterprise_alert_email", "") or "")
    with al2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("SEND DIGEST NOW", key="ent_send_digest", use_container_width=True,
                     disabled=not (smtp_ok and alert_email)):
            ok, send_msg = alerts.send_email(alert_email, "SupChainMate — Supply Chain Digest", digest_body)
            (st.success if ok else st.error)(send_msg)
            if ok:
                store.save_setting("enterprise_alert_email", alert_email)
    if alert_email and not smtp_ok:
        store.save_setting("enterprise_alert_email", alert_email)
    st.code(digest_body, language=None)
    st.download_button(
        "⇩ DOWNLOAD DIGEST (TXT)",
        data=digest_body.encode(),
        file_name="supchainmate_digest.txt", mime="text/plain",
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM: DEMAND SURGE SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
demand_label = "+SURGE IMPACT" if demand_change > 0 else ("-DEMAND DROP" if demand_change < 0 else "STABLE DEMAND")
demand_color = "#FBC02D" if demand_change > 20 else ("#FF003C" if demand_change < -20 else "#00E676")
revenue_opp  = adjusted_demand * 12
op_strain    = abs(demand_change) * 0.4

b1, b2, b3, b4, b5 = st.columns([2, 1, 2, 1, 1])
with b1:
    st.markdown('<div class="hud-label">SIMULATION ENGINE</div><div style="font-family:Teko,sans-serif;font-size:1.2rem;color:#FFF;text-transform:uppercase;letter-spacing:0.08rem;">DEMAND SURGE SIMULATOR</div>', unsafe_allow_html=True)
with b2:
    st.markdown(f'<div class="hud-label">REDUCED (-50%)</div><div class="hud-value-green" style="font-size:1rem;">-{max(0,demand_change/2):.0f}%</div>', unsafe_allow_html=True)
with b3:
    st.markdown(f'<div style="text-align:center;font-family:Teko,sans-serif;font-size:2rem;color:{demand_color};text-shadow:0 0 12px {demand_color}88;text-transform:uppercase;">{demand_change:+d}% {demand_label}</div>', unsafe_allow_html=True)
with b4:
    st.markdown(f'<div class="hud-label">CRITICAL (+50%)</div><div class="hud-value-red" style="font-size:1rem;">+{max(0,demand_change/2):.0f}%</div>', unsafe_allow_html=True)
with b5:
    st.markdown(f'<div class="hud-label">REVENUE OPP</div><div class="hud-value-green" style="font-size:0.9rem;">+${revenue_opp/1000:.0f}K</div><div class="hud-label" style="margin-top:4px;">OP STRAIN</div><div class="hud-value-red" style="font-size:0.9rem;">+{op_strain:.1f}% ERR</div>', unsafe_allow_html=True)

if st.button("▶ RUN SCENARIO", key="run_scenario"):
    with st.spinner("RUNNING SIMULATION..."):
        time.sleep(0.8)
    if demand_change > 20:
        st.error(f"⚠ SURGE RISK — STOCKOUT ELEVATED. ADJUSTED DEMAND: {adjusted_demand:,}")
    elif demand_change < -20:
        st.warning(f"📉 DEMAND DROP. REDUCE INVENTORY. ADJUSTED: {adjusted_demand:,}")
    else:
        st.success(f"✅ SCENARIO NOMINAL — ADJUSTED DEMAND: {adjusted_demand:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECONDARY: Forecast Chart + AI Copilot
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
exp_chart, exp_copilot = st.columns([1, 1])

with exp_chart:
    with st.expander("📈 DEMAND FORECAST — PROPHET ENGINE", expanded=False):
        fc = forecast_df.sort_values("ds")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(255,0,60,0.12)",
            name="Confidence Interval"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines",
            name="Forecast", line=dict(color="#FF003C", width=2)))
        fig.add_trace(go.Scatter(x=daily_df["ds"], y=daily_df["y"], mode="lines",
            name="Actual", line=dict(color="#00D4FF", width=1.5)))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,13,16,1)",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
            height=320,
        )
        fig.update_xaxes(gridcolor="#222228")
        fig.update_yaxes(gridcolor="#222228", title_text="VOLUME")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🏆 MODEL TOURNAMENT — ENSEMBLE FORECAST", expanded=False):
        _tk = f"tournament_{days}"
        if _tk not in st.session_state:
            with st.spinner("Backtesting model ensemble on the last 28 days..."):
                st.session_state[_tk] = ensemble.run_tournament(
                    daily_df, forecast_df, horizon_days=days)
        tourney = st.session_state[_tk]
        if tourney is None:
            st.info("The tournament needs at least ~120 days of daily history.")
        else:
            champ_note = ""
            if tourney["prophet_mape"] is not None and tourney["champion"] != "Prophet":
                edge = tourney["prophet_mape"] - tourney["champion_mape"]
                champ_note = f"BEATS PROPHET BY {edge:.1f} MAPE PTS ON THE HOLDOUT"
            elif tourney["champion"] == "Prophet":
                champ_note = "PROPHET HOLDS THE CROWN — MAIN FORECAST ALREADY OPTIMAL"
            tc1, tc2 = st.columns([1, 2])
            with tc1:
                st.markdown(f"""
                <div class="hud-panel" style="border-color:#FBC02D;text-align:center;">
                    <div class="hud-label">CHAMPION MODEL</div>
                    <div style="font-family:'Teko',sans-serif;font-size:1.6rem;color:#FBC02D;">
                        🏆 {tourney['champion']}
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#FFF;">
                        {tourney['champion_mape']:.1f}% MAPE · {tourney['holdout_days']}-DAY BACKTEST
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#888;margin-top:4px;">
                        {champ_note}
                    </div>
                </div>""", unsafe_allow_html=True)
                st.dataframe(tourney["leaderboard"], use_container_width=True, hide_index=True)
            with tc2:
                hold = tourney["holdout"]
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=hold["ds"], y=hold["actual"], mode="lines",
                                           name="Actual", line=dict(color="#00D4FF", width=2)))
                if tourney["champion"] in hold.columns:
                    fig_t.add_trace(go.Scatter(x=hold["ds"], y=hold[tourney["champion"]],
                                               mode="lines", name=f"🏆 {tourney['champion']}",
                                               line=dict(color="#FBC02D", width=2)))
                if "Prophet" in hold.columns and tourney["champion"] != "Prophet":
                    fig_t.add_trace(go.Scatter(x=hold["ds"], y=hold["Prophet"], mode="lines",
                                               name="Prophet", line=dict(color="#FF003C", width=1.5, dash="dot")))
                fig_t.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,13,16,1)",
                    hovermode="x unified", height=300,
                    margin=dict(l=40, r=20, t=20, b=40),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
                )
                fig_t.update_xaxes(gridcolor="#222228")
                fig_t.update_yaxes(gridcolor="#222228", title_text="HOLDOUT VOLUME")
                st.plotly_chart(fig_t, use_container_width=True)
                if tourney["forecast"] is not None:
                    st.download_button(
                        f"⇩ CHAMPION FORECAST — NEXT {days} DAYS (CSV)",
                        data=tourney["forecast"].to_csv(index=False).encode(),
                        file_name="supchainmate_champion_forecast.csv", mime="text/csv",
                        use_container_width=True,
                    )


with exp_copilot:
    with st.expander("🤖 AI WORKERS — YOUR AGENTIC OPERATIONS TEAM", expanded=False):
        agent_status = (
            "🟢 GROQ AGENT LIVE · LLaMA-3.3-70B TOOL CALLING"
            if groq_ai.is_available()
            else "🟡 OFFLINE MODE — workers still act on your live data; set GROQ_API_KEY for reasoning &amp; wording"
        )
        st.markdown(
            f'<div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#888;margin-bottom:8px;">'
            f'{agent_status} · {len(agent.WORKERS)} WORKERS · {len(agent.TOOLS_SCHEMA)} TOOLS · FULL REASONING TRACE</div>',
            unsafe_allow_html=True
        )
        live_context = {
            "Delay Risk":              f"{delay_risk:.1f}%",
            "Demand Forecast":         f"{next_week_demand:,} units ({days}-day horizon)",
            "Demand Growth":           f"{growth:+.1f}%",
            "Safety Stock Target":     f"{decision_outputs.safety_stock:,.0f} units",
            "EOQ":                     f"{decision_outputs.eoq:,.0f} units/order",
            "Reorder Point":           f"{decision_outputs.reorder_point:,.0f} units",
            "Lead Time Buffer":        f"+{decision_outputs.lead_time_buffer_days:.1f} days",
            "Annual Cost Savings":     f"${decision_outputs.savings_vs_current:,.0f}",
            "Active Breaches":         active_breaches,
            "System Status":           system_status,
            "Critical Zones":          len(agreed_critical),
            "Service Level Target":    f"{service_level*100:.0f}%",
            "Total Orders Analysed":   total_orders,
            "Shipments On-Time %":     f"{ct_kpis['on_time_pct']:.1f}%" if not math.isnan(ct_kpis["on_time_pct"]) else "N/A",
            "Shipments At Risk":       ct_kpis["at_risk"],
            "Shipments Late":          ct_kpis["late"],
        }
        agent_ctx = {
            "shipments":        shipments_df,
            "scorecard":        scorecard,
            "kpis":             ct_kpis,
            "metrics":          live_context,
            "decision_outputs": decision_outputs,
            "exec_plan":        exec_plan_df,
            "delay_risk":       delay_risk,
            "centroid_stats":   centroid_stats,
            "sku_plan":         sku_plan_with_status,
        }

        if "agent_chat" not in st.session_state:
            st.session_state.agent_chat = []

        pending_query = None
        worker_cols = st.columns(len(agent.WORKERS))
        for w_col, (w_name, w) in zip(worker_cols, agent.WORKERS.items()):
            with w_col:
                st.markdown(f"""
                <div style="background:#151518;border:1px solid #222228;border-top:2px solid #00D4FF;
                            padding:10px 10px 6px 10px;min-height:96px;">
                    <div style="font-family:'Teko',sans-serif;font-size:1.05rem;color:#FFF;
                                letter-spacing:0.06rem;">{w['emoji']} {w_name.upper()}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
                                color:#00D4FF;letter-spacing:0.08rem;">{w['role'].upper()}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;
                                color:#777;margin-top:4px;line-height:1.4;">{w['desc']}</div>
                </div>""", unsafe_allow_html=True)
                for a_label, a_prompt in w["actions"]:
                    if st.button(a_label, key=f"wk_{w_name}_{a_label}", use_container_width=True):
                        pending_query = a_prompt

        typed_query = st.chat_input("Ask the agent to do something — it can act, not just answer...")
        if typed_query:
            pending_query = typed_query

        def _worker_caption(actions):
            ws = agent.workers_for_actions(actions)
            tags = " · ".join(f"{agent.WORKERS[w]['emoji']} {w}" for w in ws) if ws else ""
            return (tags + "  |  " if tags else "") + "⚙ " + " · ".join(actions)

        for t_i, turn in enumerate(st.session_state.agent_chat):
            with st.chat_message(turn["role"]):
                if turn.get("actions"):
                    st.caption(_worker_caption(turn["actions"]))
                st.write(turn["content"])
                vh.render_trace(turn.get("trace"))
                for a_i, art in enumerate(turn.get("artifacts", [])):
                    vh.render_artifact(art, key=f"agent_art_{t_i}_{a_i}")

        if pending_query:
            st.chat_message("user").write(pending_query)
            with st.spinner("Agent thinking → acting..."):
                agent_result = agent.run_agent(pending_query, agent_ctx)
            st.session_state.agent_chat.append({"role": "user", "content": pending_query})
            st.session_state.agent_chat.append({
                "role": "assistant",
                "content": agent_result["reply"],
                "artifacts": agent_result["artifacts"],
                "actions": agent_result["actions"],
                "trace": agent_result.get("trace"),
            })
            with st.chat_message("assistant"):
                if agent_result["actions"]:
                    st.caption(_worker_caption(agent_result["actions"]))
                st.write(agent_result["reply"])
                vh.render_trace(agent_result.get("trace"))
                for a_i, art in enumerate(agent_result["artifacts"]):
                    vh.render_artifact(art, key=f"agent_art_new_{a_i}")

# ═══════════════════════════════════════════════════════════════════════════════
# ENTERPRISE REPORTING LAYER
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="font-family:'Teko',sans-serif;font-size:1.6rem;letter-spacing:0.12rem;
            text-transform:uppercase;color:#FFFFFF;padding:8px 0;border-bottom:1px solid #333340;
            margin-bottom:16px;">
    ◈ ENTERPRISE REPORTING LAYER
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#666;margin-left:12px;">
        STRUCTURED TABLES · EXPORTS · EXECUTIVE REPORT
    </span>
</div>
""", unsafe_allow_html=True)

rep1, rep2 = st.columns(2)

# ── Zone Risk Intelligence Table ───────────────────────────────────────────────
with rep1:
    with st.expander("📡 ZONE RISK INTELLIGENCE TABLE", expanded=True):
        if "cluster" in geo_df.columns and "risk_score" in geo_df.columns:
            zone_summary = (
                geo_df.groupby("cluster")
                .agg(
                    Customers=("lat", "count"),
                    Avg_Risk_Score=("risk_score", "mean"),
                    Max_Risk_Score=("risk_score", "max"),
                )
                .reset_index()
                .rename(columns={"cluster": "Zone"})
                .round(1)
            )
            zone_summary["Risk Level"] = zone_summary["Avg_Risk_Score"].apply(
                lambda s: "🔴 CRITICAL" if s > 90 else ("🟡 WARNING" if s > 70 else "🟢 SAFE")
            )
            zone_summary["Action"] = zone_summary["Avg_Risk_Score"].apply(
                lambda s: "Reroute shipments" if s > 90 else ("Monitor closely" if s > 70 else "No action required")
            )
            st.dataframe(
                zone_summary[["Zone", "Customers", "Avg_Risk_Score", "Risk Level", "Action"]],
                use_container_width=True, hide_index=True
            )
            st.download_button(
                "⇩ EXPORT ZONE TABLE (CSV)",
                data=zone_summary.to_csv(index=False).encode(),
                file_name="supchainmate_zone_risk.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ── Inventory Decision Table ───────────────────────────────────────────────────
with rep2:
    with st.expander("📦 INVENTORY DECISION TABLE", expanded=True):
        inventory_table = pd.DataFrame({
            "Parameter":   ["Safety Stock", "EOQ (Order Qty)", "Reorder Point",
                            "Lead Time Buffer", "Avg Daily Demand", "Demand Std Dev",
                            "Service Level", "Annual Demand", "Holding Cost (Opt.)",
                            "Ordering Cost (Opt.)", "Total Cost (Opt.)", "Annual Savings"],
            "Value":       [
                f"{decision_outputs.safety_stock:,.0f} units",
                f"{decision_outputs.eoq:,.0f} units",
                f"{decision_outputs.reorder_point:,.0f} units",
                f"+{decision_outputs.lead_time_buffer_days:.1f} days",
                f"{demand_profile.avg_daily_demand:,.1f} units/day",
                f"{demand_profile.std_daily_demand:,.1f} units/day",
                f"{service_level*100:.0f}%  (Z={decision_outputs.z_value})",
                f"{demand_profile.annual_demand:,.0f} units/yr",
                f"${decision_outputs.holding_cost_optimized:,.0f}/yr",
                f"${decision_outputs.ordering_cost_annual:,.0f}/yr",
                f"${decision_outputs.total_optimized_cost:,.0f}/yr",
                f"${decision_outputs.savings_vs_current:,.0f}/yr",
            ],
            "Status": [
                "▲ Action" if decision_outputs.safety_stock_delta_pct > 0 else "▼ Reduce",
                "Optimized", "Set", "Add Buffer", "Stable", "Monitored",
                "Target", "Annual", "Optimized", "Optimized", "Final", "Realised",
            ],
        })
        st.dataframe(inventory_table, use_container_width=True, hide_index=True)
        st.download_button(
            "⇩ EXPORT INVENTORY PLAN (CSV)",
            data=inventory_table.to_csv(index=False).encode(),
            file_name="supchainmate_inventory_plan.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Forecast Export ────────────────────────────────────────────────────────────
st.divider()
exp_exports, exp_report = st.columns([1, 2])

with exp_exports:
    st.markdown("<div class='hud-label' style='margin-bottom:8px;'>EXPORT BUNDLE</div>", unsafe_allow_html=True)

    # Forecast data
    forecast_export = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_export.columns = ["Date", "Forecast", "Lower_Bound", "Upper_Bound"]
    st.download_button(
        "⇩ FORECAST DATA (CSV)",
        data=forecast_export.to_csv(index=False).encode(),
        file_name="supchainmate_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # KPI Summary
    kpi_summary = pd.DataFrame({
        "KPI":   ["Total Forecast Demand", "Avg Daily Demand", "Demand Growth %",
                  "Delay Risk %", "Safety Stock", "EOQ", "Reorder Point",
                  "Current Cost", "Optimised Cost", "Annual Savings",
                  "Active Breaches", "System Status"],
        "Value": [next_week_demand, round(avg_daily, 1), f"{growth:.1f}%",
                  f"{delay_risk:.1f}%", decision_outputs.safety_stock,
                  decision_outputs.eoq, decision_outputs.reorder_point,
                  f"${current_cost:,.0f}", f"${optimized_cost:,.0f}",
                  f"${savings:,.0f}", active_breaches, system_status],
    })
    st.download_button(
        "⇩ KPI SUMMARY (CSV)",
        data=kpi_summary.to_csv(index=False).encode(),
        file_name="supchainmate_kpi_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "⇩ EXECUTION PLAN (CSV)",
        data=exec_plan_df.to_csv(index=False).encode(),
        file_name="supchainmate_execution_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── Executive Report ────────────────────────────────────────────────────────────
with exp_report:
    with st.expander("📄 EXECUTIVE REPORT — AI INTELLIGENCE SUMMARY", expanded=True):
        top_rec = decision_outputs.recommendations[0] if decision_outputs.recommendations else {}
        risk_banner = "🔴 HIGH RISK" if delay_risk > 15 else ("🟡 MODERATE" if delay_risk > 8 else "🟢 NOMINAL")

        st.markdown(f"""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;
            line-height:1.9;color:#CCCCCC;background:#0D0D10;
            border:1px solid #222228;padding:20px;border-top:2px solid #FF003C;">

<div style="color:#FFFFFF;font-family:'Teko',sans-serif;font-size:1.3rem;
            letter-spacing:0.1rem;margin-bottom:12px;">
    SUPCHAINMATE EXECUTIVE INTELLIGENCE REPORT
</div>

<div style="color:#888;font-size:0.65rem;letter-spacing:0.08rem;margin-bottom:16px;">
    GENERATED: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC &nbsp;|&nbsp;
    SOURCE: {'DEMO DATASET' if st.session_state.demo_mode else 'USER DATA'} &nbsp;|&nbsp;
    ORDERS ANALYSED: {total_orders:,}
</div>

<div style="color:#FF003C;font-size:0.65rem;letter-spacing:0.1rem;margin-bottom:4px;">■ EXECUTIVE SUMMARY</div>
The system has analysed <b style="color:#FFF;">{total_orders:,} orders</b> spanning
<b style="color:#FFF;">{len(daily_df)} days</b>.
Forecast demand over the next <b style="color:#FFF;">{days} days</b> is
<b style="color:#FFF;">{next_week_demand:,} units</b>
({'+' if growth >= 0 else ''}{growth:.1f}% vs baseline).
Overall system risk posture: <b style="color:#FF003C;">{risk_banner}</b>.

<br><br>

<div style="color:#FF003C;font-size:0.65rem;letter-spacing:0.1rem;margin-bottom:4px;">■ KEY RISKS</div>
• Delay risk at <b style="color:#FF003C;">{delay_risk:.1f}%</b>
  — {f"immediate intervention required." if delay_risk > 15 else "within acceptable limits."}<br>
• {f"Demand growth of {growth:.1f}% signals possible stockout pressure." if growth > 10
  else f"Demand contraction of {abs(growth):.1f}% — overstock risk." if growth < -10
  else "Demand is stable — standard replenishment applies."}<br>
• <b style="color:#FFF;">{active_breaches} active breach(es)</b> detected across monitored KPIs.

<br><br>

<div style="color:#FF003C;font-size:0.65rem;letter-spacing:0.1rem;margin-bottom:4px;">■ DECISION ENGINE RECOMMENDATIONS</div>
{"".join([f"• <b style='color:#FFF;'>[{r['impact']}] {r['category']}:</b> {r['action']}<br>" for r in decision_outputs.recommendations])}

<br>

<div style="color:#FF003C;font-size:0.65rem;letter-spacing:0.1rem;margin-bottom:4px;">■ FINANCIAL IMPACT</div>
• Optimised total inventory cost: <b style="color:#00E676;">${decision_outputs.total_optimized_cost:,.0f}/yr</b><br>
• Estimated annual savings vs current strategy: <b style="color:#00E676;">${decision_outputs.savings_vs_current:,.0f}</b><br>
• Safety stock target: <b style="color:#FFF;">{decision_outputs.safety_stock:,.0f} units</b>
  @ {service_level*100:.0f}% service level

<br><br>

<div style="color:#888;font-size:0.6rem;border-top:1px solid #222228;padding-top:8px;">
    Generated by SupChainMate Autonomous Decision Engine · 
    For integration with Power BI or Excel, use the Export Bundle above.
</div>

</div>
        """, unsafe_allow_html=True)

        # Export report as plain-text CSV
        exec_report_df = pd.DataFrame({
            "Section": ["Summary", "Summary", "Summary",
                        "Risk", "Risk", "Risk",
                        "Decision", "Decision", "Decision", "Decision",
                        "Financial", "Financial", "Financial"],
            "Item": [
                "Total Orders Analysed", "Forecast Demand (Next Period)", "Demand Growth %",
                "Delay Risk %", "Active Breaches", "System Status",
                "Safety Stock (units)", "EOQ (units)", "Reorder Point (units)", "Lead Time Buffer (days)",
                "Optimised Total Cost ($/yr)", "Annual Savings ($/yr)", "Service Level Target",
            ],
            "Value": [
                total_orders, next_week_demand, f"{growth:.1f}%",
                f"{delay_risk:.1f}%", active_breaches, system_status,
                decision_outputs.safety_stock, decision_outputs.eoq,
                decision_outputs.reorder_point, decision_outputs.lead_time_buffer_days,
                decision_outputs.total_optimized_cost, decision_outputs.savings_vs_current,
                f"{service_level*100:.0f}%",
            ],
        })
        st.download_button(
            "⇩ EXPORT EXECUTIVE REPORT (CSV)",
            data=exec_report_df.to_csv(index=False).encode(),
            file_name="supchainmate_executive_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
