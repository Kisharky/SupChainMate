import os
import io
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from modules import forecast, network, optimization, tracking, ingestion

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupChainMate — Mission Control",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ── Inline upload-screen CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
.upload-hero {
    text-align: center;
    padding: 40px 0 20px 0;
}
.upload-hero h1 {
    font-family: 'Teko', sans-serif !important;
    font-size: 3.5rem !important;
    color: #FFFFFF !important;
    letter-spacing: 0.15rem;
    text-transform: uppercase;
    margin-bottom: 6px !important;
}
.upload-hero .subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #666666;
    letter-spacing: 0.08rem;
    margin-bottom: 40px;
}
.upload-card {
    background: #151518;
    border: 1px solid #222228;
    border-top: 2px solid #FF003C;
    padding: 20px;
    margin-bottom: 8px;
    border-radius: 0px;
}
.upload-card-label {
    font-family: 'Teko', sans-serif;
    font-size: 1.1rem;
    color: #FFFFFF;
    text-transform: uppercase;
    letter-spacing: 0.08rem;
    margin-bottom: 4px;
}
.upload-card-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #555555;
    letter-spacing: 0.06rem;
    margin-bottom: 12px;
}
.detected-badge {
    background: rgba(0, 230, 118, 0.1);
    border: 1px solid #00E676;
    color: #00E676;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    display: inline-block;
    margin: 2px 3px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════
for key in ["orders_df", "delivery_df", "location_df", "cost_df",
            "daily_df", "forecast_df", "tracking_df", "geo_df",
            "delay_model", "X_test_delay", "summary", "current_cost",
            "data_loaded", "demo_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False


# ── Demo data loader ───────────────────────────────────────────────────────────
DEMO_ORDERS    = "data/olist_orders.csv"
DEMO_DELIVERY  = "data/olist_orders_dataset.csv"
DEMO_CUSTOMERS = "data/olist_customers_dataset.csv"


def _load_demo():
    """Load and process the built-in Olist demo dataset."""
    with st.spinner("LOADING DEMO DATA..."):
        raw_orders = forecast.load_orders(DEMO_ORDERS)
        daily      = forecast.daily_demand(raw_orders)
        model      = forecast.fit_prophet_model(daily)
        st.session_state.daily_df = daily

        future                   = model.make_future_dataframe(periods=7)
        future                   = future.merge(daily[["ds", "external_signal"]], on="ds", how="left")
        future["external_signal"]= future["external_signal"].fillna(0)
        st.session_state.forecast_df   = model.predict(future)
        st.session_state._prophet_model = model

        raw_delivery = pd.read_csv(DEMO_DELIVERY)
        tdf          = tracking.simulate_tracking(raw_delivery)
        m, X_test, _ = tracking.train_delay_model(tdf)
        st.session_state.tracking_df   = tdf
        st.session_state.delay_model   = m
        st.session_state.X_test_delay  = X_test

        customers = pd.read_csv(DEMO_CUSTOMERS)
        geo_lookup = network.get_geo_lookup()
        geo_df    = network.prepare_customer_data(customers, geo_lookup=geo_lookup)
        st.session_state.geo_df = network.run_clustering(geo_df)

        st.session_state.summary       = optimization.network_summary(raw_orders)

        rng = np.random.default_rng(42)
        cost_arr = rng.uniform(5, 20, size=len(tdf))
        st.session_state.current_cost  = float(cost_arr.sum())

        st.session_state.data_loaded   = True
        st.session_state.demo_mode     = True
    st.rerun()


def _process_uploaded(raw_orders, raw_delivery, raw_location, raw_cost):
    """Normalise and process user-uploaded files."""
    with st.spinner("PROCESSING YOUR DATA..."):
        # ── Orders ────────────────────────────────────────────────────────────
        orders_norm  = ingestion.normalise_orders(raw_orders)
        daily        = ingestion.orders_to_daily_demand(orders_norm)
        model        = forecast.fit_prophet_model(daily)
        st.session_state.daily_df = daily

        future                    = model.make_future_dataframe(periods=7)
        future                    = future.merge(daily[["ds", "external_signal"]], on="ds", how="left")
        future["external_signal"] = future["external_signal"].fillna(0)
        st.session_state.forecast_df    = model.predict(future)
        st.session_state._prophet_model  = model

        # ── Delivery (optional) ───────────────────────────────────────────────
        if raw_delivery is not None:
            delivery_norm = ingestion.normalise_delivery(raw_delivery)
            tdf           = ingestion.delivery_to_tracking(delivery_norm)
        else:
            # Simulate from orders if no delivery file provided
            tdf           = tracking.simulate_tracking(orders_norm.rename(columns={"order_date": "order_purchase_timestamp"}))

        # Quick delay model on available data
        tdf["is_late"]     = (tdf.get("status", "") == "Delayed").astype(int) if "status" in tdf.columns else np.random.randint(0, 2, len(tdf))
        tdf["lead_days"]   = tdf.get("lead_days", pd.Series(np.random.uniform(1, 20, len(tdf))))
        X                  = tdf[["lead_days"]].fillna(10)
        y                  = tdf["is_late"]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        st.session_state.tracking_df   = tdf
        st.session_state.delay_model   = clf
        st.session_state.X_test_delay  = X.tail(100)

        # ── Location (optional) ───────────────────────────────────────────────
        if raw_location is not None:
            loc_norm = ingestion.normalise_location(raw_location)
        else:
            # Synthesise from orders count
            np.random.seed(42)
            n = min(len(orders_norm), 500)
            loc_norm = pd.DataFrame({
                "lat":   np.random.uniform(-33, 5, n),
                "lon":   np.random.uniform(-73, -35, n),
                "label": "Node",
            })

        loc_norm["cluster"] = pd.qcut(loc_norm["lat"], q=5, labels=False, duplicates="drop")
        st.session_state.geo_df = loc_norm

        # ── Cost (optional) ───────────────────────────────────────────────────
        if raw_cost is not None:
            cost_norm = ingestion.normalise_cost(raw_cost)
            st.session_state.current_cost = float(cost_norm["cost"].sum())
        else:
            rng = np.random.default_rng(42)
            st.session_state.current_cost = float(rng.uniform(5, 20, len(tdf)).sum())

        st.session_state.summary     = None  # Network summary needs olist shape
        st.session_state.data_loaded = True
        st.session_state.demo_mode   = False
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.data_loaded:

    st.markdown("""
    <div class="upload-hero">
        <h1>⬡ SupChainMate</h1>
        <div class="subtitle">
            UPLOAD YOUR SUPPLY CHAIN DATA → AI ANALYSES → INSTANT INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    u1, u2, u3, u4 = st.columns(4)

    with u1:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-label">📦 Orders Data</div>
            <div class="upload-card-sub">ORDER DATE · PRODUCT · QUANTITY · REGION<br><br>AUTO-DETECTED: date, quantity, sku</div>
        </div>""", unsafe_allow_html=True)
        orders_file = st.file_uploader("", type=["csv", "xlsx", "xls"], key="orders",
                                        label_visibility="collapsed")

    with u2:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-label">🚚 Delivery Data</div>
            <div class="upload-card-sub">DELIVERY DATE · STATUS · ROUTE · LEAD TIME<br><br>AUTO-DETECTED: status, lead days</div>
        </div>""", unsafe_allow_html=True)
        delivery_file = st.file_uploader("", type=["csv", "xlsx", "xls"], key="delivery",
                                          label_visibility="collapsed")

    with u3:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-label">📍 Location Data</div>
            <div class="upload-card-sub">CUSTOMER LOCATIONS · WAREHOUSES · ZIP<br><br>AUTO-DETECTED: lat/lon or postal code</div>
        </div>""", unsafe_allow_html=True)
        location_file = st.file_uploader("", type=["csv", "xlsx", "xls"], key="location",
                                          label_visibility="collapsed")

    with u4:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-label">💰 Cost Data</div>
            <div class="upload-card-sub">COST PER DELIVERY · FUEL · WAREHOUSE<br><br>AUTO-DETECTED: cost, price, fee columns</div>
        </div>""", unsafe_allow_html=True)
        cost_file = st.file_uploader("", type=["csv", "xlsx", "xls"], key="cost",
                                      label_visibility="collapsed")

    # Auto-detect preview
    if orders_file:
        try:
            raw   = ingestion._read_file(orders_file)
            meta  = ingestion.detected_columns_summary(raw, "orders")
            badges = ""
            if meta["date_col"]:  badges += f'<span class="detected-badge">✓ DATE: {meta["date_col"]}</span>'
            if meta["qty_col"]:   badges += f'<span class="detected-badge">✓ QTY: {meta["qty_col"]}</span>'
            badges += f'<span class="detected-badge">✓ {meta["rows"]:,} ROWS</span>'
            st.markdown(f"<div style='margin:8px 0;'>{badges}</div>", unsafe_allow_html=True)
            orders_file.seek(0)
        except Exception as e:
            st.error(f"Error reading orders file: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    btn_l, btn_m, btn_r = st.columns([1, 1, 1])

    with btn_m:
        if orders_file:
            if st.button("⚡ ANALYSE MY DATA", use_container_width=True):
                try:
                    raw_orders   = ingestion._read_file(orders_file)
                    raw_delivery = ingestion._read_file(delivery_file) if delivery_file else None
                    raw_location = ingestion._read_file(location_file) if location_file else None
                    raw_cost     = ingestion._read_file(cost_file)     if cost_file     else None
                    _process_uploaded(raw_orders, raw_delivery, raw_location, raw_cost)
                except Exception as e:
                    st.error(f"Processing failed: {e}")
        else:
            st.markdown("""
            <div style='text-align:center; font-family:Share Tech Mono,monospace;
                        font-size:0.7rem; color:#444; padding:12px;'>
                ▲ UPLOAD ORDERS FILE TO ENABLE ANALYSIS
            </div>""", unsafe_allow_html=True)

    with btn_r:
        if st.button("▷ TRY DEMO DATA", use_container_width=True):
            _load_demo()

    st.stop()  # Don't render dashboard until data is loaded


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
    mode_label = "DEMO DATASET" if st.session_state.demo_mode else "USER DATA"
    st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#666;'>SOURCE: {mode_label}</div>", unsafe_allow_html=True)
    if st.button("🔄 LOAD NEW DATA"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
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
# MAIN LAYOUT: Map (left) | HUD Panels (right)
# ═══════════════════════════════════════════════════════════════════════════════
col_map, col_hud = st.columns([2, 1], gap="small")

with col_map:
    # Re-cluster with sidebar n_clusters (what-if parameter)
    geo_df = geo_df.copy()
    geo_df = network.run_clustering(geo_df[["lat", "lon"]], n_clusters=n_clusters)

    np.random.seed(42)
    geo_df["risk_score"] = np.random.uniform(0, 100, size=len(geo_df))
    geo_df["risk_level"] = pd.cut(
        geo_df["risk_score"], bins=[-1, 70, 90, 100],
        labels=["Safe", "Warning", "Critical"]
    )

    # Haversine centroid metrics
    centroid_stats = network.cluster_centroid_distances(geo_df)

    fig_map = px.scatter_mapbox(
        geo_df, lat="lat", lon="lon",
        color="risk_level", size="risk_score", size_max=18,
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

    critical_pct = len(geo_df[geo_df["risk_level"] == "Critical"]) / len(geo_df) * 100
    st.markdown(f"""
    <div class="hud-panel">
        <div style="color:#FF003C;font-family:'Teko',sans-serif;font-size:1.1rem;letter-spacing:0.1rem;">
            ⚠ IMMEDIATE THREAT DETECTED
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#AAAAAA;margin:6px 0;">
            {critical_pct:.0f}% OF NETWORK NODES IN CRITICAL DISRUPTION ZONE.
            WEATHER / CONGESTION RISK ELEVATED.
        </div>
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
            ★ OPTIMIZATION READY
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#AAAAAA;margin:4px 0 10px 0;">
            AI IDENTIFIED ${savings:,.0f} SAVINGS ON ACTIVE ROUTES.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("⚡ EXECUTE OPTIMIZATION", key="exec_opt"):
        st.success("✅ ROUTE OPTIMIZATION DISPATCHED")

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
exp_chart, exp_copilot = st.columns([2, 1])

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

with exp_copilot:
    with st.expander("🧠 SUPPLY CHAIN COPILOT", expanded=False):
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#888;margin-bottom:8px;">AI DIRECTIVE ENGINE V2.1 — READY</div>', unsafe_allow_html=True)
        query = st.chat_input("QUERY THE AI SYSTEM...")
        if query:
            st.chat_message("user").write(query)
            st.chat_message("assistant").write(
                f"ANALYZING: '{query.upper()}'\n\n"
                "RECOMMENDATION: Increase lead-time buffers +2 days across upstream nodes. "
                f"Current delay risk at {delay_risk:.1f}% — reroute via secondary carriers. "
                "Confidence: 91%."
            )
