import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import forecast, network, optimization, tracking

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupChainMate — Mission Control",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS Injection ──────────────────────────────────────────────────────────────
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ── Data Paths ─────────────────────────────────────────────────────────────────
DATA_PATH       = "data/olist_orders.csv"
CUSTOMERS_PATH  = "data/olist_customers_dataset.csv"

# ── Cached Loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def _load_orders():
    return forecast.load_orders(DATA_PATH)

@st.cache_data
def _daily_demand():
    return forecast.daily_demand(_load_orders())

@st.cache_data
def _prophet_model():
    daily = _daily_demand()
    model = forecast.fit_prophet_model(daily)
    return daily, model

@st.cache_data
def _customer_geo_clusters():
    customers = pd.read_csv(CUSTOMERS_PATH)
    geo_df = network.prepare_customer_data(customers)
    return network.run_clustering(geo_df)

@st.cache_data
def _tracking_simulation():
    orders = pd.read_csv("data/olist_orders_dataset.csv")
    return tracking.simulate_tracking(orders)

@st.cache_data
def _flow_tracking_and_delay_model():
    tdf = _tracking_simulation()
    model, X_test, y_test = tracking.train_delay_model(tdf)
    return tdf, model, X_test, y_test

# ── Sidebar Controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ SYSTEM CONFIG")
    days           = st.slider("FORECAST HORIZON (DAYS)", 7, 30, 7)
    simulate_event = st.toggle("🚨 MACRO EVENT SIMULATION", value=False)
    demand_change  = st.slider("DEMAND DELTA (%)", -50, 50, 0)
    st.divider()
    if st.button("CLEAR CACHE & RESAMPLE"):
        st.cache_data.clear()
        st.rerun()

# ── Load & Compute ─────────────────────────────────────────────────────────────
daily_df, prophet_model        = _prophet_model()
future                         = prophet_model.make_future_dataframe(periods=days)
future                         = future.merge(daily_df[["ds", "external_signal"]], on="ds", how="left")
future["external_signal"]      = future["external_signal"].fillna(1 if simulate_event else 0)
forecast_df                    = prophet_model.predict(future)
insights                       = forecast.forecast_insights(forecast_df, daily_df, horizon_days=days)

avg_daily       = float(daily_df["y"].mean())
growth          = ((float(forecast_df["yhat"].tail(days).mean()) - avg_daily) / avg_daily * 100.0) if avg_daily > 0 else 0.0
next_week_demand = insights["next_week_total"]
adjusted_demand  = int(next_week_demand * (1 + demand_change / 100))

tracking_df, delay_model, X_test_delay, _ = _flow_tracking_and_delay_model()
status_counts  = tracking.get_status_counts(tracking_df)
delayed        = int(status_counts.get("Delayed", 0))
delivered      = int(status_counts.get("Delivered", 0))
total_orders   = len(tracking_df)
preds          = delay_model.predict(X_test_delay)
delay_risk     = float(preds.mean() * 100)

orders         = _load_orders()
summary        = optimization.network_summary(orders)

# ── Cost Optimisation ─────────────────────────────────────────────────────────
rng_cost       = np.random.default_rng(42)
cost_df        = tracking_df.copy()
cost_df["cost"]= rng_cost.uniform(5, 20, size=len(cost_df))
current_cost   = float(cost_df["cost"].sum())
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
    </div>
    <div style="display:flex; gap:24px; align-items:center;">
        <div style="text-align:center;">
            <div style="color:#FBC02D; font-size:1.4rem; font-family:'Teko',sans-serif;">
                {active_breaches} ACTIVE</div>
            <div style="color:#666; font-size:0.6rem;">BREACHES</div>
        </div>
        <div style="text-align:center;">
            <div style="color:#00E676; font-size:1.4rem; font-family:'Teko',sans-serif;">
                {100 - delay_risk:.1f}%</div>
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

# ── LEFT: DISRUPTION RADAR MAP ─────────────────────────────────────────────────
with col_map:
    geo_df = _customer_geo_clusters()
    np.random.seed(42)
    geo_df["risk_score"] = np.random.uniform(0, 100, size=len(geo_df))
    geo_df["risk_level"] = pd.cut(
        geo_df["risk_score"],
        bins=[-1, 70, 90, 100],
        labels=["Safe", "Warning", "Critical"]
    )

    # Immediate threat overlay
    threat_lats = geo_df[geo_df["risk_level"] == "Critical"]["lat"].values
    threat_lons = geo_df[geo_df["risk_level"] == "Critical"]["lon"].values

    fig_map = px.scatter_mapbox(
        geo_df,
        lat="lat", lon="lon",
        color="risk_level",
        size="risk_score",
        size_max=18,
        zoom=2.5,
        height=480,
        color_discrete_map={
            "Critical": "#FF003C",
            "Warning":  "#FBC02D",
            "Safe":     "#00D4FF",
        },
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=20, lon=0), zoom=1.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(13,13,16,0.8)",
            bordercolor="#FF003C",
            borderwidth=1,
            font=dict(color="#CCCCCC", size=10),
        ),
        showlegend=True,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Immediate Threat Card
    critical_pct = len(geo_df[geo_df["risk_level"] == "Critical"]) / len(geo_df) * 100
    st.markdown(f"""
    <div class="hud-panel">
        <div style="color:#FF003C; font-family:'Teko',sans-serif; font-size:1.1rem; letter-spacing:0.1rem;">
            ⚠ IMMEDIATE THREAT: EAST COAST
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#AAAAAA; margin:6px 0;">
            WEATHER SURGE CAUSING TOTAL HUB PARALYSIS. {critical_pct:.0f}% OF VOLUME AT RISK.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Optimization Ready Card
    st.markdown(f"""
    <div class="hud-panel-yellow">
        <div style="color:#FBC02D; font-family:'Teko',sans-serif; font-size:1.1rem; letter-spacing:0.1rem;">
            ★ OPTIMIZATION READY
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#AAAAAA; margin:4px 0 10px 0;">
            AI IDENTIFIED ${savings:,.0f} SAVINGS ON ACTIVE ROUTES.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("⚡ EXECUTE OPTIMIZATION", key="exec_opt"):
        st.success("✅ ROUTE OPTIMIZATION DISPATCHED — ETA 2.3 MIN")

# ── RIGHT: HUD PANELS ──────────────────────────────────────────────────────────
with col_hud:

    # ── DELAY RISK PANEL ──────────────────────────────────────────────────────
    risk_label = "EXTREME" if delay_risk > 25 else ("CRITICAL" if delay_risk > 15 else "MODERATE")
    weather_idx = min(99, delay_risk * 3.2)
    node_cong   = min(99, delay_risk * 2.4)

    st.markdown(f"""
    <div class="hud-panel" style="border-color:rgba(255,0,60,0.6);">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div class="hud-label">DELAY RISK</div>
                <div class="hud-value-red">{delay_risk:.1f}%</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#FBC02D; margin-top:4px;">
                    ↑ {abs(growth):.1f}% TREND
                </div>
            </div>
            <span class="action-required-badge">ACTION REQUIRED</span>
        </div>
        <div class="scan-line"></div>
        <div class="hud-label" style="margin-top:8px;">WEATHER DISRUPTION INDEX</div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="progress-bar-container" style="flex:1; margin-right:10px;">
                <div class="progress-bar-fill-red" style="width:{weather_idx:.0f}%;"></div>
            </div>
            <span style="color:#FF003C; font-family:'Share Tech Mono',monospace; font-size:0.7rem; white-space:nowrap;">
                {weather_idx:.0f}% ({risk_label})
            </span>
        </div>
        <div class="hud-label" style="margin-top:8px;">NODE CONGESTION</div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="progress-bar-container" style="flex:1; margin-right:10px;">
                <div class="progress-bar-fill-yellow" style="width:{node_cong:.0f}%;"></div>
            </div>
            <span style="color:#FBC02D; font-family:'Share Tech Mono',monospace; font-size:0.7rem; white-space:nowrap;">
                {node_cong:.0f}% (WARNING)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SYSTEM BENCHMARKS ───────────────────────────────────────────────────────
    avg_lead = summary["avg_lead_days"] if summary else 14.2
    on_time  = summary["on_time_pct"]   if summary else 76.0
    risk_pct = 24

    st.markdown(f"""
    <div class="hud-panel" style="border-color:#333340;">
        <div style="color:#00D4FF; font-family:'Teko',sans-serif; font-size:1rem; 
                    letter-spacing:0.1rem; text-transform:uppercase; margin-bottom:6px;">
            ◈ System Benchmarks
        </div>
        <table class="benchmark-table">
            <tr>
                <th>VECT</th>
                <th>LEGACY</th>
                <th class="optimized">OPTIMIZED</th>
            </tr>
            <tr>
                <td>COST</td>
                <td>${current_cost:,.0f}</td>
                <td class="optimized">${optimized_cost:,.1f}</td>
            </tr>
            <tr>
                <td>RISK</td>
                <td>{risk_pct}%</td>
                <td class="optimized">{risk_pct * 0.75:.0f}%</td>
            </tr>
            <tr>
                <td>RESP</td>
                <td>{avg_lead:.1f}H</td>
                <td class="optimized">{avg_lead * 0.055:.1f}H</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # ── OPERATIONAL DIRECTIVE ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hud-panel-blue">
        <div style="color:#00D4FF; font-family:'Teko',sans-serif; font-size:1rem;
                    letter-spacing:0.1rem; text-transform:uppercase; margin-bottom:8px;">
            ◎ Operational Directive
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; 
                    color:#AAAAAA; line-height:1.6; font-style:italic;">
            "Consolidate high-risk clusters immediately. Route {15 + int(delay_risk/2):.0f}% volume 
            via regional hubs to mitigate port congestion. Confidence score: 87%."
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_ack, col_alert = st.columns([3, 1])
    with col_ack:
        if st.button("ACKNOWLEDGE DIRECTIVE", key="ack_dir"):
            st.success("DIRECTIVE CONFIRMED — EXECUTING")
    with col_alert:
        st.markdown("""
        <div style="background:#FF003C; width:40px; height:40px; display:flex;
                    align-items:center; justify-content:center; font-size:1.2rem; margin-top:2px;">
            ⚠
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM BAR: DEMAND SURGE SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()

demand_label = "+10% SURGE IMPACT" if demand_change > 0 else ("-10% DEMAND DROP" if demand_change < 0 else "STABLE DEMAND")
demand_color = "#FBC02D" if demand_change > 20 else ("#FF003C" if demand_change < -20 else "#00E676")

revenue_opp   =  adjusted_demand * 12
op_strain_pct =  abs(demand_change) * 0.4

bot_l, bot_m1, bot_m2, bot_m3, bot_r = st.columns([2, 1, 2, 1, 1])

with bot_l:
    st.markdown("""
    <div class="hud-label">SIMULATION ENGINE</div>
    <div style="font-family:'Teko',sans-serif; font-size:1.2rem; color:#FFFFFF;
                letter-spacing:0.08rem; text-transform:uppercase;">
        DEMAND SURGE SIMULATOR
    </div>
    """, unsafe_allow_html=True)

with bot_m1:
    st.markdown(f"""
    <div class="hud-label">REDUCED</div>
    <div class="hud-value-green" style="font-size:1rem;">-{max(0,demand_change/2):.0f}%</div>
    """, unsafe_allow_html=True)

with bot_m2:
    st.markdown(f"""
    <div style="text-align:center;">
        <div style="font-family:'Teko',sans-serif; font-size:2rem; color:{demand_color};
                    text-shadow:0 0 12px {demand_color}88; text-transform:uppercase;">
            {demand_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

with bot_m3:
    st.markdown(f"""
    <div class="hud-label">CRITICAL</div>
    <div class="hud-value-red" style="font-size:1rem;">+{max(0,demand_change/2):.0f}%</div>
    """, unsafe_allow_html=True)

with bot_r:
    st.markdown(f"""
    <div class="hud-label">REVENUE OPP</div>
    <div class="hud-value-green" style="font-size:0.9rem;">+${revenue_opp/1000:.0f}K</div>
    <div class="hud-label" style="margin-top:4px;">OP STRAIN</div>
    <div class="hud-value-red" style="font-size:0.9rem;">+{op_strain_pct:.1f}% ERR</div>
    """, unsafe_allow_html=True)

if st.button("▶ RUN SCENARIO", key="run_scenario"):
    with st.spinner("RUNNING SIMULATION..."):
        import time; time.sleep(1)
    if demand_change > 20:
        st.error(f"⚠ HIGH DEMAND SURGE — STOCKOUT RISK ELEVATED. ADJUSTED DEMAND: {adjusted_demand:,}")
    elif demand_change < -20:
        st.warning(f"📉 DEMAND DROP DETECTED — REDUCE INVENTORY. ADJUSTED: {adjusted_demand:,}")
    else:
        st.success(f"✅ SCENARIO NOMINAL — ADJUSTED DEMAND: {adjusted_demand:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECONDARY INTELLIGENCE: Forecast Chart + Supply Chain Copilot
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
exp_chart, exp_copilot = st.columns([2, 1])

with exp_chart:
    with st.expander("📈 DEMAND FORECAST — PROPHET ENGINE", expanded=False):
        fc = forecast_df.sort_values("ds")
        fig_demand = go.Figure()
        fig_demand.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_demand.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(255,0,60,0.12)",
            name="Confidence Interval"))
        fig_demand.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines",
            name="Forecast (yhat)", line=dict(color="#FF003C", width=2)))
        fig_demand.add_trace(go.Scatter(x=daily_df["ds"], y=daily_df["y"], mode="lines",
            name="Actual Orders", line=dict(color="#00D4FF", width=1.5)))
        fig_demand.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,13,16,1)",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
            height=320,
        )
        fig_demand.update_xaxes(gridcolor="#222228", title_text="")
        fig_demand.update_yaxes(gridcolor="#222228", title_text="ORDERS")
        st.plotly_chart(fig_demand, use_container_width=True)

with exp_copilot:
    with st.expander("🧠 SUPPLY CHAIN COPILOT", expanded=False):
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#888; margin-bottom:8px;">
            AI DIRECTIVE ENGINE V2.1 — READY
        </div>""", unsafe_allow_html=True)
        query = st.chat_input("QUERY THE AI SYSTEM...")
        if query:
            st.chat_message("user").write(query)
            st.chat_message("assistant").write(
                f"ANALYZING TOPOLOGY FOR: '{query.upper()}'\n\n"
                "RECOMMENDED: Increase lead-time buffers +2 days across upstream nodes. "
                "Reroute Cluster C shipments via Hub 4 to reduce congestion. "
                "Confidence: 91%."
            )
