import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import forecast, network, optimization, tracking

st.set_page_config(page_title="SupChainMate", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Logistics Intelligence Dashboard")
st.caption("Unified demand, delivery, and flow intelligence — Olist orders dataset.")

DATA_PATH = "data/olist_orders.csv"
CUSTOMERS_PATH = "data/olist_customers_dataset.csv"


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
    """Single cache so simulated statuses and delay model stay aligned."""
    tdf = _tracking_simulation()
    model, X_test, y_test = tracking.train_delay_model(tdf)
    return tdf, model, X_test, y_test


st.sidebar.header("Controls")
days = st.sidebar.slider("Forecast Days", 7, 30, 7)

daily_df, prophet_model = _prophet_model()
future = prophet_model.make_future_dataframe(periods=days)
forecast_df = prophet_model.predict(future)
insights = forecast.forecast_insights(forecast_df, daily_df, horizon_days=days)

avg_daily = float(daily_df["y"].mean())
if avg_daily > 0:
    growth = (
        (float(forecast_df["yhat"].tail(days).mean()) - avg_daily) / avg_daily
    ) * 100.0
else:
    growth = 0.0

next_week_demand = insights["next_week_total"]

tab_demand, tab_network, tab_flow = st.tabs(
    [
        "Demand prediction",
        "Network / delivery insights",
        "Flow / status simulation",
    ]
)

with tab_demand:
    st.write("### 📈 Demand Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Total Forecast Demand", next_week_demand)
    col2.metric("📊 Avg Daily Demand", round(avg_daily, 2))
    col3.metric("📈 Growth %", f"{growth:.2f}%")

    if growth > 10:
        st.success(
            f"📈 Demand is expected to increase by {growth:.2f}% — consider increasing inventory."
        )
    elif growth < -10:
        st.warning(
            f"📉 Demand is expected to drop by {abs(growth):.2f}% — reduce stock levels."
        )
    else:
        st.info("⚖️ Demand is stable — maintain current inventory strategy.")

    stock_threshold = avg_daily * 0.8
    if float(forecast_df["yhat"].tail(days).mean()) > stock_threshold:
        st.error("⚠️ High stockout risk detected — increase inventory immediately.")

    st.subheader("💡 Recommendations")
    if growth > 10:
        st.write("- Increase procurement volume")
        st.write("- Prioritise high-demand SKUs")
    elif growth < -10:
        st.write("- Reduce inventory levels")
        st.write("- Avoid overstock risk")
    else:
        st.write("- Maintain current inventory strategy")

    st.divider()
    st.subheader("📉 Historical vs Forecast Demand")
    st.caption(
        "Meta **Prophet** (open source): build `Prophet()`, `fit` on history with columns **`ds` / `y`**, then "
        "`make_future_dataframe` → `predict` — exactly like the official Python quick start. "
        "The chart shows **`yhat`** plus **`yhat_lower` / `yhat_upper`** from `forecast[[...]]`."
    )

    fc = forecast_df.sort_values("ds")
    fig_demand = go.Figure()
    fig_demand.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_demand.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc["yhat_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.25)",
            name="Predictive interval (yhat lower–upper)",
        )
    )
    fig_demand.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc["yhat"],
            mode="lines",
            name="Forecast (yhat)",
            line=dict(color="rgb(99, 110, 250)", width=2),
        )
    )
    fig_demand.add_trace(
        go.Scatter(
            x=daily_df["ds"],
            y=daily_df["y"],
            mode="lines",
            name="Actual (daily orders)",
            line=dict(color="rgb(44, 160, 44)", width=2),
        )
    )
    fig_demand.update_layout(
        title="Demand: actuals vs Prophet forecast and uncertainty",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig_demand.update_xaxes(title_text="Date (ds)")
    fig_demand.update_yaxes(title_text="Orders (y / yhat)")
    st.plotly_chart(fig_demand, width="stretch")

with tab_network:
    st.subheader("Network / delivery insights")
    st.caption(
        "Spatial clusters from customer ZIP prefixes (simulated coordinates), plus delivery KPIs from orders."
    )

    st.subheader("🗺️ Delivery Zone Intelligence")
    geo_df = _customer_geo_clusters()
    fig_zones = px.scatter_mapbox(
        geo_df,
        lat="lat",
        lon="lon",
        color="cluster",
        zoom=3,
        height=500,
        title="Customer clusters (proxy geography for zone planning)",
    )
    fig_zones.update_layout(mapbox_style="open-street-map")
    fig_zones.update_mapboxes(center=dict(lat=-14.0, lon=-51.0), zoom=3)
    st.plotly_chart(fig_zones, use_container_width=True)

    st.subheader("📦 Zone Insights")
    cluster_counts = geo_df["cluster"].value_counts()
    st.write("High-density delivery zones:")
    st.write(cluster_counts.head())

    st.divider()

    orders = _load_orders()
    summary = optimization.network_summary(orders)
    if summary is None:
        st.warning("No completed customer delivery dates found — cannot compute network KPIs.")
    else:
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Avg lead time (days)", f"{summary['avg_lead_days']:.1f}")
        n2.metric("Median lead (days)", f"{summary['median_lead_days']:.1f}")
        n3.metric("On-time vs estimate", f"{summary['on_time_pct']:.1f}%")
        n4.metric("Orders in sample", f"{summary['n_delivered_observed']:,}")

        st.divider()
        daily_net = optimization.daily_network_metrics(orders)
        seg = optimization.cluster_operating_days(daily_net, k=3)

        st.markdown("##### Daily volume vs median lead time")
        st.line_chart(
            daily_net.set_index("day")[["orders", "median_lead"]].rename(
                columns={"orders": "Orders", "median_lead": "Median lead (days)"}
            )
        )

        st.markdown("##### Operating-day segments (volume + delay)")
        st.dataframe(
            seg[["day", "orders", "median_lead", "on_time_rate", "segment"]]
            .tail(30)
            .sort_values("day", ascending=False),
            width="stretch",
            hide_index=True,
        )

with tab_flow:
    st.subheader("📦 Supply Chain Flow Tracking")
    st.caption(
        "Simulated operational statuses (Processing / Shipped / Delivered / Delayed) for live-style "
        "visibility — same row count as `olist_orders_dataset.csv`. Clear Streamlit cache to resample."
    )

    tracking_df, delay_model, X_test_delay, _ = _flow_tracking_and_delay_model()
    status_counts = tracking.get_status_counts(tracking_df)

    st.write("### 🚚 Order Status Overview")
    st.bar_chart(status_counts)

    delayed = int(status_counts.get("Delayed", 0))
    if delayed > 0.15 * len(tracking_df):
        st.error("⚠️ High delay risk detected in supply chain")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders", len(tracking_df))
    col2.metric("Delivered", int(status_counts.get("Delivered", 0)))
    col3.metric("Delayed", delayed)

    st.divider()
    st.subheader("🤖 Delay Prediction Model")
    preds = delay_model.predict(X_test_delay)
    delay_risk = float(preds.mean() * 100)
    st.metric("Predicted Delay Risk", f"{delay_risk:.2f}%")
    if delay_risk > 15:
        st.error("⚠️ High predicted delay risk — investigate logistics bottlenecks.")
    else:
        st.success("✅ Delay risk is within acceptable limits.")

    st.divider()
    st.subheader("💰 Cost Optimisation Scenario")
    cost_df = tracking_df.copy()
    rng_cost = np.random.default_rng(42)
    cost_df["cost"] = rng_cost.uniform(5, 20, size=len(cost_df))
    current_cost = float(cost_df["cost"].sum())
    optimized_cost = current_cost * 0.85
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Cost", f"${current_cost:,.2f}")
    m2.metric("Optimized Cost", f"${optimized_cost:,.2f}")
    m3.metric("Savings", f"${current_cost - optimized_cost:,.2f}")
    st.info(
        "💡 Optimising delivery routes and batching orders could reduce logistics costs by ~15%."
    )

    st.divider()
    st.subheader("🔮 What-If Simulation")
    demand_change = st.slider("Change in Demand (%)", -50, 50, 0)
    adjusted_demand = next_week_demand * (1 + demand_change / 100)
    st.metric("Adjusted Demand", int(adjusted_demand))
    if demand_change > 20:
        st.warning("⚠️ High demand surge — risk of stockouts and delays.")
    elif demand_change < -20:
        st.info("📉 Demand drop — consider reducing inventory.")
    else:
        st.success("✅ Demand within manageable range.")
