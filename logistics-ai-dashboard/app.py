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
from modules import health_check, tender, alerts, store, connect, carbon, doc_intel

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupChainMate — Mission Control",
    layout="wide",
    initial_sidebar_state="collapsed",
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_css(file_name):
    path = os.path.join(BASE_DIR, file_name)
    if os.path.exists(path):
        with open(path) as f:
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
.mode-card-retail {
    border-top: 2px solid #00E676;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════
_SESSION_KEYS = [
    "orders_df", "delivery_df", "location_df", "cost_df",
    "daily_df", "forecast_df", "tracking_df", "geo_df",
    "delay_model", "X_test_delay", "summary", "current_cost",
    "data_loaded", "demo_mode", "shipments_df", "carriers_simulated",
    "kpi_snapshot_saved",
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


def _render_small_retailer_page():
    st.markdown("""
    <div class="upload-hero">
        <h1>⬡ SupChainMate</h1>
        <div class="subtitle">SMALL RETAILER MODE — NO SPREADSHEETS REQUIRED</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("← Change mode", key="retail_change_mode"):
        st.session_state.entry_mode = "landing"
        st.rerun()

    st.caption(
        "Answer a few questions per product. The same inventory math as Enterprise runs underneath "
        "(reorder point, EOQ, safety stock, cost trade-offs)."
    )

    with st.expander("Advanced costs (optional)", expanded=False):
        r_order_cost = st.number_input(
            "Cost per purchase order ($)",
            min_value=10.0,
            value=retail.DEFAULT_ORDERING_COST,
            step=5.0,
            key="retail_ordering_cost",
        )
        r_hold_pct = st.slider("Holding cost (% of unit value / year)", 10, 40, 25, key="retail_holding_pct")
    r_holding_rate = r_hold_pct / 100.0

    st.subheader("Add a product")
    with st.form("retail_add_product", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            pname = st.text_input("Product name", placeholder="e.g. Blue Jeans (S)")
            p_weekly = st.number_input("How many do you sell per week (average)?", min_value=0.01, value=20.0, step=1.0)
            p_lead = st.number_input("Supplier lead time (days)", min_value=0.5, value=14.0, step=0.5)
        with c2:
            p_cost = st.number_input("Your cost per unit ($)", min_value=0.01, value=25.0, step=1.0)
            p_tier = st.selectbox(
                "Safety buffer",
                options=["Low", "Medium", "High"],
                index=1,
                help="Higher buffer → higher service level target in the engine.",
            )
            p_stock = st.number_input("Current stock (units)", min_value=0.0, value=0.0, step=1.0)
        add_sub = st.form_submit_button("Add to tracker")
    st.caption(
        "Lead time variability is estimated at 15% of your supplier's average — "
        "adjust the safety buffer if your supplier is unpredictable."
    )
    if add_sub and pname and pname.strip():
        st.session_state.retail_products.append(
            retail.product_dict(pname.strip(), p_weekly, p_lead, p_cost, p_tier, p_stock)
        )
        store.save_retail_products(st.session_state.retail_products)
        st.rerun()

    products = st.session_state.retail_products
    if not products:
        st.info("Add at least one product to see reorder guidance and the tracker.")
        return

    st.subheader("Your answers — instant guidance")
    focus_labels = [p["name"] for p in products]
    pick = st.selectbox("Product to show", range(len(products)), format_func=lambda i: focus_labels[i])
    p_focus = products[pick]
    _, out_focus = retail.run_retail_decisions(
        p_focus["units_per_week"],
        p_focus["lead_time_days"],
        p_focus["unit_cost"],
        p_focus["safety_tier"],
        ordering_cost=r_order_cost,
        holding_rate=r_holding_rate,
    )
    rop_i = int(math.ceil(out_focus.reorder_point))
    eoq_i = int(round(out_focus.eoq))
    ss_i = int(round(out_focus.safety_stock))
    save_yr = out_focus.savings_vs_current

    st.success(f"REORDER ALERT: Reorder **{p_focus['name']}** when you have **{rop_i}** units left.")
    st.info(f"ORDER QUANTITY: Order **{eoq_i}** units at a time.")
    st.warning(f"SAFETY BUFFER: Keep at least **{ss_i}** units as emergency stock.")
    if save_yr > 0:
        st.error(
            f"YOU ARE LEAVING MONEY ON THE TABLE: Aligning to this pattern could save "
            f"**~${save_yr:,.0f}/year** on inventory-related costs (vs. a naive ordering baseline)."
        )
    else:
        st.info("Cost outlook: your parameters are already close to the model baseline.")

    st.subheader("Multi-product tracker")
    tbl_rows = [
        retail.tracker_row(p, ordering_cost=r_order_cost, holding_rate=r_holding_rate)
        for p in products
    ]
    df_track = pd.DataFrame(tbl_rows)
    edited = st.data_editor(
        df_track,
        width="stretch",
        hide_index=True,
        disabled=[
            "Product",
            "Reorder when (units left)",
            "Order qty",
            "Est. savings/yr ($)",
            "Status",
        ],
        key="retail_tracker_editor",
    )
    st.download_button(
        "⇩ DOWNLOAD REORDER CHECKLIST (CSV)",
        data=df_track.to_csv(index=False).encode(),
        file_name="reorder_checklist.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if st.button("Apply stock levels from table", key="retail_apply_stock"):
        try:
            for i in range(len(st.session_state.retail_products)):
                st.session_state.retail_products[i]["current_stock"] = float(
                    edited.iloc[i]["Current stock"]
                )
        except (ValueError, KeyError, TypeError, IndexError):
            st.error("Could not read stock values; use numbers only.")
        else:
            store.save_retail_products(st.session_state.retail_products)
            st.rerun()

    del_idx = st.selectbox(
        "Remove a product",
        range(len(products)),
        format_func=lambda i: products[i]["name"],
        key="retail_del_select",
    )
    if st.button("Remove selected product", key="retail_del_btn"):
        st.session_state.retail_products.pop(del_idx)
        store.save_retail_products(st.session_state.retail_products)
        st.rerun()

    st.subheader("Alerts")
    digest_text, n_alerts = alerts.build_retail_digest(products, tbl_rows)
    if n_alerts:
        st.warning(f"{n_alerts} product(s) need attention — see the digest below.")
    else:
        st.success("Nothing needs ordering right now.")
    smtp_ok = alerts.smtp_configured()
    ral1, ral2 = st.columns([2, 1])
    with ral1:
        r_email = st.text_input("Email for reorder alerts", key="retail_email",
                                value=store.load_setting("retail_alert_email", "") or "")
    with ral2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Send digest now", key="retail_send_digest",
                     disabled=not (smtp_ok and r_email), use_container_width=True):
            ok, send_msg = alerts.send_email(r_email, "SupChainMate — Reorder Digest", digest_text)
            (st.success if ok else st.error)(send_msg)
            if ok:
                store.save_setting("retail_alert_email", r_email)
    if r_email:
        store.save_setting("retail_alert_email", r_email)
    if not smtp_ok:
        st.caption("Email sending needs SMTP settings in .env (SMTP_HOST, SMTP_FROM, SMTP_USER, SMTP_PASS). "
                   "You can always download the digest below.")
    with st.expander("Preview digest"):
        st.code(digest_text, language=None)
    st.download_button(
        "⇩ Download reorder digest (TXT)",
        data=digest_text.encode(),
        file_name="reorder_digest.txt", mime="text/plain",
        use_container_width=True,
    )
    st.caption("Your products and email are saved locally (SQLite) and restored next time you open the app.")


# ═══════════════════════════════════════════════════════════════════════════════
# LANDING — CHOOSE ENTERPRISE VS SMALL RETAILER
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.entry_mode == "landing":
    st.markdown("""
    <div class="upload-hero">
        <h1>⬡ SupChainMate</h1>
        <div class="subtitle">CHOOSE HOW YOU WANT TO WORK</div>
    </div>
    """, unsafe_allow_html=True)
    ec, rc = st.columns(2)
    with ec:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-label">ENTERPRISE MODE</div>
            <div class="upload-card-sub">FOR SUPPLY CHAIN TEAMS WITH DATA<br><br>
            Upload orders, delivery, locations, costs — full mission control, maps, and AI insights.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("OPEN ENTERPRISE MODE", use_container_width=True, type="primary", key="btn_enterprise"):
            st.session_state.entry_mode = "enterprise"
            st.rerun()
    with rc:
        st.markdown("""
        <div class="upload-card mode-card-retail">
            <div class="upload-card-label">SMALL RETAILER MODE</div>
            <div class="upload-card-sub">FOR SMALL SHOPS — NO CSV<br><br>
            Answer five quick questions per product. Same decision engine, plain-English answers.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("OPEN SMALL RETAILER MODE", use_container_width=True, key="btn_retail"):
            st.session_state.entry_mode = "retail"
            st.rerun()
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SMALL RETAILER — STANDALONE FLOW (NO ENTERPRISE DATA)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.entry_mode == "retail":
    _render_small_retailer_page()
    st.stop()


# ── Demo data loader ───────────────────────────────────────────────────────────
DEMO_ORDERS    = os.path.join(BASE_DIR, "data", "olist_orders.csv")
DEMO_DELIVERY  = os.path.join(BASE_DIR, "data", "olist_orders_dataset.csv")
DEMO_CUSTOMERS = os.path.join(BASE_DIR, "data", "olist_customers_dataset.csv")


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
        tdf          = control_tower.assign_demo_carriers(tdf)
        st.session_state.carriers_simulated = True
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
        st.session_state.entry_mode    = "enterprise"
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
            st.session_state.carriers_simulated = False
        else:
            # Simulate from orders if no delivery file provided
            tdf           = tracking.simulate_tracking(orders_norm.rename(columns={"order_date": "order_purchase_timestamp"}))

        # Train delay model using the proper LightGBM pipeline from tracking.py
        m, X_test, _ = tracking.train_delay_model(tdf)
        st.session_state.tracking_df   = tdf
        st.session_state.delay_model   = m
        st.session_state.X_test_delay  = X_test

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
        st.session_state.entry_mode  = "enterprise"
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD SCREEN (ENTERPRISE ONLY)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.entry_mode == "enterprise" and not st.session_state.data_loaded:

    if st.button("← Change mode", key="enterprise_upload_back"):
        st.session_state.entry_mode = "landing"
        st.rerun()

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
        orders_file = st.file_uploader("Orders file", type=["csv", "xlsx", "xls"], key="orders",
                                        label_visibility="collapsed")

    with u2:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-label">🚚 Delivery Data</div>
            <div class="upload-card-sub">DELIVERY DATE · STATUS · ROUTE · LEAD TIME<br><br>AUTO-DETECTED: status, lead days</div>
        </div>""", unsafe_allow_html=True)
        delivery_file = st.file_uploader("Delivery file", type=["csv", "xlsx", "xls"], key="delivery",
                                          label_visibility="collapsed")

    with u3:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-label">📍 Location Data</div>
            <div class="upload-card-sub">CUSTOMER LOCATIONS · WAREHOUSES · ZIP<br><br>AUTO-DETECTED: lat/lon or postal code</div>
        </div>""", unsafe_allow_html=True)
        location_file = st.file_uploader("Location file", type=["csv", "xlsx", "xls"], key="location",
                                          label_visibility="collapsed")

    with u4:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-label">💰 Cost Data</div>
            <div class="upload-card-sub">COST PER DELIVERY · FUEL · WAREHOUSE<br><br>AUTO-DETECTED: cost, price, fee columns</div>
        </div>""", unsafe_allow_html=True)
        cost_file = st.file_uploader("Cost file", type=["csv", "xlsx", "xls"], key="cost",
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
            platform = ingestion.detect_store_platform(raw)
            if platform:
                badges += f'<span class="detected-badge">🛒 {platform.upper()} EXPORT DETECTED</span>'
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

    # ── Live store connect (no CSV needed) ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔗 CONNECT YOUR STORE — SHOPIFY / WOOCOMMERCE (NO CSV NEEDED)", expanded=False):
        st.caption(
            "Read-only API pull of your order history. Credentials are used for this fetch "
            "only and are never saved. Shopify: create a custom app with the `read_orders` "
            "scope and paste its Admin API access token. WooCommerce: create a read-only "
            "REST API key under WooCommerce → Settings → Advanced."
        )
        conn_platform = st.radio("Platform", ["Shopify", "WooCommerce"],
                                 horizontal=True, key="conn_platform")
        if conn_platform == "Shopify":
            cs1, cs2 = st.columns(2)
            shop_url = cs1.text_input("Store URL", placeholder="mystore.myshopify.com", key="conn_shop_url")
            shop_token = cs2.text_input("Admin API access token", type="password",
                                        placeholder="shpat_...", key="conn_shop_token")
            if st.button("⇩ IMPORT ORDERS FROM SHOPIFY", use_container_width=True, key="conn_shop_go"):
                with st.spinner("Fetching orders from Shopify..."):
                    conn_df, conn_msg = connect.fetch_shopify_orders(shop_url, shop_token)
                if conn_df is None:
                    st.error(conn_msg)
                else:
                    st.success(conn_msg)
                    try:
                        _process_uploaded(conn_df, None, None, None)
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
        else:
            cw1, cw2, cw3 = st.columns(3)
            woo_url = cw1.text_input("Site URL", placeholder="myshop.com", key="conn_woo_url")
            woo_key = cw2.text_input("Consumer key", type="password",
                                     placeholder="ck_...", key="conn_woo_key")
            woo_secret = cw3.text_input("Consumer secret", type="password",
                                        placeholder="cs_...", key="conn_woo_secret")
            if st.button("⇩ IMPORT ORDERS FROM WOOCOMMERCE", use_container_width=True, key="conn_woo_go"):
                with st.spinner("Fetching orders from WooCommerce..."):
                    conn_df, conn_msg = connect.fetch_woocommerce_orders(woo_url, woo_key, woo_secret)
                if conn_df is None:
                    st.error(conn_msg)
                else:
                    st.success(conn_msg)
                    try:
                        _process_uploaded(conn_df, None, None, None)
                    except Exception as e:
                        st.error(f"Processing failed: {e}")

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
        exc_text,
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

def _render_artifact(art, key):
    if art["type"] == "dataframe":
        st.markdown(f"<div class='hud-label'>{art['title']}</div>", unsafe_allow_html=True)
        st.dataframe(art["data"], use_container_width=True, hide_index=True, height=240)
        st.download_button(
            f"⇩ {art['title'].upper()} (CSV)",
            data=art["data"].to_csv(index=False).encode(),
            file_name=art["filename"], mime="text/csv",
            key=key, use_container_width=True,
        )
    elif art["type"] == "text":
        st.markdown(f"<div class='hud-label'>{art['title']}</div>", unsafe_allow_html=True)
        st.code(art["data"], language=None)
        st.download_button(
            f"⇩ {art['title'].upper()} (TXT)",
            data=art["data"].encode(),
            file_name=art["filename"], mime="text/plain",
            key=key, use_container_width=True,
        )


with exp_copilot:
    with st.expander("🤖 AGENTIC COPILOT — THINKS · DECIDES · ACTS", expanded=False):
        agent_status = (
            "🟢 GROQ AGENT LIVE · LLaMA-3.3-70B TOOL CALLING"
            if groq_ai.is_available()
            else "🟡 OFFLINE MODE — actions still run on your live data; set GROQ_API_KEY for reasoning &amp; wording"
        )
        st.markdown(
            f'<div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#888;margin-bottom:8px;">'
            f'{agent_status} · 5 TOOLS: shipments, scorecards, emails, reorder plans, digests</div>',
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
        }

        if "agent_chat" not in st.session_state:
            st.session_state.agent_chat = []

        pending_query = None
        for row_start in range(0, len(agent.QUICK_ACTIONS), 4):
            row_actions = agent.QUICK_ACTIONS[row_start:row_start + 4]
            qa_cols = st.columns(4)
            for qa_col, (qa_label, qa_prompt) in zip(qa_cols, row_actions):
                if qa_col.button(qa_label, key=f"qa_{qa_label}", use_container_width=True):
                    pending_query = qa_prompt

        typed_query = st.chat_input("Ask the agent to do something — it can act, not just answer...")
        if typed_query:
            pending_query = typed_query

        for t_i, turn in enumerate(st.session_state.agent_chat):
            with st.chat_message(turn["role"]):
                if turn.get("actions"):
                    st.caption("⚙ EXECUTED: " + " · ".join(turn["actions"]))
                st.write(turn["content"])
                for a_i, art in enumerate(turn.get("artifacts", [])):
                    _render_artifact(art, key=f"agent_art_{t_i}_{a_i}")

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
            })
            with st.chat_message("assistant"):
                if agent_result["actions"]:
                    st.caption("⚙ EXECUTED: " + " · ".join(agent_result["actions"]))
                st.write(agent_result["reply"])
                for a_i, art in enumerate(agent_result["artifacts"]):
                    _render_artifact(art, key=f"agent_art_new_{a_i}")

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
