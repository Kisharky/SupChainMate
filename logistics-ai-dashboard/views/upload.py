"""
views/upload.py
Enterprise upload screen — file cards, auto-detection badges, store connect.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules import connect, ingestion
from views import pipeline


def render():

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
                    pipeline.process_uploaded(raw_orders, raw_delivery, raw_location, raw_cost)
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
            pipeline.load_demo()

    # ── Live store connect (no CSV needed) ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔗 CONNECT YOUR STORE — SHOPIFY / WOOCOMMERCE (NO CSV NEEDED)", expanded=False):
        st.caption(
            "Read-only API pull of your order history. Credentials are used for this fetch "
            "only and are never saved. Shopify: create a custom app with the `read_orders` "
            "scope and paste its Admin API access token. WooCommerce: create a read-only "
            "REST API key under WooCommerce → Settings → Advanced."
        )
        conn_platform = st.radio(
            "Platform", ["Shopify", "WooCommerce", "ERPNext", "Generic REST API"],
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
                        pipeline.process_uploaded(conn_df, None, None, None)
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
        elif conn_platform == "ERPNext":
            ce1, ce2, ce3 = st.columns(3)
            erp_url = ce1.text_input("ERPNext site URL", placeholder="erp.mycompany.com",
                                     key="conn_erp_url")
            erp_key = ce2.text_input("API key", type="password", key="conn_erp_key")
            erp_secret = ce3.text_input("API secret", type="password", key="conn_erp_secret")
            if st.button("⇩ IMPORT SALES ORDERS FROM ERPNEXT",
                         use_container_width=True, key="conn_erp_go"):
                with st.spinner("Fetching sales orders from ERPNext..."):
                    conn_df, conn_msg = connect.fetch_erpnext_orders(erp_url, erp_key, erp_secret)
                if conn_df is None:
                    st.error(conn_msg)
                else:
                    st.success(conn_msg)
                    try:
                        pipeline.process_uploaded(conn_df, None, None, None)
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
        elif conn_platform == "Generic REST API":
            st.caption("For SAP / Oracle Fusion / Dynamics 365 gateways or any custom JSON "
                       "API: point at the endpoint, name the records array and fields.")
            cr1, cr2 = st.columns([2, 1])
            rest_url = cr1.text_input("Endpoint URL",
                                      placeholder="https://api.mycompany.com/v1/orders",
                                      key="conn_rest_url")
            rest_token = cr2.text_input("Bearer token (optional)", type="password",
                                        key="conn_rest_token")
            cr3, cr4, cr5 = st.columns(3)
            rest_path = cr3.text_input("Records path", placeholder="data.orders (blank = root)",
                                       key="conn_rest_path")
            rest_date = cr4.text_input("Date field", placeholder="order_date",
                                       key="conn_rest_date")
            rest_qty = cr5.text_input("Quantity field (optional)", placeholder="qty",
                                      key="conn_rest_qty")
            if st.button("⇩ IMPORT FROM REST API", use_container_width=True, key="conn_rest_go"):
                with st.spinner("Fetching from the REST endpoint..."):
                    conn_df, conn_msg = connect.fetch_rest_orders(
                        rest_url, rest_path, rest_date, rest_qty, rest_token)
                if conn_df is None:
                    st.error(conn_msg)
                else:
                    st.success(conn_msg)
                    try:
                        pipeline.process_uploaded(conn_df, None, None, None)
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
                        pipeline.process_uploaded(conn_df, None, None, None)
                    except Exception as e:
                        st.error(f"Processing failed: {e}")

    st.stop()  # Don't render dashboard until data is loaded
