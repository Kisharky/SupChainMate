"""
views/retail.py
Small Retailer mode — form-based per-product decisions, tracker, alerts.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from modules import alerts, retail, store


def render():
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


