"""
views/pipeline.py
Data pipeline — demo loading and upload processing into session state.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from modules import (control_tower, forecast, ingestion, network,
                     optimization, sku, tracking)

# ── Demo data loader ───────────────────────────────────────────────────────────
import config as app_config

DEMO_ORDERS    = app_config.DEMO_ORDERS
DEMO_DELIVERY  = app_config.DEMO_DELIVERY
DEMO_CUSTOMERS = app_config.DEMO_CUSTOMERS


def load_demo():
    """Load and process the built-in Olist demo dataset."""
    for _demo_path in (DEMO_ORDERS, DEMO_DELIVERY, DEMO_CUSTOMERS):
        if not os.path.exists(_demo_path):
            st.error(f"Demo dataset missing: {_demo_path}. Re-clone the repository "
                     f"or restore the CSVs under logistics-ai-dashboard/data/.")
            return
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

        # Per-SKU intelligence: simulated catalogue over the real order dates
        _sku_orders = pd.DataFrame({
            "order_date": raw_orders["order_purchase_timestamp"],
            "quantity": 1.0,
        })
        st.session_state.orders_sku_df = sku.assign_demo_skus(_sku_orders)
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


def process_uploaded(raw_orders, raw_delivery, raw_location, raw_cost):
    """Normalise and process user-uploaded files."""
    with st.spinner("PROCESSING YOUR DATA..."):
        # ── Orders ────────────────────────────────────────────────────────────
        orders_norm  = ingestion.normalise_orders(raw_orders)
        st.session_state.orders_sku_df = orders_norm if "sku" in orders_norm.columns else None
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


