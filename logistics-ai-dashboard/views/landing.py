"""
views/landing.py
Launch screen — choose Enterprise or Small Retailer mode.
"""

from __future__ import annotations

import streamlit as st


def render():
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
