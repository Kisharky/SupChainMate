"""
views/helpers.py
Shared UI plumbing: theme/CSS injection and chat render helpers.
"""

from __future__ import annotations

import os

import streamlit as st

import config

BASE_DIR = config.BASE_DIR



def load_css(file_name):
    path = os.path.join(BASE_DIR, file_name)
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def apply_theme():
    """Inject the HUD stylesheet and the upload-screen inline CSS."""
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



_TRACE_ICONS = {"route": "🧭", "think": "🧠", "tool": "⚙", "answer": "✓"}


def render_trace(trace):
    if not trace:
        return
    total_ms = sum(s.get("ms", 0) for s in trace)
    with st.expander(f"⚙ REASONING TRACE — {len(trace)} steps"
                     + (f" · {total_ms:,} ms" if total_ms else ""), expanded=False):
        for i, s in enumerate(trace):
            icon = _TRACE_ICONS.get(s.get("step"), "•")
            ms = f" · {s['ms']:,} ms" if s.get("ms") else ""
            st.markdown(f"""
            <div style="border-left:2px solid #333340;padding:4px 12px;margin-left:6px;
                        font-family:'Share Tech Mono',monospace;font-size:0.68rem;">
                <span style="color:#00D4FF;">{icon} STEP {i+1} — {s.get('label','')}{ms}</span><br>
                <span style="color:#999;">{s.get('detail','')}</span>
            </div>""", unsafe_allow_html=True)


def render_artifact(art, key):
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

