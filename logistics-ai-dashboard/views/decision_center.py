"""
views/decision_center.py
Decision Center — the human-in-the-loop trust layer.

Renders pending AI recommendations as cards with WHY drivers, a confidence
score (with its stated basis), quantified business impact, and the
Approve / Reject / Modify workflow, plus decision history and the audit
trail. All state changes go through modules.trust (which writes the audit
log); this view never touches the database directly for writes.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules import store, trust

_CONF_COLORS = [(80, "#00E676"), (55, "#FBC02D"), (0, "#FF003C")]


def _conf_color(score: float) -> str:
    for threshold, color in _CONF_COLORS:
        if score >= threshold:
            return color
    return "#FF003C"


def _impact_chips(impact: dict) -> str:
    chips = []
    if impact.get("cost_savings_yr"):
        chips.append(f"<span style='background:rgba(0,230,118,0.12);border:1px solid #00E676;"
                     f"color:#00E676;padding:2px 8px;font-size:0.62rem;margin-right:6px;'>"
                     f"💰 ${impact['cost_savings_yr']:,.0f} SAVINGS</span>")
    if impact.get("stockout_risk_pct") is not None:
        chips.append(f"<span style='background:rgba(255,0,60,0.12);border:1px solid #FF003C;"
                     f"color:#FF003C;padding:2px 8px;font-size:0.62rem;margin-right:6px;'>"
                     f"📉 {impact['stockout_risk_pct']:.0f}% STOCKOUT RISK</span>")
    if impact.get("service_level_pct") is not None:
        chips.append(f"<span style='background:rgba(0,212,255,0.12);border:1px solid #00D4FF;"
                     f"color:#00D4FF;padding:2px 8px;font-size:0.62rem;margin-right:6px;'>"
                     f"🎯 {impact['service_level_pct']:.0f}% SERVICE LEVEL</span>")
    if impact.get("other"):
        chips.append(f"<span style='border:1px solid #555;color:#AAA;padding:2px 8px;"
                     f"font-size:0.62rem;'>{impact['other']}</span>")
    return "".join(chips) or "<span style='color:#666;font-size:0.62rem;'>impact n/a</span>"


def _render_card(rec: dict) -> None:
    conf = float(rec.get("confidence", 0))
    color = _conf_color(conf)
    st.markdown(f"""
    <div style="background:#151518;border:1px solid #222228;border-left:3px solid {color};
                padding:12px 16px;margin-bottom:2px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <span style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;
                             color:#888;letter-spacing:0.1rem;">{rec.get('category','')} ·
                             BY {rec.get('source','').upper()} · {rec.get('created_ts','')}</span><br>
                <span style="font-family:'Teko',sans-serif;font-size:1.25rem;color:#FFF;
                             letter-spacing:0.04rem;">{rec.get('title','')}</span>
            </div>
            <div style="text-align:center;min-width:90px;">
                <div style="font-family:'Teko',sans-serif;font-size:1.6rem;color:{color};">{conf:.0f}%</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.52rem;color:#666;">CONFIDENCE</div>
            </div>
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#CCC;
                    margin:6px 0;">{rec.get('action','')}</div>
        <div style="margin:6px 0;">{_impact_chips(rec.get('impact') or {})}</div>
    </div>""", unsafe_allow_html=True)

    with st.expander(f"WHY — {len(rec.get('drivers', []))} drivers · confidence basis"):
        for d in rec.get("drivers", []):
            st.markdown(f"""
            <div style="border-left:2px solid #333340;padding:3px 12px;margin-left:6px;
                        font-family:'Share Tech Mono',monospace;font-size:0.7rem;">
                <span style="color:#00D4FF;">{d.get('reason','')}</span>
                <span style="color:#999;"> — {d.get('evidence','')}</span>
            </div>""", unsafe_allow_html=True)
        st.caption(f"Confidence basis: {rec.get('confidence_basis', 'n/a')} · "
                   "Heuristic score (data support + signal strength), not a calibrated probability.")

    key = rec["rec_key"]
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    if c1.button("✓ APPROVE", key=f"dc_ap_{key}", use_container_width=True):
        trust.decide(key, "APPROVED")
        st.rerun()
    if c2.button("✗ REJECT", key=f"dc_rj_{key}", use_container_width=True):
        trust.decide(key, "REJECTED")
        st.rerun()
    mod_note = c3.text_input("Modification", key=f"dc_note_{key}",
                             placeholder="e.g. approve at half quantity",
                             label_visibility="collapsed")
    if c4.button("✎ MODIFY", key=f"dc_md_{key}", use_container_width=True,
                 disabled=not mod_note.strip()):
        trust.decide(key, "MODIFIED", note=mod_note.strip())
        st.rerun()
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


def render(recs) -> None:
    """Sync freshly generated recommendations, then render the center."""
    trust.sync_recommendations(recs)
    kpis = trust.summary_kpis()

    k1, k2, k3, k4 = st.columns(4)
    for col, (label, value, sub, color) in zip([k1, k2, k3, k4], [
        ("AWAITING DECISION", f"{kpis['pending']}", "PENDING RECOMMENDATIONS", "#FBC02D"),
        ("APPROVED", f"{kpis['approved']}", "INCL. MODIFIED", "#00E676"),
        ("APPROVED VALUE", f"${kpis['approved_savings']:,.0f}", "COST SAVINGS COMMITTED", "#00E676"),
        ("AVG CONFIDENCE", f"{kpis['avg_confidence']:.0f}%" if kpis["avg_confidence"] else "—",
         "ACROSS PENDING", "#00D4FF"),
    ]):
        col.markdown(f"""
        <div class="hud-panel" style="border-color:#333340;">
            <div class="hud-label">{label}</div>
            <div style="font-family:'Teko',sans-serif;font-size:1.8rem;color:{color};">{value}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#666;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    pending = trust.pending()
    if not pending:
        st.success("No recommendations awaiting decision — the queue is clear.")
    else:
        for rec in pending:
            _render_card(rec)

    with st.expander(f"📜 DECISION HISTORY — {kpis['approved'] + kpis['rejected']} decided"):
        hist = trust.history()
        if not hist:
            st.caption("No decisions yet.")
        else:
            hist_df = pd.DataFrame([{
                "Decided (UTC)": h.get("decided_ts"),
                "Status": h.get("status"),
                "Recommendation": h.get("title"),
                "Source": h.get("source"),
                "Confidence": h.get("confidence"),
                "Savings ($/yr)": (h.get("impact") or {}).get("cost_savings_yr"),
                "Note": h.get("note") or "",
                "By": h.get("decided_by"),
            } for h in hist])
            st.dataframe(hist_df, use_container_width=True, hide_index=True, height=240)
            st.download_button(
                "⇩ EXPORT DECISION HISTORY (CSV)",
                data=hist_df.to_csv(index=False).encode(),
                file_name="supchainmate_decisions.csv", mime="text/csv",
                use_container_width=True,
            )

    with st.expander("🔏 AUDIT TRAIL — EVERY EVENT, IMMUTABLE"):
        log = store.load_audit_log()
        if not log:
            st.caption("No audit events yet.")
        else:
            log_df = pd.DataFrame(log)
            log_df.columns = ["Timestamp (UTC)", "Actor", "Event", "Rec Key", "Details"]
            st.dataframe(log_df, use_container_width=True, hide_index=True, height=240)
            st.download_button(
                "⇩ EXPORT AUDIT TRAIL (CSV)",
                data=log_df.to_csv(index=False).encode(),
                file_name="supchainmate_audit_trail.csv", mime="text/csv",
                use_container_width=True,
            )
