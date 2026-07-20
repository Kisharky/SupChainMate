"""
views/agents_hub.py
Agent Orchestrator UI — pick a workflow, watch the agents run in sequence,
see each agent's reasoning, confidence, impact, and what it handed
downstream. Recommendations route to the Decision Center for approval.
"""

from __future__ import annotations

import streamlit as st

from modules.agents import build_default_orchestrator

_CONF = [(80, "#00E676"), (55, "#FBC02D"), (0, "#FF003C")]


def _color(score: float) -> str:
    for t, c in _CONF:
        if score >= t:
            return c
    return "#FF003C"


def _render_result(result, step: int, total: int) -> None:
    color = _color(result.confidence)
    handoff = (" · ".join(f"{k}={v}" for k, v in list(result.outputs.items())[:4])
               if result.outputs else "—")
    st.markdown(f"""
    <div style="background:#151518;border:1px solid #222228;border-left:3px solid {color};
                padding:12px 16px;margin-bottom:6px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <span style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;
                             color:#888;letter-spacing:0.1rem;">STEP {step}/{total} ·
                             {result.duration_ms:,} MS
                             {'· <span style="color:#FBC02D;">NEEDS APPROVAL</span>' if result.requires_approval else ''}</span><br>
                <span style="font-family:'Teko',sans-serif;font-size:1.25rem;color:#FFF;
                             letter-spacing:0.05rem;">🤖 {result.agent.replace('_', ' ').upper()}</span><br>
                <span style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;
                             color:#00D4FF;">{result.objective}</span>
            </div>
            <div style="text-align:center;min-width:90px;">
                <div style="font-family:'Teko',sans-serif;font-size:1.6rem;color:{color};">
                    {result.confidence:.0f}%</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.52rem;color:#666;">
                    CONFIDENCE</div>
            </div>
        </div>
        <div style="margin-top:8px;">
            {"".join(f'<div style="border-left:2px solid #333340;padding:2px 12px;margin:2px 0 2px 6px;'
                     f'font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#CCC;">{f}</div>'
                     for f in result.findings)}
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#666;
                    margin-top:6px;">
            BASIS: {result.confidence_basis} &nbsp;·&nbsp; HANDS DOWNSTREAM: {handoff}
        </div>
        {f'''<div style="background:#0D0D10;border-left:2px solid #00D4FF;padding:6px 12px;
                    margin-top:8px;font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                    color:#AACCDD;white-space:pre-wrap;">🧠 {result.ai_narrative}</div>'''
          if getattr(result, 'ai_narrative', None) else ''}
    </div>""", unsafe_allow_html=True)


def render(shared_context: dict) -> None:
    orch = build_default_orchestrator()
    wf_labels = {
        "planning_chain": "📦 Planning chain — Demand → Inventory → Procurement → Executive",
        "logistics_review": "⛟ Logistics review — Logistics → Supplier Risk → Sustainability → Executive",
        "full_control_tower": "🕸 Full control tower — all specialists → Executive",
    }
    from ai import AI
    _ai_status = AI.status()
    _ai_ready = any(_ai_status.get(c) for c in
                    ("reasoning.operations", "reasoning.executive"))

    c1, c2, c3 = st.columns([3, 1, 1])
    choice = c1.selectbox("Workflow", list(orch.workflows),
                          format_func=lambda k: wf_labels.get(k, k),
                          key="orch_workflow", label_visibility="collapsed")
    ai_on = c2.toggle("🧠 AI reasoning", value=False, key="orch_ai",
                      help=("Each agent adds an LLM narrative via the AI Router "
                            "(capability → model). "
                            + ("Reasoning models configured." if _ai_ready else
                               "No NVIDIA reasoning key set — uses Groq/offline fallback.")))
    go = c3.button("▶ RUN WORKFLOW", key="orch_run", use_container_width=True)
    st.caption("PIPELINE: " + "  →  ".join(
        a.replace("_", " ").upper() for a in orch.workflows[choice])
        + "   ·   AI ROUTER: " + " / ".join(
            f"{k.split('.')[-1]}={'✓' if v else '—'}"
            for k, v in _ai_status.items() if k.startswith("reasoning")))

    if go:
        with st.spinner(f"Running {len(orch.workflows[choice])} agents"
                        f"{' with AI reasoning' if ai_on else ''}..."):
            st.session_state.orch_last_run = orch.run_workflow(
                choice, shared_context, ai_enabled=ai_on)

    run = st.session_state.get("orch_last_run")
    if run is None:
        st.info("Run a workflow — each agent reasons on live data, hands its outputs to "
                "the next, and routes anything material to the Decision Center for your approval.")
        return

    total = len(run.results)
    for i, result in enumerate(run.results, start=1):
        _render_result(result, i, total)

    st.markdown(f"""
    <div style="background:#0D0D10;border:1px solid #00E676;padding:10px 16px;
                font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#CCC;">
        WORKFLOW <b style="color:#00E676;">{run.workflow.upper()}</b> COMPLETE ·
        {total} AGENTS · {run.total_ms:,} MS ·
        <b style="color:#FBC02D;">{run.recommendations_created} NEW RECOMMENDATION(S)</b>
        ROUTED TO THE DECISION CENTER FOR APPROVAL
    </div>""", unsafe_allow_html=True)
    _render_observability()


def _render_observability() -> None:
    """AI platform observability — request log, tokens, latency, cache."""
    import pandas as pd
    from ai import observability
    from ai.router import AI

    with st.expander("🔭 AI PLATFORM OBSERVABILITY — REQUESTS · TOKENS · LATENCY · CACHE"):
        s = observability.stats()
        cache = AI.router.cache.stats()
        if not s.get("requests"):
            st.caption("No AI requests logged yet. Enable 🧠 AI reasoning and run a "
                       "workflow, or ask the Executive Copilot, to populate the log. "
                       "(In this environment NVIDIA is only reachable from your machine.)")
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("AI REQUESTS", f"{s.get('requests', 0):,}")
        o2.metric("TOTAL TOKENS", f"{s.get('total_tokens', 0):,}")
        o3.metric("AVG LATENCY", f"{s.get('avg_latency_ms', 0):.0f} ms")
        o4.metric("SUCCESS RATE",
                  f"{s['success_rate']:.0f}%" if s.get("success_rate") is not None else "—")
        o5.metric("CACHE HIT",
                  f"{cache['hit_rate']:.0f}%" if cache.get("hit_rate") is not None else "—")
        if s.get("by_capability"):
            st.markdown("<div class='hud-label' style='margin:6px 0 2px 0;'>BY CAPABILITY</div>",
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(s["by_capability"]), use_container_width=True,
                         hide_index=True)
        recent = observability.recent(limit=25)
        if recent:
            df = pd.DataFrame(recent)[
                ["ts", "task", "capability", "model", "latency_ms", "total_tokens",
                 "ok", "cached", "fell_back"]]
            st.markdown("<div class='hud-label' style='margin:8px 0 2px 0;'>RECENT AI REQUESTS</div>",
                        unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True, height=220)

    exec_result = next((r for r in run.results if r.agent == "executive"), None)
    if exec_result and exec_result.outputs.get("brief"):
        st.download_button(
            "⇩ EXECUTIVE BRIEF (TXT)",
            data=(f"SUPCHAINMATE EXECUTIVE BRIEF — workflow: {run.workflow}\n"
                  f"{'=' * 48}\n{exec_result.outputs['brief']}\n").encode(),
            file_name="supchainmate_executive_brief.txt", mime="text/plain",
            use_container_width=True,
        )
