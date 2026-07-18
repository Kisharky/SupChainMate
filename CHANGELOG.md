# Changelog

All notable changes to SupChainMate are documented here.

## v4.9.0 — Market Signals: External Factor Engine
- **NEW**: `modules/factors.py` — keyless factor sources: FX (frankfurter/ECB), Brent crude (Stooq), weather (Open-Meteo), offline holiday calendar (`holidays`), PostHog/GA daily-events CSV import
- **NEW**: Market Signals panel — Bloomberg-style ticker strip, factor↔demand correlations (same-day + 7-day leading), factor frame export
- **NEW**: Factor-aware model tournament — factors join the feature set and their uplift is measured on the same 28-day holdout (baseline vs factor-aware champion MAPE)
- Every online source degrades gracefully; offline calendar factors always available
- Tests: +5 (43 total); requirements: `holidays`

## v4.8.0 — AI Workers, Reasoning Trace, Model Tournament
- **NEW**: AI Workers roster — 5 named workers (Tracker, Auditor, Carrier Manager, Procurement, Planner) over the existing tool registry, with per-worker action buttons and reply attribution
- **NEW**: Reasoning trace on every agent turn — routing, LLM turns, tool calls with args, per-step timings
- **NEW**: `modules/ensemble.py` — model tournament (LightGBM/RF/GBM/Ridge + ensemble mean) backtested vs Prophet on a 28-day holdout; champion forecast export; tail-off trimming
- Tests: +5 (38 total)

## v4.7.0 — Document Intelligence + Carbon Lens
- **NEW**: `modules/doc_intel.py` — invoice/BOL scanner: PDF/TXT extraction (Groq LLM or offline regex), reconciliation against the shipment board and audited rate bands, pay/review verdicts, sample-invoice demo
- **NEW**: `modules/carbon.py` — CO₂e estimates (DEFRA-style mode factors) by carrier and zone, greenest-vs-cheapest scatter, route-savings tCO₂e
- **NEW**: `transport_mode` support in the shipment board (demo modes simulated + labelled)
- **NEW**: pypdf dependency for PDF text extraction; +7 tests (33 total)

## v4.6.0 — Live Store Connect, Performance History, Test Suite
- **NEW**: `modules/connect.py` — Shopify Admin API + WooCommerce REST connectors with pagination, clear credential errors, and no credential persistence
- **NEW**: "Connect your store" panel on the upload screen — API import feeds the same pipeline as CSV
- **NEW**: KPI snapshot history in SQLite + Performance History panel (health score & on-time % trend across sessions)
- **NEW**: `tests/test_modules.py` — 26 tests across 8 modules (connectors tested against mocked HTTP)

## v4.5.0 — Health Check, Tender Toolkit, Alerts, Persistence
- **NEW**: `modules/health_check.py` — 6-dimension scored assessment with DIFOT and priority actions
- **NEW**: `modules/tender.py` — freight tender pack (lane volumes, carrier summary, RFP draft) + rate-shift simulator
- **NEW**: `modules/alerts.py` — enterprise + retail alert digests, optional SMTP email delivery (`SMTP_*` in `.env`)
- **NEW**: `modules/store.py` — SQLite persistence; retail tracker and alert emails survive restarts
- **NEW**: Retail alerts activated (replaces the "coming soon" placeholder) — digest preview, download, send
- **NEW**: Shopify / WooCommerce export detection badge on upload
- **NEW**: 2 more agent tools (health check, tender pack) → 8 total, quick actions in two rows
- **NEW**: `docs/index.html` marketing landing page (GitHub Pages ready)

## v4.4.0 — Freight Cost Audit + What-If Lab
- **NEW**: `modules/cost_audit.py` — deterministic billing checks: per-carrier IQR outliers, potential duplicates, late-delivery premiums, re-tender opportunity vs network-median rate
- **NEW**: Cost Audit panel in the Control Tower — KPI strip, findings, flagged-charges table, per-carrier cost profile, CSV/TXT exports
- **NEW**: What-If Lab — demand / lead-time / variability / service-level sliders with live decision-engine recalculation and deltas vs baseline
- **NEW**: 6th agent tool `freight_cost_audit` + "⚖ Cost audit" quick action
- **NEW**: Freight-cost column auto-detection in delivery uploads
- **DEMO**: ~0.4% simulated billing errors injected into demo costs so the outlier detector has realistic anomalies (labelled in UI)

## v4.3.0 — Agentic Copilot
- **NEW**: `modules/agent.py` — tool-calling agent loop (Groq LLaMA-3.3-70B function calling, max 4 turns)
- **NEW**: 5 tools acting on live data: at-risk shipments, carrier scorecards, SLA-review email drafts, reorder plans, exception digests
- **NEW**: Offline deterministic router — every action works with zero API keys; numbers always come from the dataframes, never the LLM
- **NEW**: Copilot UI rebuilt: quick-action buttons, chat history, executed-tool trace, downloadable artifacts
- **UPGRADE**: Agent context now includes Control Tower KPIs (on-time %, at-risk, late)

## v4.2.0 — Freight Control Tower
- **NEW**: `modules/control_tower.py` — shipment board, carrier scorecards, on-time KPIs
- **NEW**: Shipment Tracking Board — per-shipment health (ON TRACK / AT RISK / LATE / DELIVERED LATE), exceptions-first sorting, CSV export
- **NEW**: Carrier Scorecards — on-time %, late count, avg delay, cost/shipment, A–D grades, on-time bar chart, plain-language insights
- **NEW**: Control Tower KPI strip — total, in transit, on-time % (real promised-vs-actual dates), late, ML at-risk
- **NEW**: Carrier column auto-detection in delivery uploads (`carrier|courier|transporter|3PL|LSP`); promised-date detection (`estimated|promised|expected|due|eta|sla`)
- **DEMO**: Fictional carriers + freight costs simulated over real Olist delivery dates (labelled in UI)

## v4.1.0 — Dual entry: Enterprise + Small Retailer
- **NEW**: Launch screen — choose **Enterprise** or **Small Retailer** mode
- **NEW**: `modules/retail.py` — retail form helpers, inventory status (ORDER NOW / SOON / OK), tracker rows
- **NEW**: `decisions.build_demand_profile_from_retail_inputs()` — builds `DemandProfile` from weekly sales + lead time + safety tier (no Prophet)
- **NEW**: Small Retailer UI — add products, instant guidance from `run_decision_engine`, multi-product table with **Apply stock levels**
- **UX**: Sidebar reset preserves `retail_products` and returns to Enterprise upload (`entry_mode` + `data_loaded` handling)

## v4.0.0 — Groq AI + NVIDIA API Integration
- **NEW**: `modules/groq_ai.py` — 4 Groq-powered features (LLaMA-3.3-70B)
- **NEW**: Auto-Insights — 3 severity-ranked AI insights generated fresh on every load
- **NEW**: Groq Supply Chain Copilot with 13 live metrics in context (<1s response)
- **NEW**: Smart column detection via Groq LLM (handles non-standard CSV naming)
- **NEW**: `modules/nvidia_api.py` — NVIDIA cuOpt VRP solver + LLaMA-4-Scout fallback
- **NEW**: cuOpt "Execute Optimization" — real Haversine matrix, fleet routing, km savings
- **UPGRADE**: Copilot shows 🟢 LIVE / 🟡 OFFLINE status with automatic fallback chain
- **NEW**: `.env` support for API keys (gitignored, never committed)
- **NEW**: `docs/dashboard_preview.png` — screenshot in README

## v3.0.0 — Enterprise Intelligence Layer
- **NEW**: 6 structured CSV exports (Power BI / Excel ready)
- **NEW**: Executive Report auto-generation
- **NEW**: Zone Risk Intelligence Table + Inventory Decision Table

## v2.5.0 — Multi-Signal Risk Engine
- **NEW**: `combined_risk_signal()` — Isolation Forest × LightGBM fusion
- **NEW**: Signal agreement detection + per-zone consulting alerts
- **UPGRADE**: Map coloured by `combined_level` (not random scores)

## v2.4.0 — ML Model Upgrades
- **UPGRADE**: LightGBM replaces RandomForest, 7 engineered features
- **UPGRADE**: `IsolationForest.decision_function()` normalised to 0–100
- **NEW**: Graceful model fallbacks throughout

## v2.3.0 — Supply Chain Decision Engine
- **NEW**: Safety Stock (combined variance formula), EOQ, ROP, Lead Time Buffer
- **NEW**: Service level Z-score table (80–99.9%), sidebar parameter controls

## v2.2.0 — Geolocation & Network Intelligence
- **UPGRADE**: Real Olist geolocation join, Haversine centroid metrics, n_clusters slider

## v2.1.0 — User Upload Flow
- **NEW**: CSV/Excel uploader with auto column detection (`modules/ingestion.py`)

## v2.0.0 — Mission Control HUD
- **REDESIGN**: Single-page dark HUD, carto-darkmatter map, system status bar

## v1.0.0 — Initial Dashboard
- Prophet, KMeans, RandomForest — basic Streamlit tab layout

