# Changelog

All notable changes to SupChainMate are documented here.

## Portfolio Release v1.0 — Enterprise readiness & final polish
- **Orchestration & memory**: added the **Planner** (objective → dynamic
  capability discovery → execution DAG → merged Decision; no business logic) and
  the offline, model-agnostic **Decision Brain** (long-term memory + semantic
  recall), both framework-free and mirroring the AI Router's ports/registry/facade shape
- **Optimization layer**: pluggable solvers beneath the domain agents
  (transportation/VRP, multi-DC allocation) via a problem-kind → solver router
  with a local fallback
- **Enterprise**: JWT auth (rotating refresh tokens, PBKDF2 hashing) with a
  six-role RBAC gate; SQLAlchemy identity layer over **SQLite or Postgres**
  (`DATABASE_URL`); one-command **Docker Compose**; **GitHub Actions CI**
  (backend tests · frontend lint+build); `.env.example` with no secrets in source
- **Polish (behavior-preserving)**: removed superseded endpoints and orphaned
  service/frontend code; migrated FastAPI startup to `lifespan`; added reusable
  `TableState` / `EmptyState` / `Skeleton` design-system primitives and applied
  consistent loading/error states across screens
- **Docs**: guided demo walkthrough (`docs/DEMO.md`) and an eight-dimension
  enterprise readiness audit (`docs/AUDIT.md`)
- Tests: 165 passing across AI router, optimization, planner, brain, auth/RBAC,
  and domain services

## v5.5.0 — Production Hardening: Observability, Caching, Enterprise RAG
- **RAG**: new `ai/rag.py` enterprise pipeline — intelligent overlap chunking, embedding generation, **cached vector index** (chunk embeddings persisted by content hash; only new chunks are embedded), hybrid semantic+lexical ranking, citation generation, and retrieval caching (TTL, mode-aware key). `modules/knowledge.py` is now a thin adapter
- **Observability**: `ai/observability.py` records every AI request — timestamp, capability, model, provider, latency, tokens, cached/fallback flags, errors — to SQLite; router uses it as the observer. New AI Platform Observability panel (requests, tokens, avg latency, success/cache/fallback rates, by-capability, recent log)
- **Performance**: response cache (`ai/cache.py`, LRU+TTL, keyed by capability+prompt+params) in the router; token usage captured from provider `usage`; pooled clients reused per key; async provider path (`achat`) + `router.aask()` for non-Streamlit deployment
- **Agents**: new **Knowledge Agent** (grounds decisions in policies/SOPs via RAG); Executive agent coordinates all specialists (depends on every one, runs last); `full_control_tower` now runs 9 agents. Every agent requests capabilities, never models (guardrail test enforces it)
- **Quality**: shared payload/response shaping in the NVIDIA provider (no sync/async duplication); fixed min-max normalization zeroing a single-candidate RAG hit; conftest isolates the DB so no test touches the real store
- Tests: +3 net (139 total) — cache hits, token/observability capture, KnowledgeAgent coverage, hybrid retriever selection

## v5.4.0 — Provider-Agnostic AI Architecture (NVIDIA NIM)
- **NEW**: `ai/` package — capability-routed AI layer. `router.py` (`AI.ask(capability, task, context)`, the sole capability→model resolver, fallback chain, audit sink), `registry.py` (capability→ModelSpec plan), `providers/nvidia.py` (one cached OpenAI-compatible client per key, own retry policy, 30s timeout, non-raising), `types.py` (Capability/ModelSpec/AIResponse dataclasses)
- **NEW**: capability services — `reasoning` (operations/executive), `embeddings`, `coding`, `safety`, `ocr`, `vision`, `memory`
- **NEW**: model plan via NVIDIA NIM — embedding→nemotron-3-embed-1b, reasoning.executive→nemotron-3-ultra-550b-a55b, reasoning.operations→z-ai/glm-5.2, coding→deepseek-v4-flash (vision/ocr/safety declared for wiring)
- **REFACTOR**: agents call `AI.ask()` via the base class (opt-in AI-reasoning toggle in the orchestrator); no agent references a model name. RAG answers + embedding retrieval route through the AI layer with lexical/Groq fallback
- **QUALITY**: dependency injection (registry/providers/offline handler), type hints throughout, retries + timeouts, graceful fallback, every AI call audit-logged
- Requirements: openai; tests: +17 (136 total) — registry, router fallback, provider retries/caching (mocked), agent narrative routing, RAG retriever selection

## v5.3.0 — Geo Stack: Leaflet · MapTiler · Routing · Weather
- **NEW**: `modules/geo.py` — vendor-neutral geo adapters: MapTiler tile URLs, Nominatim geocoding (keyless, SQLite-cached, rate-polite), road matrices (HERE Matrix API with key → OSRM public server keyless), current weather (OpenWeatherMap with key → Open-Meteo keyless) with delivery-risk notes
- **NEW**: `views/map_view.py` — Leaflet (folium) disruption radar with MapTiler dark tiles, risk-colored markers, hub pins; plotly mapbox fallback when unavailable
- **NEW**: Geo Services panel — geocoder with nearest-hub distance, hub-to-hub road matrix with detour factors and drive times, zone weather watch
- **NEW**: env keys `MAPTILER_API_KEY`, `HERE_API_KEY`, `OPENWEATHER_API_KEY` (all optional, keyless fallbacks)
- Requirements: folium, streamlit-folium (pinned); tests: +7 (119 total)

## v5.2.0 — Auto Carrier Allocation + Dispute Manager
- **NEW**: `modules/allocation.py` — multi-criteria carrier allocation: cost/service/emissions/reliability scoring under user weights, score-proportional shares with a 50% concentration cap, blended-mix impact (cost, on-time, CO₂e), Decision Center proposal builder
- **NEW**: `modules/disputes.py` — dispute lifecycle: raise from audit-flagged charges (deduped), validated OPEN → SENT → RESOLVED / WRITTEN_OFF transitions, recovered amounts, recovery-rate KPIs, audit events
- **NEW**: Carrier Allocation panel (weight sliders, score table, current-vs-recommended chart) + Dispute Manager panel (KPIs, table, transition controls, CSV export)
- Tests: +5 (112 total)

## v5.1.0 — Memory, Events, RAG, Executive Copilot, ERP Connectors
- **NEW**: Agent memory — every orchestrator agent run persists its outputs (`agent_runs`); the Executive reports run-over-run deltas; `business_deltas` copilot tool answers "what changed?"
- **NEW**: `modules/events.py` — event-driven automation: supplier-delay, inventory-below-threshold, demand-spike, and at-risk-surge detectors auto-run the mapped workflows on data load with full audit chain
- **NEW**: `modules/knowledge.py` — RAG knowledge base: upload SOPs/policies/contracts (PDF/TXT), TF-IDF + char-n-gram retrieval (offline), Groq-composed cited answers when configured
- **NEW**: Executive Copilot — 6th worker with `get_pending_decisions` ("what should I approve today?"), `business_deltas`, `ask_knowledge_base` (13 tools total)
- **NEW**: ERPNext connector (token auth, Sales Orders) + generic REST adapter (endpoint + records path + field mapping) for SAP/Oracle/D365 gateways
- Tests: +14 (107 total)

## v5.0.0 — Agent Orchestrator: Eight Domain Agents
- **NEW**: `modules/agents/` — multi-agent layer: `base.py` (typed contracts, ScopedContext enforcing declared-only data access, execution template with timing/containment), `domain.py` (8 single-responsibility agents wrapping existing engines), `orchestrator.py` (validated workflows, context passing, Decision Center routing, audit logging)
- **NEW**: Agents: Demand Forecast, Inventory, Procurement, Logistics, Supplier Risk (HHI concentration + reliability variance — new analysis), Warehouse, Sustainability, Executive (chained confidence bounded by the weakest upstream agent)
- **NEW**: Built-in workflows: planning_chain, logistics_review, full_control_tower — inter-agent communication (Demand → Inventory → Procurement → Executive)
- **NEW**: `views/agents_hub.py` — run workflows, per-agent reasoning cards, downstream handoffs, executive brief export
- Tests: +13 (93 total) — scoping enforcement, per-agent runs, dependency validation, full-pipeline context passing, audit verification

## v4.13.0 — Decision Center: the Trust Layer
- **NEW**: `modules/trust.py` — typed Recommendation records with WHY drivers (evidence-backed), transparent confidence scores (data support + signal strength, with stated basis), and quantified impact (savings $/yr, stockout risk %, service level %)
- **NEW**: Builders wrap the deterministic engines — inventory policy (Planner), urgent SKU reorders, carrier volume shifts (with rate-simulator savings), billing disputes
- **NEW**: `views/decision_center.py` — pending-recommendation cards with Approve / Reject / Modify (with note), decision history, and an immutable audit trail (all CSV-exportable)
- **NEW**: SQLite `recommendations` + `audit_log` tables with dedupe-by-key and full event logging
- Tests: +7 (80 total)

## v4.12.0 — Engineering Hardening
- **NEW**: `config.py` — central paths, .env lookup (de-duplicated from 3 modules), model IDs, API endpoints, tunable thresholds, logging setup
- **REFACTOR**: `app.py` reduced 2,438 → 1,872 lines; landing/retail/upload pages, data pipeline, and shared render helpers extracted into a `views/` package
- **NEW**: `tests/test_core.py` — 16 tests for the decision engine (formula-exact + monotonicity), forecasting, optimisation, and network scoring (73 total)
- **QUALITY**: logging replaces print; SMTP/Groq/SQLite failures logged; demo-data and orders-file loads fail gracefully with clear messages
- **BUILD**: requirements pinned to the exact tested versions (incl. pytest)

## v4.11.0 — SKU Intelligence: Per-Product Decisions
- **NEW**: `modules/sku.py` — per-SKU demand profiling, ABC classification (revenue Pareto), differentiated service levels by class (A = target, B −3 pts, C −8 pts), per-SKU safety stock / ROP / EOQ via the shared decision engine
- **NEW**: SKU Intelligence section — KPI strip, editable per-SKU decision table with ORDER NOW / SOON / OK status, ABC Pareto chart, CSV export
- **NEW**: SKU/product + unit-price column auto-detection in orders uploads
- **NEW**: Planner's reorder tool now attaches the per-SKU plan
- **DEMO**: simulated 12-SKU catalogue over real Olist order dates (labelled)
- Tests: +4 (57 total)

## v4.10.0 — Autonomous Workforce: Runbook + Status Board
- **NEW**: `modules/runbook.py` — plain-English standing rules ("flag any shipment over $50", "alert me when SwiftLine on-time drops below 95%") parsed deterministically, auto-assigned to the right AI Worker, persisted in SQLite, evaluated on every data load
- **NEW**: `agent.autonomous_sweep()` — background monitoring; every worker reports live status without being asked
- **NEW**: Autonomous Workforce section — 5 worker status cards with green/yellow/red indicators and RULE FIRED badges, plus the Runbook management panel
- **NEW**: Triggered runbook rules included in the enterprise alert digest
- Tests: +10 (53 total)

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

