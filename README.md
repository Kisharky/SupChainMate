# SupChainMate — Autonomous Supply Chain Decision System

**Built by [Kishan Nagesh](https://www.linkedin.com/in/kisharky-n-5147941a4)**
Master of Business (Supply Chain & International Business) — Monash University, Melbourne

> **Beyond dashboards. Beyond visualisation. A multi-signal AI engine that detects risk, calculates decisions, and generates execution-ready outputs.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Version](https://img.shields.io/badge/version-4.7.0-brightgreen)](CHANGELOG)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green)](https://lightgbm.readthedocs.io)
[![Prophet](https://img.shields.io/badge/Prophet-1.1%2B-blue)](https://facebook.github.io/prophet/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA--3.3--70B-orange)](https://groq.com)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-cuOpt%20%7C%20DeepSeek%20V4-76b900)](https://build.nvidia.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🖥️ Dashboard Preview

![SupChainMate Mission Control](docs/dashboard_preview.png)

*Mission Control HUD — Multi-signal disruption radar, Groq AI auto-insights, and decision engine active*

---

## 🎯 What Is This?

SupChainMate is **not a reporting tool**. It is an autonomous decision layer with **two entry points**:

| Mode | Who it is for | Input |
|------|----------------|-------|
| **Enterprise** | Supply chain teams with data | Upload CSV/Excel (orders, delivery, locations, costs) or try the Olist demo |
| **Small Retailer** | Shops without spreadsheets | A short form per product (weekly sales, lead time, unit cost, safety buffer) — **no file upload** |

Both paths use the **same decision engine** under the hood (`modules/decisions.py`: safety stock, EOQ, reorder point, cost trade-offs). Enterprise adds Prophet, maps, multi-signal risk, Groq, and NVIDIA routing on top of that core.

1. **Detects disruptions** before they cascade (Isolation Forest + LightGBM combined signal) — *Enterprise*
2. **Calculates optimal inventory decisions** (Safety Stock, EOQ, Reorder Point — domain math) — *Enterprise + Small Retailer*
3. **Reasons over your data in real-time** (Groq LLaMA-3.3-70B copilot + auto-insights on load) — *Enterprise*
4. **Optimises routes** with NVIDIA cuOpt (real Capacitated VRP solver) — *Enterprise*
5. **Generates execution-ready outputs** (CSV exports consumable by Power BI, Excel, ERP) — *Enterprise*

```
                    ┌─ Enterprise: CSV/Excel + Prophet + Maps + AI
  Launch screen ────┤
                    └─ Small Retailer: form inputs → DemandProfile (retail helper)
                                    ↓
                    Decision Engine (SS, EOQ, ROP, savings)
                                    ↓
        Enterprise: full HUD + exports    |    Retailer: plain-language alerts + product tracker
```

```
Your Data (CSV / Excel) — Enterprise only
        ↓
AI Analysis Layer
  ├── Demand Sensing     (Prophet + External Regressors)
  ├── Disruption Radar   (Isolation Forest × LightGBM fusion)
  ├── Decision Engine    (Safety Stock, EOQ, ROP, LT Buffer)  ← shared with Small Retailer
  ├── Groq AI            (Auto-Insights + Live Copilot + Executive Narrative)
  └── NVIDIA cuOpt       (Real VRP Route Optimisation)
        ↓
Prescriptive Actions + Execution Plan
        ↓
Export → Power BI / Excel / ERP
```

---

## 🏗️ Architecture

```
logistics-ai-dashboard/
├── app.py                        # Main Streamlit application
├── style.css                     # Mission Control HUD theme (cyberpunk-dark)
├── requirements.txt              # All 11 dependencies
├── .env                          # API keys (gitignored — never committed)
├── modules/
│   ├── forecast.py               # Prophet demand forecasting + external regressors
│   ├── network.py                # Geolocation, KMeans, Isolation Forest,
│   │                             #   Haversine metrics, combined_risk_signal()
│   ├── tracking.py               # LightGBM delay prediction + feature engineering
│   ├── decisions.py              # Decision Engine (SS, EOQ, ROP) + build_demand_profile_from_retail_inputs()
│   ├── retail.py                 # Small Retailer helpers (tier → service level, tracker rows, status)
│   ├── control_tower.py          # Freight Control Tower: shipment board, carrier scorecards,
│   │                             #   on-time KPIs, exception insights
│   ├── agent.py                  # Agentic Copilot: LLaMA-3.3 tool calling + offline router,
│   │                             #   6 tools acting on live data
│   ├── cost_audit.py             # Freight Cost Audit: outliers, duplicates, late-premiums,
│   │                             #   re-tender opportunity
│   ├── health_check.py           # Scored supply chain health check (6 dimensions, DIFOT)
│   ├── tender.py                 # Freight tender / RFP pack + rate-shift simulator
│   ├── alerts.py                 # Alert digests + optional SMTP email delivery
│   ├── store.py                  # SQLite persistence (retail tracker, settings, KPI snapshots)
│   ├── connect.py                # Live store connectors: Shopify Admin API, WooCommerce REST
│   ├── doc_intel.py              # Invoice/BOL scanner: extraction + board reconciliation
│   ├── carbon.py                 # Carbon Lens: CO2e estimates by carrier, zone, mode
│   ├── ingestion.py              # Auto-detect CSV/Excel column mapping
│   ├── groq_ai.py                # Groq: copilot, auto-insights, executive narrative,
│   │                             #   smart column detection (LLaMA-3.3-70B)
│   ├── nvidia_api.py             # NVIDIA: cuOpt VRP optimisation, LLaMA-4 fallback
│   └── optimization.py           # Network KPI summary
└── data/
    └── olist_*.csv               # Auto-downloaded demo data (99k orders)
```

---

## 🤖 AI / ML Pipeline

### 1. Groq AI Layer — LLaMA-3.3-70B (v4.0)

| Feature | What It Does |
|---|---|
| **Auto-Insights** | 3 AI insights surfaced on every load — severity-ranked (HIGH/MEDIUM/LOW), metric-specific |
| **Live Copilot** | Context-aware Q&A with 13 live metrics injected per query (<1s response time) |
| **Executive Narrative** | AI writes the board-ready executive summary paragraph |
| **Smart Column Detection** | LLM maps non-standard CSV columns to internal schema — handles any naming convention |

### 2. NVIDIA cuOpt — Real Route Optimisation

| Property | Detail |
|---|---|
| Model | NVIDIA cuOpt via NIM API |
| Input | Haversine distance matrix from KMeans cluster centroids |
| Fleet | Auto-scaled to `n_clusters / 2` vehicles |
| Output | Optimised route km, savings vs naive sequential, per-vehicle breakdown |
| Fallback | Local greedy estimate if API unavailable (honest labelling) |

### 3. Disruption Radar — Multi-Signal Fusion

Two independent ML models fused into a single `combined_risk` score:

| Signal | Model | What It Detects |
|---|---|---|
| **Spatial Anomaly** | `IsolationForest` (sklearn) | Geographically isolated delivery nodes — thin coverage = higher risk |
| **Delay Probability** | `LGBMClassifier` (LightGBM) | Likelihood of delivery delay from 7 engineered features |
| **Combined Signal** | Weighted fusion (50/50) | High-confidence when both signals simultaneously agree (≥70 each) |

```
combined_risk = 0.5 × IF_score + 0.5 × LGBM_delay_proba

combined_risk ≥ 85  →  ⚡ CRITICAL  (signal agreement — high confidence)
combined_risk ≥ 65  →  ⚠  WARNING
else                →  ✅ SAFE
```

### 4. LightGBM Delay Prediction — 7 Engineered Features

| Feature | Type | Rationale |
|---|---|---|
| `hour`, `day_of_week`, `month` | Calendar | Time-of-day/week delivery patterns |
| `is_weekend`, `is_month_end` | Binary flag | High-risk operational periods |
| `lead_days`, `lead_days_sq` | Continuous | Primary delay driver (non-linear signal) |
| `long_lead` | Binary flag | Orders with >14 day lead times |

### 5. Supply Chain Decision Engine — Domain Mathematics

| Formula | Implementation |
|---|---|
| Safety Stock | `SS = Z × √(μ_LT × σ_d² + μ_d² × σ_LT²)` — combined variance formula |
| EOQ | `Q* = √(2DS/H)` — minimises total annual inventory cost |
| Reorder Point | `ROP = μ_d × μ_LT + SS` |
| Lead Time Buffer | `LTB = Z × σ_LT` |
| Annual Savings | `Cost(current) − Cost(EOQ-based)` |

---

## 📊 Features

### Dual entry (launch screen)
- **Enterprise mode** — existing upload flow, demo data, and Mission Control dashboard.
- **Small Retailer mode** — five questions per product (name, avg weekly sales, supplier lead time, unit cost, safety buffer). Outputs plain-language reorder, order quantity, safety stock, and estimated savings; multi-product tracker with editable current stock; optional advanced ordering/holding costs. Phone/email alerts are reserved for a future release (UI placeholder).

### Upload Flow (Enterprise)
- Upload **CSV or Excel** for Orders, Delivery, Location, or Cost data
- **Auto-detect column names** — regex engine + Groq LLM fallback for ambiguous columns
- **Try Demo Data** — instant load of 99k real Brazilian e-commerce orders
- **Load new data** (sidebar) returns to the Enterprise upload screen without wiping your Small Retailer product list

### Mission Control Dashboard
- Real-time system status bar (breach count, nominal %, override button)
- **Groq Auto-Insights** — 3 AI-generated insights on every dashboard load, severity-coloured
- **Disruption Radar** — carto-darkmatter Mapbox coloured by combined ML risk signal
- **Zone Risk Alerts** — per-cluster consulting-grade narratives with decomposed signal breakdown
- **Decision Engine HUD** — Safety Stock, EOQ, ROP, Annual Savings — dynamically recalculated

### Agentic Copilot (v4.3) — thinks, decides, acts
The copilot no longer just answers — it **executes tools on your live data**:

| Tool | What it does |
|---|---|
| `get_at_risk_shipments` | Lists open shipments flagged AT RISK / LATE, worst ML risk first |
| `get_carrier_scorecard` | Full scorecard or a single carrier's stats by name |
| `draft_carrier_email` | Writes an SLA-review email citing the carrier's real scorecard numbers |
| `generate_reorder_plan` | Produces the EOQ / ROP / safety-stock execution plan (CSV) |
| `exception_summary` | Builds a digest: late + at-risk counts, weak carriers, top risks |
| `freight_cost_audit` | Audits freight charges: outliers, duplicates, late-premiums, re-tender opportunity |
| `supply_chain_health_check` | Scored assessment (0-100, A-F) across 6 dimensions with DIFOT |
| `generate_tender_pack` | Data-backed freight tender: lane volumes, carrier summary, RFP draft |

- **Groq LLaMA-3.3-70B function calling** picks and chains tools, then answers citing tool results
- **Offline mode** — with no API key, a deterministic router still runs every action on live data (LLM adds reasoning + wording, never the numbers)
- **Quick actions** — one-click buttons: Exception digest · At-risk shipments · Email worst carrier · Reorder plan
- Chat history with per-turn "EXECUTED: tool" trace and downloadable artifacts (CSV / TXT)
- **16 live metrics** injected into the agent's system prompt

### Freight Control Tower (v4.2)
- **Shipment Tracking Board** — every shipment classified as ON TRACK / AT RISK / LATE / DELIVERED ON TIME / DELIVERED LATE / CANCELLED, exceptions surfaced first
- **Real on-time performance** — computed from promised vs actual delivery dates when both exist (93.2% on the Olist demo — real dates, not simulated)
- **ML risk flagging** — open shipments in the top decile of LightGBM delay probability are flagged AT RISK
- **Carrier Scorecards** — per-carrier volume, on-time %, late count, avg delay, avg cost/shipment, A–D grade, plus plain-language insights (volume-shift and SLA-review suggestions)
- **Carrier auto-detection** — upload a delivery file with a carrier/courier/transporter/3PL column to unlock scorecards on your own carriers (demo uses fictional carriers, clearly labelled)
- **Exports** — tracking board + carrier scorecard CSVs

### Freight Cost Audit (v4.4)
- **Outlier detection** — charges above the carrier's own Q3 + 1.5×IQR band, with estimated overcharge vs the carrier median
- **Duplicate detection** — same shipment billed twice, or identical (carrier, day, cost) charges repeated
- **Late-premiums** — above-median rates paid on shipments that still arrived late (ammunition for rate negotiations)
- **Re-tender opportunity** — total spend sitting above the network-median rate
- KPI strip + plain-language findings + flagged-charges table, all exportable (CSV/TXT)
- Works on any delivery upload with a cost/freight/charge column (auto-detected)

### What-If Lab (v4.4)
- Stress-test sliders: demand ±%, lead time ±%, lead-time variability ±%, service level
- Live recalculation of safety stock, reorder point, EOQ, order cadence, and total cost — with deltas vs your current baseline

### Supply Chain Health Check (v4.5)
- Scored assessment (0–100, grade A–F) across six dimensions: delivery performance, risk posture, cost discipline, inventory discipline, network efficiency, data quality
- **DIFOT** (delivered-in-full-on-time, approx) headline metric
- Dimensions without data are excluded from the weighting — never guessed
- Priority actions + exportable report

### Freight Tender / RFP Toolkit (v4.5)
- One-click, data-backed tender pack: monthly lane volumes, peak-month capacity callout, incumbent carrier summary
- Ready-to-edit **RFP document** populated with your real volumes, spend, and on-time baseline
- **Rate-shift simulator** — estimate the cost impact of moving X% of one carrier's volume to another's rate

### Alerts & Persistence (v4.5)
- **Alert digests** for both modes: enterprise (exceptions + audit + health) and retail (ORDER NOW / SOON per product)
- **Email delivery** via SMTP settings in `.env` (degrades to download when unset)
- **SQLite persistence** — the Small Retailer tracker and alert emails survive restarts (`data/supchainmate.db`, gitignored)
- **Shopify / WooCommerce** order exports auto-recognised on upload (badge confirms the platform)

### Invoice / BOL Scanner — Document Intelligence (v4.7)
- Upload a freight invoice or BOL (PDF/TXT) — fields are extracted (Groq LLM when configured, regex offline) and **reconciled against the shipment board**
- Checks: carrier known · shipment references resolve · invoice total vs recorded costs (±10% tolerance) or the carrier's audited rate band · already-billed warning
- Verdicts: OK TO PAY / REVIEW — RATE MISMATCH / UNKNOWN SHIPMENTS / UNKNOWN CARRIER
- "Try sample invoice" builds one from real board shipments (with one inflated line) so the scanner is demoable instantly
- Numbers always come from the data — the LLM only parses text

### Carbon Lens (v4.7)
- Freight **CO₂e estimates** from your real route distances: `distance × weight × DEFRA-style mode factor` (road/rail/air/sea)
- Per-carrier footprint with a **greenest-vs-cheapest scatter**, per-zone footprint, network total, adjustable shipment weight
- Route optimisation savings now shown in **tCO₂e avoided**
- Honest by construction: carriers only differ when their transport mode differs (add a `transport_mode` column; demo modes are simulated and labelled)

### Live Store Connect & Performance History (v4.6)
- **Connect your store** — pull order history straight from Shopify (Admin API token, `read_orders` scope) or WooCommerce (read-only REST keys); no CSV needed. Credentials are used for the fetch only and never saved
- **Performance History** — one KPI snapshot (health score, on-time %, late, at-risk) saved per data load; trend chart + table build up across sessions
- **Test suite** — `python -m pytest tests/ -q` covers the control tower, audits, health check, tender, alerts, persistence, agent routing, and connectors (mocked HTTP)

### Marketing Site
- `docs/index.html` — a self-contained dark landing page, ready for GitHub Pages (Settings → Pages → `docs/`)

### Route Optimisation
- **NVIDIA cuOpt** — real Capacitated Vehicle Routing Problem solver
- Haversine distance matrix built from delivery cluster centroids
- Shows total route km, naive baseline, and % distance savings

### Enterprise Reporting Layer
| Export | Contents | Format |
|---|---|---|
| Forecast Data | Date, yhat, lower/upper bounds | CSV (Power BI ready) |
| KPI Summary | All 12 headline metrics | CSV |
| Inventory Plan | All Decision Engine parameters | CSV |
| Zone Risk Table | Per-cluster scores + recommended actions | CSV |
| Execution Plan | Ranked actions with owners + target dates | CSV |
| Executive Report | Full intelligence brief | CSV |

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/Kisharky/SupChainMate.git
cd SupChainMate/logistics-ai-dashboard
pip install -r requirements.txt
```

### API Keys (Optional — all features degrade gracefully without them)
Copy the example and fill in your keys:
```bash
cp logistics-ai-dashboard/.env.example logistics-ai-dashboard/.env
```
```env
GROQ_API_KEY=your_groq_key_here
NVIDIA_CUOPT_API_KEY=your_nvidia_cuopt_key_here
NVIDIA_LLAMA_API_KEY=your_nvidia_llama_key_here
NVIDIA_DEEPSEEK_API_KEY=your_deepseek_key_here
```

### Run
```bash
streamlit run app.py
# or on Windows
python -m streamlit run app.py
```
Open http://localhost:8501

---

## 📦 Data Requirements

Upload any CSV or Excel. The auto-detection engine handles any naming convention:

| Data Type | Required | Key Columns Auto-Detected |
|---|---|---|
| **Orders** | ✅ Required | date/timestamp, quantity/volume |
| **Delivery** | Optional | delivery date, status, lead time |
| **Location** | Optional | lat/lon or zip/postal code |
| **Cost** | Optional | cost/price/fee columns |

**Demo Data**: [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 99k orders, real geolocation.

---

## 🔄 Changelog

### v4.7.0 — Document Intelligence + Carbon Lens
- **NEW**: `modules/doc_intel.py` — invoice/BOL scanner: PDF/TXT extraction (Groq LLM or offline regex), reconciliation against the shipment board and audited rate bands, pay/review verdicts, sample-invoice demo
- **NEW**: `modules/carbon.py` — CO₂e estimates (DEFRA-style mode factors) by carrier and zone, greenest-vs-cheapest scatter, route-savings tCO₂e
- **NEW**: `transport_mode` support in the shipment board (demo modes simulated + labelled)
- **NEW**: pypdf dependency for PDF text extraction; +7 tests (33 total)

### v4.6.0 — Live Store Connect, Performance History, Test Suite
- **NEW**: `modules/connect.py` — Shopify Admin API + WooCommerce REST connectors with pagination, clear credential errors, and no credential persistence
- **NEW**: "Connect your store" panel on the upload screen — API import feeds the same pipeline as CSV
- **NEW**: KPI snapshot history in SQLite + Performance History panel (health score & on-time % trend across sessions)
- **NEW**: `tests/test_modules.py` — 26 tests across 8 modules (connectors tested against mocked HTTP)

### v4.5.0 — Health Check, Tender Toolkit, Alerts, Persistence
- **NEW**: `modules/health_check.py` — 6-dimension scored assessment with DIFOT and priority actions
- **NEW**: `modules/tender.py` — freight tender pack (lane volumes, carrier summary, RFP draft) + rate-shift simulator
- **NEW**: `modules/alerts.py` — enterprise + retail alert digests, optional SMTP email delivery (`SMTP_*` in `.env`)
- **NEW**: `modules/store.py` — SQLite persistence; retail tracker and alert emails survive restarts
- **NEW**: Retail alerts activated (replaces the "coming soon" placeholder) — digest preview, download, send
- **NEW**: Shopify / WooCommerce export detection badge on upload
- **NEW**: 2 more agent tools (health check, tender pack) → 8 total, quick actions in two rows
- **NEW**: `docs/index.html` marketing landing page (GitHub Pages ready)

### v4.4.0 — Freight Cost Audit + What-If Lab
- **NEW**: `modules/cost_audit.py` — deterministic billing checks: per-carrier IQR outliers, potential duplicates, late-delivery premiums, re-tender opportunity vs network-median rate
- **NEW**: Cost Audit panel in the Control Tower — KPI strip, findings, flagged-charges table, per-carrier cost profile, CSV/TXT exports
- **NEW**: What-If Lab — demand / lead-time / variability / service-level sliders with live decision-engine recalculation and deltas vs baseline
- **NEW**: 6th agent tool `freight_cost_audit` + "⚖ Cost audit" quick action
- **NEW**: Freight-cost column auto-detection in delivery uploads
- **DEMO**: ~0.4% simulated billing errors injected into demo costs so the outlier detector has realistic anomalies (labelled in UI)

### v4.3.0 — Agentic Copilot
- **NEW**: `modules/agent.py` — tool-calling agent loop (Groq LLaMA-3.3-70B function calling, max 4 turns)
- **NEW**: 5 tools acting on live data: at-risk shipments, carrier scorecards, SLA-review email drafts, reorder plans, exception digests
- **NEW**: Offline deterministic router — every action works with zero API keys; numbers always come from the dataframes, never the LLM
- **NEW**: Copilot UI rebuilt: quick-action buttons, chat history, executed-tool trace, downloadable artifacts
- **UPGRADE**: Agent context now includes Control Tower KPIs (on-time %, at-risk, late)

### v4.2.0 — Freight Control Tower
- **NEW**: `modules/control_tower.py` — shipment board, carrier scorecards, on-time KPIs
- **NEW**: Shipment Tracking Board — per-shipment health (ON TRACK / AT RISK / LATE / DELIVERED LATE), exceptions-first sorting, CSV export
- **NEW**: Carrier Scorecards — on-time %, late count, avg delay, cost/shipment, A–D grades, on-time bar chart, plain-language insights
- **NEW**: Control Tower KPI strip — total, in transit, on-time % (real promised-vs-actual dates), late, ML at-risk
- **NEW**: Carrier column auto-detection in delivery uploads (`carrier|courier|transporter|3PL|LSP`); promised-date detection (`estimated|promised|expected|due|eta|sla`)
- **DEMO**: Fictional carriers + freight costs simulated over real Olist delivery dates (labelled in UI)

### v4.1.0 — Dual entry: Enterprise + Small Retailer
- **NEW**: Launch screen — choose **Enterprise** or **Small Retailer** mode
- **NEW**: `modules/retail.py` — retail form helpers, inventory status (ORDER NOW / SOON / OK), tracker rows
- **NEW**: `decisions.build_demand_profile_from_retail_inputs()` — builds `DemandProfile` from weekly sales + lead time + safety tier (no Prophet)
- **NEW**: Small Retailer UI — add products, instant guidance from `run_decision_engine`, multi-product table with **Apply stock levels**
- **UX**: Sidebar reset preserves `retail_products` and returns to Enterprise upload (`entry_mode` + `data_loaded` handling)

### v4.0.0 — Groq AI + NVIDIA API Integration
- **NEW**: `modules/groq_ai.py` — 4 Groq-powered features (LLaMA-3.3-70B)
- **NEW**: Auto-Insights — 3 severity-ranked AI insights generated fresh on every load
- **NEW**: Groq Supply Chain Copilot with 13 live metrics in context (<1s response)
- **NEW**: Smart column detection via Groq LLM (handles non-standard CSV naming)
- **NEW**: `modules/nvidia_api.py` — NVIDIA cuOpt VRP solver + LLaMA-4-Scout fallback
- **NEW**: cuOpt "Execute Optimization" — real Haversine matrix, fleet routing, km savings
- **UPGRADE**: Copilot shows 🟢 LIVE / 🟡 OFFLINE status with automatic fallback chain
- **NEW**: `.env` support for API keys (gitignored, never committed)
- **NEW**: `docs/dashboard_preview.png` — screenshot in README

### v3.0.0 — Enterprise Intelligence Layer
- **NEW**: 6 structured CSV exports (Power BI / Excel ready)
- **NEW**: Executive Report auto-generation
- **NEW**: Zone Risk Intelligence Table + Inventory Decision Table

### v2.5.0 — Multi-Signal Risk Engine
- **NEW**: `combined_risk_signal()` — Isolation Forest × LightGBM fusion
- **NEW**: Signal agreement detection + per-zone consulting alerts
- **UPGRADE**: Map coloured by `combined_level` (not random scores)

### v2.4.0 — ML Model Upgrades
- **UPGRADE**: LightGBM replaces RandomForest, 7 engineered features
- **UPGRADE**: `IsolationForest.decision_function()` normalised to 0–100
- **NEW**: Graceful model fallbacks throughout

### v2.3.0 — Supply Chain Decision Engine
- **NEW**: Safety Stock (combined variance formula), EOQ, ROP, Lead Time Buffer
- **NEW**: Service level Z-score table (80–99.9%), sidebar parameter controls

### v2.2.0 — Geolocation & Network Intelligence
- **UPGRADE**: Real Olist geolocation join, Haversine centroid metrics, n_clusters slider

### v2.1.0 — User Upload Flow
- **NEW**: CSV/Excel uploader with auto column detection (`modules/ingestion.py`)

### v2.0.0 — Mission Control HUD
- **REDESIGN**: Single-page dark HUD, carto-darkmatter map, system status bar

### v1.0.0 — Initial Dashboard
- Prophet, KMeans, RandomForest — basic Streamlit tab layout

---

## 🧠 Design Philosophy

> **"While tools like Power BI focus on visualisation, SupChainMate acts as a decision intelligence layer — generating prescriptive actions from multi-signal AI and exporting execution-ready plans directly into enterprise workflows."**

| Dimension | Power BI | SupChainMate |
|---|---|---|
| Purpose | Visualisation | Decision-making |
| Output | Charts | Execution plans + CSV exports |
| AI / ML | None | Groq LLaMA-3.3 + LightGBM + Prophet + Isolation Forest |
| Route Optimisation | None | NVIDIA cuOpt (real VRP) |
| Integration | Data in | Data in + decisions out → Power BI / ERP |

---

## 📄 License

MIT © Kishan Nagesh — SupChainMate
