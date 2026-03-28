# SupChainMate — Autonomous Supply Chain Decision System

> **Beyond dashboards. Beyond visualisation. A multi-signal AI engine that detects risk, calculates decisions, and generates execution-ready outputs.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green)](https://lightgbm.readthedocs.io)
[![Prophet](https://img.shields.io/badge/Prophet-1.1%2B-blue)](https://facebook.github.io/prophet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What Is This?

SupChainMate is **not a reporting tool**. It is an autonomous decision layer that sits on top of your supply chain data and:

1. **Detects disruptions** before they cascade (Isolation Forest + LightGBM combined signal)
2. **Calculates optimal inventory decisions** (Safety Stock, EOQ, Reorder Point — pure domain mathematics)
3. **Generates execution-ready outputs** (CSV exports, Executive Reports) directly consumable by Power BI, Excel, or any ERP

```
Your Data (CSV / Excel)
        ↓
AI Analysis Layer
  ├── Demand Sensing (Prophet + External Regressors)
  ├── Disruption Radar (Isolation Forest + LightGBM fusion)
  └── Decision Engine (Safety Stock, EOQ, ROP, LT Buffer)
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
├── style.css                     # Mission Control HUD theme
├── requirements.txt              # All dependencies
├── modules/
│   ├── forecast.py               # Prophet demand forecasting + external regressors
│   ├── network.py                # Geolocation, KMeans clustering, Isolation Forest,
│   │                             #   Haversine centroid metrics, combined_risk_signal()
│   ├── tracking.py               # LightGBM delay prediction + feature engineering
│   ├── optimization.py           # Network KPI summary
│   ├── decisions.py              # Supply Chain Decision Engine (SS, EOQ, ROP)
│   └── ingestion.py              # Auto-detect CSV/Excel column mapping
└── data/
    ├── olist_orders.csv
    ├── olist_orders_dataset.csv
    └── olist_customers_dataset.csv
```

---

## 🤖 AI / ML Pipeline

### 1. Demand Sensing — Meta Prophet

| Property | Detail |
|---|---|
| Model | `Prophet` with external regressor |
| Signal | Synthetic macro event toggle (weather, market shocks) |
| Output | 7–30 day forecast with confidence intervals |
| Purpose | Demand-driven safety stock + EOQ recalculation |

### 2. Disruption Radar — Multi-Signal Fusion

This is the system's core competitive advantage. Two independent ML signals are fused into a single `combined_risk` score:

| Signal | Model | What It Detects |
|---|---|---|
| **Spatial Anomaly** | `IsolationForest` (sklearn) | Geographically isolated delivery nodes with thin coverage |
| **Delay Probability** | `LGBMClassifier` (LightGBM) | Likelihood of delivery delay per cluster |
| **Combined Signal** | Weighted fusion (50/50) | High-confidence risk when both signals agree (≥70 each) |

```
combined_risk = 0.5 × IF_score + 0.5 × LGBM_delay_proba

combined_risk ≥ 85 → ⚡ CRITICAL  (both signals agree — high confidence)
combined_risk ≥ 65 → ⚠ WARNING
else               → ✅ SAFE
```

### 3. LightGBM Delay Prediction

**Engineered features** (7 inputs vs the original 2):

| Feature | Type | Rationale |
|---|---|---|
| `hour`, `day_of_week`, `month` | Calendar | Time-of-day/week delivery patterns |
| `is_weekend`, `is_month_end` | Binary flag | High-risk operational periods |
| `lead_days`, `lead_days_sq` | Continuous | Primary driver of delays (non-linear) |
| `long_lead` | Binary flag | Orders with >14 day lead times |

### 4. Supply Chain Decision Engine (Domain Mathematics)

Unlike generic AI, this module implements **textbook supply chain formulas**:

| Formula | Implementation |
|---|---|
| Safety Stock | `SS = Z × √(μ_LT × σ_d² + μ_d² × σ_LT²)` — combined variance |
| EOQ | `Q* = √(2DS/H)` — minimises total annual inventory cost |
| Reorder Point | `ROP = μ_d × μ_LT + SS` |
| Lead Time Buffer | `LTB = Z × σ_LT` |
| Annual Savings | `Cost(current) − Cost(EOQ-based)` |

---

## 📊 Features

### Upload Flow
- Upload **CSV or Excel** for Orders, Delivery, Location, Cost data
- **Auto-detect column names** — handles any naming convention via regex pattern matching
- **Try Demo Data** button for instant demonstration

### Mission Control Dashboard
- Real-time system status bar with breach count
- 2-column HUD layout (Map + Metrics)
- **Disruption Radar**: carto-darkmatter Mapbox with combined risk colouring
- **Zone Risk Alerts**: per-cluster consulting-level intelligence narratives

### Decision Engine
- Safety Stock, EOQ, Reorder Point, Lead Time Buffer — all dynamically recalculated
- Responds to sidebar parameter changes (service level, unit cost, lead time)

### What-If Simulation
- **Demand surge**: slider from -50% to +50%, immediate impact on all metrics
- **Network hubs**: 2–12 cluster slider — instantly reconfigures the map and Haversine topology

### Enterprise Reporting Layer
| Export | Contents | Format |
|---|---|---|
| Forecast Data | Date, yhat, lower/upper bounds | CSV (Power BI ready) |
| KPI Summary | All headline metrics | CSV |
| Inventory Plan | All Decision Engine parameters | CSV |
| Zone Risk Table | Per-cluster risk scores + actions | CSV |
| Execution Plan | Ranked actions with owners + dates | CSV |
| Executive Report | Full intelligence brief | CSV |

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
```

### Installation
```bash
git clone https://github.com/Kisharky/SupChainMate.git
cd SupChainMate/logistics-ai-dashboard
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
# or
python -m streamlit run app.py
```

Open http://localhost:8501

---

## 📦 Data Requirements

### Your Own Data
Upload any CSV or Excel file. The auto-detection engine recognises common column patterns:

| Data Type | Required | Key Columns Auto-Detected |
|---|---|---|
| **Orders** | ✅ Required | date/timestamp, quantity/volume |
| **Delivery** | Optional | delivery date, status, lead time |
| **Location** | Optional | lat/lon or zip/postal code |
| **Cost** | Optional | cost/price/fee columns |

### Demo Data
Uses the public [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (99k orders, real geolocation).

---

## 🔄 Changelog

### v3.0.0 — Enterprise Intelligence Layer
- **NEW**: Enterprise Reporting Layer — 6 structured CSV exports (Power BI / Excel ready)
- **NEW**: Executive Report auto-generation (risk summary + recommendations + financial impact)
- **NEW**: Zone Risk Intelligence Table with per-cluster actions
- **NEW**: Inventory Decision Table (all Decision Engine parameters in tabular format)

### v2.5.0 — Multi-Signal Risk Engine
- **NEW**: `combined_risk_signal()` — fuses Isolation Forest + LightGBM into single score
- **NEW**: Signal agreement detection — high-confidence alerts when both models agree
- **NEW**: Per-zone consulting alerts with decomposed signal breakdown
- **UPGRADE**: Map now colours by `combined_level` (not random scores)

### v2.4.0 — ML Model Upgrades
- **UPGRADE**: `LGBMClassifier` replaces `RandomForestClassifier` for delay prediction
- **NEW**: 7 engineered features vs original 2 (lead days, weekend flags, month-end)
- **NEW**: `predict_proba()` output for probabilistic delay risk (not binary)
- **UPGRADE**: `IsolationForest` with proper `decision_function()` normalised to 0–100
- **NEW**: Graceful fallback to RandomForest if LightGBM not installed
- **NEW**: `requirements.txt` with all pinned dependencies

### v2.3.0 — Supply Chain Decision Engine
- **NEW**: `modules/decisions.py` — Safety Stock (combined variance formula), EOQ, ROP, LTB
- **NEW**: Service level Z-score table (80%–99.9%)
- **NEW**: Sidebar parameter controls (unit cost, ordering cost, holding rate, service level)
- **NEW**: Prescriptive recommendations with impact classification

### v2.2.0 — Geolocation & Network Intelligence
- **UPGRADE**: Real Olist geolocation join (median lat/lon per zip prefix) replaces zip % 90 proxy
- **NEW**: Haversine centroid distance metrics per cluster (avg, max, efficiency score)
- **NEW**: `n_clusters` what-if slider — reconfigures network topology instantly
- **NEW**: Auto-downloads geolocation dataset if not present locally

### v2.1.0 — User Upload Flow
- **NEW**: CSV / Excel file uploader for 4 data types (Orders, Delivery, Location, Cost)
- **NEW**: Auto-column detection via regex pattern matching
- **NEW**: `modules/ingestion.py` — normalises any column naming convention
- **NEW**: "Try Demo Data" button loads Olist dataset instantly
- **CHANGE**: App requires data upload before dashboard renders (proper product flow)

### v2.0.0 — Mission Control HUD
- **REDESIGN**: Single-page Mission Control layout (no tabs)
- **NEW**: `carto-darkmatter` Mapbox with neon risk colouring
- **NEW**: System status bar (breach count, nominal %, override button)
- **NEW**: AI Control Tower with prescriptive action buttons + Supply Chain Copilot
- **NEW**: Demand Surge Simulator bottom bar

### v1.0.0 — Initial Dashboard
- Prophet demand forecasting
- KMeans network clustering
- RandomForest delay prediction
- Basic Streamlit tab layout

---

## 🧠 Design Philosophy

> **"While tools like Power BI focus on visualisation, SupChainMate acts as a decision intelligence layer — generating prescriptive actions from multi-signal AI and exporting execution-ready plans directly into enterprise workflows."**

| Dimension | Power BI | SupChainMate |
|---|---|---|
| Purpose | Visualisation | Decision-making |
| Output | Charts | Execution plans |
| Intelligence | None | Safety Stock, EOQ, Risk Fusion |
| Integration | Data in | Data in + decisions out |
| ML | None | LightGBM + Isolation Forest + Prophet |

---

## 📄 License

MIT © Ishai — SupChainMate
