# SupChainMate — Frontend (Next.js)

The React/Next.js control plane for SupChainMate, on top of the existing Python
backend. The Streamlit app and all business logic in `logistics-ai-dashboard/`
are **unchanged** — this frontend talks to a thin FastAPI layer (`api/`) that
imports those same engines and exposes them as JSON.

## Architecture

```
Browser ─▶ Next.js (App Router)          frontend/
             │  fetch /api/*
             ▼  (dev: proxied by next.config.mjs)
          FastAPI  api/main.py    ─────▶  modules/ + ai/   (unchanged engines)
             │                              forecast · sku · agents · rag …
             ▼
          SQLite + Olist demo data
```

- **No standalone "AI" page.** AI is embedded in the workflows (the AI Executive
  Summary, the per-agent runs, the RAG answers).
- One consistent brand + navigation across every screen (the raw Stitch mockups
  drifted on both; `components/AppShell.tsx` is the canonical IA).

## Screens

| Route | Screen | Data |
| --- | --- | --- |
| `/` | Executive Control Tower | Representative headline KPIs + **live** 9-agent run |
| `/inventory` | Inventory Intelligence | **Live** — reorder points / EOQ / safety stock from the decision engine |
| `/logistics` | Logistics Command Center | Lanes + delayed shipments |
| `/knowledge` | Knowledge Center | **Live** — RAG answer with citations (`ai/rag.py`) |
| `/reports` | Executive Reports | Report library + exports |

Operations, Forecasting, Procurement, Warehouse, Administration are placeholders
on the same shell.

## Run it (two terminals)

**1 — Backend API** (from `logistics-ai-dashboard/`):
```bash
pip install -r requirements.txt          # adds fastapi + uvicorn
uvicorn api.main:app --reload --port 8000
```

**2 — Frontend** (from `frontend/`):
```bash
npm install
npm run dev            # http://localhost:3000
```

`next.config.mjs` proxies `/api/*` to `http://localhost:8000`, so there's no CORS
to configure in dev. For a split deployment, set `API_PROXY_TARGET` (build) or
`NEXT_PUBLIC_API_BASE` (browser) to the API's URL.

## Design system

Tokens live in `app/globals.css` (mirrors `design/tokens.css`) and are wired into
Tailwind via `tailwind.config.ts`. Components are in `components/ui/primitives.tsx`
(Button, Card, Badge, KpiCard, Alert, Sparkline, DataTable). Inter + JetBrains Mono
are loaded with `next/font`. Both light and dark themes ship; the topbar toggles.

## Verified

`npm run build` compiles all 13 routes; the app was driven end-to-end against the
live API (Control Tower KPIs + real 9-agent run, 12 live inventory SKUs, RAG answer)
with no runtime errors.
