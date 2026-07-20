# SupChainMate — Design System & Front-End Handoff

The visual foundation and screen designs for the SupChainMate control plane, plus
everything Cursor needs to recreate them in React/Next.js. **The existing Streamlit
backend and AI layer (`logistics-ai-dashboard/`) stay exactly as they are** — this is
a presentation layer, not a rewrite.

## What's here

| File | What it is |
| --- | --- |
| `design-system.html` | **Phase 1–2.** Foundations (color, type) + the eight reusable components, rendered with real supply-chain content. Open in a browser; toggle theme top-right. |
| `prototype.html` | **Phase 3.** Clickable 5-screen prototype. Landing = Executive Control Tower; left nav switches between Inventory, Logistics, Knowledge, and Reports. |
| `tokens.css` | The design tokens as CSS custom properties. Drop into `app/globals.css`. Single source of truth — components read variables, never hard-coded hex. |
| `tailwind.tokens.js` | Tailwind `theme.extend` bound to the same tokens. |

> The design tool step was specced for Stitch MCP. This cloud session can't attach an
> MCP server mid-run, so the design was authored directly to the same spec (Navy /
> Emerald / Slate / White, Inter, SAP Analytics Cloud · Microsoft Fabric · Palantir
> Foundry · Linear). The artifacts above are the deliverable Cursor implements against.

## Design language

- **Color** — Navy grounds the interface; Slate builds structure; **Emerald is the one
  accent**, spent only on the primary action and positive state. Amber/red/sky are
  semantic state signals and never double as the accent.
- **Type** — Inter throughout. Load with `next/font/google` and expose it as
  `--font-sans` so `tokens.css` picks it up.
- **Density** — hairline rules, tabular figures (`font-variant-numeric: tabular-nums`)
  wherever numbers align, summary-before-detail. It's an operated control plane, not a
  document.
- **Both themes** ship. The app stamps `data-theme="light|dark"` on `<html>`; the token
  file handles the rest.

## Navigation (final IA — no "AI" page)

`Dashboard · Operations · Forecasting · Inventory · Procurement · Warehouse · Logistics · Knowledge · Reports · Administration`

AI is integrated into every workflow (the "AI Executive Summary" panel, the per-agent
recommendation cards) — never a standalone destination.

## The eight components → React

| Component | Notes for implementation |
| --- | --- |
| Button | `primary / secondary / ghost / danger`, `sm` size, disabled. Primary = emerald. |
| KPI card | figure + delta + trend rail (left stripe) + canvas sparkline. |
| Table | sticky header, right-aligned tabular numerics, inline bars & status badges. |
| Chart | Canvas area/line with faint grid, area fill, emphasized endpoint. (Swap for Recharts/visx if preferred.) |
| Status badge | `good / warning / critical / info / neutral`, dot + label. |
| Navigation | left rail, active state, count badges. |
| Form | input / select / toggle / segmented control / inline validation. |
| Alert | inline & banner, left severity stripe, semantic icon. |

## The five screens

1. **Executive Control Tower** *(landing)* — 6 KPIs (Health 96%, Today's Risks 4, Late
   Shipments 3, Inventory Value $12.4M, Forecast Accuracy 95%, Supplier Health 91%),
   the **AI Executive Summary** with `Approve Recommendations` / `Review Details`,
   throughput chart, priority risks, agent activity.
2. **Inventory Intelligence** — inventory KPIs, stock-level table with reorder
   recommendations, ABC distribution.
3. **Logistics Command Center** — shipment KPIs, active-lane map, delayed-shipment feed,
   re-route recommendation.
4. **Knowledge Center** — RAG ask bar, grounded answer with inline citations, cited
   source cards. (Backed by `logistics-ai-dashboard/ai/rag.py`.)
5. **Executive Reports** — report cards with PDF export, KPI trend chart, agent summary.

## Suggested Next.js structure

```
frontend/
  app/
    layout.tsx            # Inter via next/font, imports globals.css, <AppShell>
    globals.css           # @import "../../design/tokens.css"; + base styles
    page.tsx              # Executive Control Tower (landing)
    inventory/page.tsx
    logistics/page.tsx
    knowledge/page.tsx
    reports/page.tsx
  components/
    ui/                   # Button, KpiCard, DataTable, Chart, Badge, Alert, Field
    AppShell.tsx          # left rail + topbar
  lib/api.ts              # thin client to the existing backend (unchanged)
  tailwind.config.ts      # theme.extend = require("../design/tailwind.tokens")
```

Wire `lib/api.ts` to the current backend's data functions — no backend changes needed.
