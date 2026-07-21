# SupChainMate — Guided Demo Walkthrough

A five-minute script for showing SupChainMate as a decision-intelligence
platform, not a dashboard. Everything below runs offline on the bundled Olist
dataset (99,441 real orders) — no API keys required. Numbers are computed live
from data; anything modeled is labelled as such in the UI.

---

## 0 · Before you present (60 seconds of setup)

```bash
# One command — API + frontend + Postgres
docker compose up

# …or run the two services directly:
#   API       (from logistics-ai-dashboard/):  uvicorn api.main:app --port 8000
#   Frontend  (from frontend/):                npm run dev
```

Open **http://localhost:3000**. You land on the login screen.

**Demo accounts** — all use password `supchain123`:

| Role | Email | What it can see |
|------|-------|-----------------|
| Admin | `admin@supchainmate.io` | Everything, incl. Administration |
| Executive | `exec@supchainmate.io` | All decision surfaces; approvals |
| Supply Chain Manager | `scm@supchainmate.io` | Operations, planning, logistics |
| Planner | `planner@supchainmate.io` | Forecasting, inventory, planner |
| Warehouse Manager | `warehouse@supchainmate.io` | Warehouse, inventory |
| Read Only | `viewer@supchainmate.io` | View-only; no approve/admin |

> Tip: the login screen has one-click chips for each account, so you never type
> credentials on stage.

---

## 1 · The one-line pitch

> "SupChainMate turns a supply chain's raw data into **evidence-backed
> decisions**. Every KPI is computed from the data, every recommendation is
> traceable, and every approval is audited. The AI is provider-agnostic and
> degrades gracefully — it runs fully offline with no keys."

---

## 2 · The five-stop tour

Sign in as **Executive** for the full surface.

### Stop 1 — Executive Control Tower (`/`)
The landing page is a control tower, not an upload form. Point out the six live
KPIs (Supply Chain Health, Today's Risks, Late Shipments, Inventory Value,
Forecast Accuracy, Supplier Health) and the **AI Executive Summary** that reads
the current state in plain language.

**Say:** *"This is what an executive opens on Monday — the state of the network
and what needs a decision today, not a wall of charts."*

### Stop 2 — Decision Center (`/decisions`) ⭐ the trust layer
Every recommendation shows its **evidence, confidence, and financial impact**,
and can be **Approved / Rejected / Modified / Escalated**. Approve one and show
it land in the **audit trail**.

**Say:** *"This is the difference between a dashboard and a decision system —
nothing is a black box, and every action is logged with who, what, and why."*

### Stop 3 — Commercial Intelligence (`/commercial`)
Executive Commercial Brief, True Customer Profitability (ABC), Revenue Leakage,
Contract Intelligence, and the AI Pricing Optimizer. Open a customer to show the
360 view.

**Say:** *"Real order volumes with transparent, labelled margin assumptions —
it finds the money that's quietly leaking out."*

### Stop 4 — Logistics (`/logistics`)
The live route map (MapTiler tiles when a key is set; graceful fallback
otherwise) plus carrier grades. Click **Optimize routing** and watch the
transportation solver improve on the naive baseline, with the % saving shown.

**Say:** *"The optimization layer sits beneath the domain agents — they call a
solver through a router, they never hard-code one."*

### Stop 5 — Inventory (`/inventory`)
Per-SKU reorder points, EOQ, and safety stock computed from real demand, plus
the **multi-DC replenishment allocation** solved with real Haversine distances.

**Say:** *"This is the optimization skill invoked by the inventory agent —
same engine, surfaced as an executive decision."*

---

## 3 · Demo scenarios (pick one to go deeper)

**A. "Show me a decision end-to-end."**
Control Tower → open a flagged risk → Decision Center → review evidence &
confidence → Approve → show the audit entry. *Theme: traceability.*

**B. "Where are we losing money?"**
Commercial → Revenue Leakage Center → open the biggest leak → Pricing Optimizer
→ generate the recommended reprice. *Theme: margin recovery.*

**C. "Plan a cross-functional objective."**
Workspace / Planner → enter an objective like *"reduce inventory holding cost
without hurting service."* → watch it decompose into capabilities, execute, and
merge into one Decision. *Theme: orchestration.*

**D. "Prove the access control."**
Sign out, sign back in as **Read Only** → note Administration is gone and
approve buttons are inert. *Theme: enterprise RBAC.*

**E. "Where does the data come from?"**
Administration → **Connectors** → show the connector catalog (ERP, WMS, TMS,
cloud, databases, BI, APIs, files) → open **Configure** on SAP S/4HANA → **Test
Connection** → then the data pipeline (Source → Validation → Transformation →
Decision Brain → Planner → Executive Dashboard). *Theme: the full narrative —
data sources feed the decisions.*

**F. "Show me the AI doing the work."**
**Workforce** → the digital-worker roster (each is a real platform capability)
with a **zero-touch rate** per worker → the live task queue where auto-completed
tasks post to the audit trail and exceptions route to the Decision Center.
*Theme: agentic automation, human-in-the-loop on exceptions.*

**G. "Catch the money leaving the door."**
**Fraud & Risk** → the anomaly feed (duplicate invoice, double-brokering,
identity risk) with amount-at-risk and a recommended action → **Send to Decision
Center** → the entity risk register scoring carriers/suppliers. *Theme: trust
and fraud prevention.*

**H. "Audit an invoice before we pay it."**
**Documents** → the document queue (exceptions first) → **Review** an exception
→ the three-way match (PO ↔ Invoice ↔ Receipt) with the over-billed line flagged
→ Approve or Escalate. *Theme: AP automation with a human check.*

---

## 4 · Talking points if asked

- **"Is the AI real?"** Yes — a provider-agnostic router maps each capability to
  a model. With no keys it falls back to deterministic/extractive logic, so the
  demo always works. Nothing is faked; modeled figures are labelled.
- **"Is the data real?"** The Olist dataset is 99,441 genuine orders with real
  promised-vs-actual delivery dates. Carrier names, freight costs, and transport
  modes are simulated and labelled as such.
- **"Could this run in production?"** It's built for it — JWT auth, six-role
  RBAC, SQLAlchemy over SQLite *or* Postgres, Docker, and CI — without pulling in
  Kubernetes, Redis, or a message bus. It's a portfolio flagship, deliberately
  lean.

---

## 5 · Reset between demos

State (approvals, audit, memory) lives in SQLite under
`logistics-ai-dashboard/data/`. To start from a clean slate, stop the stack and
remove the app databases (`*_app.db`, `brain_memory`, `planner_runs` tables) or
just re-create the container — the demo users re-seed automatically on boot.
