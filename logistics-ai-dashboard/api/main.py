"""
api/main.py — SupChainMate REST API (FastAPI).

A thin JSON surface over the existing engines so the Next.js control plane can
read live data. Run from the ``logistics-ai-dashboard`` directory so the
``modules`` / ``ai`` imports resolve:

    uvicorn api.main:app --reload --port 8000

The existing Streamlit app is unaffected — this is additive.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api import commercial_intel, services, workspace

app = FastAPI(
    title="SupChainMate API",
    version="1.0.0",
    description="Decision-intelligence JSON API over the SupChainMate engines.",
)

# Next.js dev server + configurable extra origins.
_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
if os.getenv("FRONTEND_ORIGIN"):
    _origins.append(os.environ["FRONTEND_ORIGIN"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str


class DecideRequest(BaseModel):
    rec_key: str
    status: str  # APPROVED | REJECTED | MODIFIED | ESCALATED
    note: str = ""
    actor: str = "user"


class EmailRequest(BaseModel):
    sku: str


class PlanRequest(BaseModel):
    request: str


class IssueRequest(BaseModel):
    issue: str


class ScenarioRequest(BaseModel):
    kind: str
    magnitude: float = 0.6


class InvoiceRequest(BaseModel):
    account_id: str
    cause: str


class CommercialDecideRequest(BaseModel):
    item: str
    action: str  # APPROVED | REJECTED | SCHEDULED
    note: str = ""


class WorkflowRequest(BaseModel):
    workflow: str = "full_control_tower"
    ai_enabled: bool = False


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "supchainmate-api", "version": app.version}


@app.get("/api/kpis")
def kpis() -> dict:
    return services.executive_kpis()


@app.get("/api/inventory")
def inventory() -> dict:
    return services.inventory_snapshot()


@app.get("/api/logistics")
def logistics() -> dict:
    return services.logistics_snapshot()


@app.get("/api/logistics/map")
def logistics_map() -> dict:
    return services.logistics_map()


@app.get("/api/optimize/route")
def optimize_route() -> dict:
    return services.optimize_route()


@app.get("/api/optimize/status")
def optimize_status() -> dict:
    return services.optimize_status()


@app.get("/api/forecast")
def forecast() -> dict:
    return services.forecast_snapshot()


@app.get("/api/procurement")
def procurement() -> dict:
    return services.procurement_snapshot()


@app.get("/api/operations")
def operations() -> dict:
    return services.operations_snapshot()


@app.get("/api/warehouse")
def warehouse() -> dict:
    return services.warehouse_snapshot()


@app.get("/api/workflows")
def workflows() -> dict:
    return services.list_workflows()


@app.post("/api/agents/run")
def agents_run(req: WorkflowRequest) -> dict:
    return services.run_workflow(req.workflow, ai_enabled=req.ai_enabled)


@app.get("/api/knowledge/stats")
def knowledge_stats() -> dict:
    return services.knowledge_stats()


@app.post("/api/knowledge/ask")
def knowledge_ask(req: AskRequest) -> dict:
    return services.knowledge_ask(req.query)


@app.get("/api/forecast/backtest")
def forecast_backtest() -> dict:
    return services.forecast_backtest()


@app.get("/api/decisions")
def decisions() -> dict:
    return services.decisions_snapshot()


@app.post("/api/decisions/decide")
def decisions_decide(req: DecideRequest) -> dict:
    return services.decide(req.rec_key, req.status, note=req.note, actor=req.actor)


@app.get("/api/audit")
def audit(limit: int = 200) -> dict:
    return services.audit_trail(limit=limit)


@app.get("/api/reports")
def reports() -> dict:
    return services.reports_list()


@app.get("/api/admin")
def admin() -> dict:
    return services.admin_snapshot()


@app.get("/api/commercial")
def commercial() -> dict:
    return services.commercial_snapshot()


@app.post("/api/commercial/email")
def commercial_email(req: EmailRequest) -> dict:
    return services.repricing_email(req.sku)


@app.get("/api/ai/status")
def ai_status() -> dict:
    return services.ai_status()


# ---- Decision & Scenario Intelligence workspace ----
@app.get("/api/workspace/brief")
def workspace_brief() -> dict:
    return workspace.executive_brief()


@app.get("/api/workspace/changed")
def workspace_changed() -> dict:
    return workspace.whats_changed()


@app.get("/api/workspace/timeline")
def workspace_timeline() -> dict:
    return workspace.decision_timeline()


@app.get("/api/workspace/catalog")
def workspace_catalog() -> dict:
    return workspace.scenario_catalog()


@app.post("/api/workspace/plan")
def workspace_plan(req: PlanRequest) -> dict:
    return workspace.plan_request(req.request)


@app.post("/api/workspace/coa")
def workspace_coa(req: IssueRequest) -> dict:
    return workspace.courses_of_action(req.issue)


@app.post("/api/workspace/scenario")
def workspace_scenario(req: ScenarioRequest) -> dict:
    return workspace.simulate_scenario(req.kind, req.magnitude)


# ---- Commercial Intelligence workspace ----
@app.get("/api/commercial/brief")
def ci_brief() -> dict:
    return commercial_intel.commercial_brief()


@app.get("/api/commercial/profitability")
def ci_profitability() -> dict:
    return commercial_intel.profitability()


@app.get("/api/commercial/accounts")
def ci_accounts() -> dict:
    return commercial_intel.account_list()


@app.get("/api/commercial/customer/{account_id}")
def ci_customer(account_id: str) -> dict:
    return commercial_intel.customer_360(account_id)


@app.get("/api/commercial/leakage")
def ci_leakage() -> dict:
    return commercial_intel.leakage_center()


@app.post("/api/commercial/invoice")
def ci_invoice(req: InvoiceRequest) -> dict:
    return commercial_intel.generate_invoice(req.account_id, req.cause)


@app.get("/api/commercial/contracts")
def ci_contracts() -> dict:
    return commercial_intel.contract_intelligence()


@app.get("/api/commercial/pricing")
def ci_pricing() -> dict:
    return commercial_intel.pricing_optimizer()


@app.get("/api/commercial/risk")
def ci_risk() -> dict:
    return commercial_intel.risk_scoring()


@app.post("/api/commercial/decide")
def ci_decide(req: CommercialDecideRequest) -> dict:
    return commercial_intel.decide(req.item, req.action, req.note)


# ---- Planner (executive decision orchestrator) ----
@app.post("/api/planner/plan")
def planner_plan(req: PlanRequest) -> dict:
    from planner import PLANNER
    try:
        return PLANNER.plan(req.request).to_dict()
    except Exception as exc:  # noqa: BLE001
        return {"objective": req.request, "executive_summary": f"Planner error: {exc}",
                "capabilities": [], "graph": [], "tasks": [], "key_findings": [],
                "recommended_actions": [], "financial_impact": {}, "operational_impact": {},
                "risks": [str(exc)], "confidence": 0, "evidence": [], "kpis": [],
                "assumptions": [], "next_steps": [], "run_id": ""}


@app.get("/api/planner/capabilities")
def planner_capabilities() -> dict:
    from planner import PLANNER
    return {"capabilities": PLANNER.capabilities()}


@app.get("/api/planner/history")
def planner_history(limit: int = 20) -> dict:
    from planner import PLANNER
    return {"runs": PLANNER.history(limit)}
