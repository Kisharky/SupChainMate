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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api import (
    agentic_ops, commercial_intel, connectors, customers, data_hub, documents, fraud,
    freight, risk_radar, services, workers, workspace,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create identity tables and seed demo users on first boot."""
    try:
        from api.db import SessionLocal, init_db
        from api.auth import service
        init_db()
        db = SessionLocal()
        try:
            service.seed_default_users(db)
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 — API must still boot
        import logging
        logging.getLogger("api").warning("identity bootstrap skipped: %s", exc)
    yield


app = FastAPI(
    title="SupChainMate API",
    version="1.0.0",
    description="Decision-intelligence JSON API over the SupChainMate engines.",
    lifespan=lifespan,
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

# ---- Authentication & RBAC (isolated in api/auth) ----
from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from api.auth import deps as auth_deps  # noqa: E402
from api.auth.router import router as auth_router  # noqa: E402


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    """Central auth + RBAC gate — protects every /api route by path, leaving the
    domain handlers untouched. Public paths and (optionally) offline demo mode
    pass straight through."""
    rejection = auth_deps.enforce(request)
    if rejection is not None:
        status, detail = rejection
        return JSONResponse({"detail": detail}, status_code=status)
    return await call_next(request)


app.include_router(auth_router)


def _wire_decision_brain() -> None:
    """Integrate the Decision Brain with the Planner via the extensibility hook —
    no change to the planner package. Registers a capability that recalls relevant
    memory (past decisions + knowledge) for whatever objective is being planned."""
    try:
        from brain import BRAIN
        from planner import PLANNER
        from planner.schemas import Capability

        def _recall_context(ctx: dict) -> dict:
            c = BRAIN.context_for(ctx.get("objective", ""), top_k=4)
            return {"summary": (f"Recalled {c['n']} relevant memory item(s) — past "
                                f"decisions, company knowledge, feedback — for context."),
                    "findings": [cit["title"] for cit in c["citations"][:3]],
                    "metrics": {"recalled": c["n"]}, "confidence": 0.7}

        PLANNER.register(Capability(
            name="recall_context",
            description="Retrieve relevant past decisions and company knowledge from the Decision Brain.",
            required_inputs=[], outputs=["recalled"], dependencies=[], confidence=0.7,
            priority=0, handler=_recall_context,
            keywords=["reduce", "increase", "improve", "optimise", "optimize", "cost", "inventory",
                      "revenue", "margin", "profit", "risk", "forecast", "demand", "supplier",
                      "customer", "service", "holding", "leakage", "contract", "warehouse",
                      "logistics", "route", "delivery", "pricing"]))
    except Exception:  # noqa: BLE001 — the API must boot even if the Brain is unavailable
        pass


_wire_decision_brain()


class AskRequest(BaseModel):
    query: str


class DecideRequest(BaseModel):
    rec_key: str
    status: str  # APPROVED | REJECTED | MODIFIED | ESCALATED
    note: str = ""
    actor: str = "user"


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


class RecallRequest(BaseModel):
    query: str
    kinds: list[str] | None = None
    top_k: int = 6


class RememberRequest(BaseModel):
    title: str
    content: str
    doc_type: str = "document"


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


# ---- Connectors & Integrations (enterprise administration) ----
class ConnectorRequest(BaseModel):
    connector_id: str


@app.get("/api/connectors")
def connectors_catalog() -> dict:
    return connectors.catalog()


@app.get("/api/connectors/config/{connector_id}")
def connector_config(connector_id: str) -> dict:
    return connectors.config_schema(connector_id)


@app.post("/api/connectors/test")
def connector_test(req: ConnectorRequest) -> dict:
    return connectors.test_connection(req.connector_id)


@app.get("/api/ai/status")
def ai_status() -> dict:
    return services.ai_status()


# ---- AI Digital Workers cockpit ----
@app.get("/api/workers")
def workers_cockpit() -> dict:
    return workers.cockpit()


@app.get("/api/agentic-ops")
def agentic_ops_workflows() -> dict:
    return agentic_ops.workflows()


# ---- Fraud & Anomaly Detection ----
@app.get("/api/fraud")
def fraud_overview() -> dict:
    return fraud.overview()


# ---- Invoice & Document Intelligence ----
@app.get("/api/documents")
def documents_overview() -> dict:
    return documents.overview()


@app.get("/api/documents/{doc_id}")
def document_detail(doc_id: str) -> dict:
    return documents.detail(doc_id)


# ---- Freight Operations (brokerage) ----
class QuoteRequest(BaseModel):
    origin: str
    destination: str
    equipment: str = "Dry Van"
    miles: int = 0


@app.get("/api/freight")
def freight_overview() -> dict:
    return freight.overview()


@app.get("/api/freight/carrier/{carrier_id}")
def freight_carrier(carrier_id: str) -> dict:
    return freight.carrier_detail(carrier_id)


@app.post("/api/freight/quote")
def freight_quote(req: QuoteRequest) -> dict:
    return freight.quote(req.origin, req.destination, req.equipment, req.miles)


# ---- Disruption & Risk Radar ----
@app.get("/api/radar")
def radar_overview() -> dict:
    return risk_radar.overview()


@app.get("/api/radar/node/{node_id}")
def radar_node(node_id: str) -> dict:
    return risk_radar.node_detail(node_id)


# ---- Data Hub (enterprise data onboarding) ----
from fastapi import File, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402


class MapRequest(BaseModel):
    id: str
    mapping: dict[str, str]


class ImportRequest(BaseModel):
    id: str
    mapping: dict[str, str] | None = None
    options: dict[str, bool] = {}
    imported_by: str = "Enterprise User"


class IndexRequest(BaseModel):
    id: str
    options: dict[str, bool] = {}


@app.post("/api/data/upload")
async def data_upload(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    return data_hub.upload(file.filename or "upload.csv", content)


@app.post("/api/data/map")
def data_map(req: MapRequest) -> dict:
    return data_hub.set_mapping(req.id, req.mapping)


@app.post("/api/data/import")
def data_import(req: ImportRequest) -> dict:
    return data_hub.do_import(req.id, req.mapping, req.options, req.imported_by)


@app.post("/api/data/index")
def data_index(req: IndexRequest) -> dict:
    return data_hub.reindex(req.id, req.options)


@app.get("/api/data/datasets")
def data_datasets() -> dict:
    return data_hub.datasets()


@app.get("/api/data/preview/{dataset_id}")
def data_preview(dataset_id: str) -> dict:
    return data_hub.preview(dataset_id)


@app.get("/api/data/quality")
def data_quality() -> dict:
    return data_hub.quality()


@app.get("/api/data/active")
def data_active() -> dict:
    from api import data_source
    return data_source.active_summary()


@app.delete("/api/data/dataset/{dataset_id}")
def data_delete(dataset_id: str) -> dict:
    return data_hub.delete(dataset_id)


@app.get("/api/data/download/{dataset_id}")
def data_download(dataset_id: str):
    fp = data_hub.filepath(dataset_id)
    if fp is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    path, name = fp
    return FileResponse(path, filename=name)


# ---- Customer 360 (single source of truth per customer) ----
class ChatRequest(BaseModel):
    message: str


@app.get("/api/customers")
def customers_list() -> dict:
    return customers.list_customers()


@app.get("/api/customers/{customer_id}")
def customer_detail(customer_id: str) -> dict:
    return customers.detail(customer_id)


@app.get("/api/customers/{customer_id}/orders")
def customer_orders(customer_id: str) -> dict:
    return customers.orders(customer_id)


@app.get("/api/customers/{customer_id}/shipments")
def customer_shipments(customer_id: str) -> dict:
    return customers.shipments(customer_id)


@app.get("/api/customers/{customer_id}/forecast")
def customer_forecast(customer_id: str) -> dict:
    return customers.forecast(customer_id)


@app.get("/api/customers/{customer_id}/recommendations")
def customer_recommendations(customer_id: str) -> dict:
    return customers.recommendations(customer_id)


@app.get("/api/customers/{customer_id}/timeline")
def customer_timeline(customer_id: str) -> dict:
    return customers.timeline(customer_id)


@app.get("/api/customers/{customer_id}/brain")
def customer_brain(customer_id: str) -> dict:
    return customers.brain(customer_id)


@app.post("/api/customers/{customer_id}/chat")
def customer_chat(customer_id: str, req: ChatRequest) -> dict:
    return customers.chat(customer_id, req.message)


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
        # Objective is passed into the shared context so the Decision Brain's
        # recall_context capability can retrieve relevant memory.
        decision = PLANNER.plan(req.request, context={"objective": req.request})
        d = decision.to_dict()
        try:  # store the decision in long-term memory
            from brain import BRAIN
            BRAIN.record_decision(decision.objective, decision.executive_summary,
                                  {"run_id": decision.run_id, "confidence": decision.confidence,
                                   "capabilities": decision.capabilities,
                                   "financial_impact": decision.financial_impact})
        except Exception:  # noqa: BLE001
            pass
        return d
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


# ---- Decision Brain (long-term memory + knowledge) ----
@app.post("/api/brain/recall")
def brain_recall(req: RecallRequest) -> dict:
    from brain import BRAIN
    hits = BRAIN.recall(req.query, kinds=req.kinds, top_k=req.top_k)
    return {"query": req.query, "results": [h.to_dict() for h in hits]}


@app.post("/api/brain/answer")
def brain_answer(req: RecallRequest) -> dict:
    from brain import BRAIN
    return BRAIN.answer(req.query, top_k=req.top_k)


@app.post("/api/brain/remember")
def brain_remember(req: RememberRequest) -> dict:
    from brain import BRAIN
    rid = BRAIN.add_knowledge(req.title, req.content, doc_type=req.doc_type)
    return {"ok": True, "id": rid}


@app.get("/api/brain/stats")
def brain_stats() -> dict:
    from brain import BRAIN
    return BRAIN.stats()


@app.get("/api/brain/recent")
def brain_recent(limit: int = 30) -> dict:
    from brain import BRAIN
    return {"records": BRAIN.recent(limit)}


@app.post("/api/brain/ingest")
def brain_ingest() -> dict:
    from brain import BRAIN
    return BRAIN.ingest_existing()
