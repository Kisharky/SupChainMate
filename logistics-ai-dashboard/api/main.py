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

from api import services

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


@app.get("/api/ai/status")
def ai_status() -> dict:
    return services.ai_status()
