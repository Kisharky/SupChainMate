/**
 * Typed client for the SupChainMate FastAPI backend.
 * In dev, next.config.mjs rewrites /api/* to the Python server, so a relative
 * base works from the browser. Override with NEXT_PUBLIC_API_BASE if needed.
 */
const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}
async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

// ---- Types (mirror api/services.py shapes) ----------------------------------
export type KpiStatus = "good" | "warning" | "critical" | "info";
export interface Kpi {
  value: number; unit: string; prefix?: string; delta?: number; status: KpiStatus;
}
export interface KpiResponse { kpis: Record<string, Kpi>; source: string; }

export interface InventoryRow {
  sku: string; abc: string; reorder_point: number; eoq: number;
  safety_stock: number; service_level: string; savings_yr: number;
}
export interface InventoryResponse {
  kpis: Record<string, unknown>; rows: InventoryRow[]; source: string;
}

export interface Lane { from: string; to: string; status: KpiStatus; }
export interface DelayedShipment { id: string; lane: string; reason: string; eta_slip: string; }
export interface LogisticsResponse {
  kpis: { in_transit: number; delayed: number; on_time_rate: number; avg_cost: number };
  lanes: Lane[]; delayed: DelayedShipment[]; source: string;
}

export interface Citation { marker?: string; source?: string; name?: string; ref?: string; }
export interface KnowledgeAnswer {
  answer: string; citations: Citation[]; passages?: unknown[];
  confidence?: number | null; retriever?: string; engine?: string; source: string;
}

export interface AgentResult {
  agent: string; objective: string; confidence: number; findings: string[];
  duration_ms: number; requires_approval: boolean; ai_narrative: string | null;
}
export interface WorkflowRun {
  workflow: string; total_ms: number; recommendations_created: number;
  results: AgentResult[]; source: string;
}

export interface ReportItem { id: string; title: string; subtitle: string; status: string; }
export interface ReportsResponse { reports: ReportItem[]; source: string; }

// ---- Endpoints --------------------------------------------------------------
export const api = {
  kpis: () => get<KpiResponse>("/api/kpis"),
  inventory: () => get<InventoryResponse>("/api/inventory"),
  logistics: () => get<LogisticsResponse>("/api/logistics"),
  reports: () => get<ReportsResponse>("/api/reports"),
  knowledgeAsk: (query: string) => post<KnowledgeAnswer>("/api/knowledge/ask", { query }),
  runWorkflow: (workflow = "full_control_tower", ai_enabled = false) =>
    post<WorkflowRun>("/api/agents/run", { workflow, ai_enabled }),
};
