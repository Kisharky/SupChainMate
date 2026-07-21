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
export interface KpiResponse { kpis: Record<string, Kpi>; live?: Record<string, boolean>; source: string; }

export interface CarrierRow { carrier: string; shipments: number; on_time: number | null; grade: string; avg_delay: number; }

export interface MapPoint { name: string; lat: number; lon: number; size: number; }
export interface MapRoute { from: { lat: number; lon: number; name: string }; to: { lat: number; lon: number; name: string }; status: KpiStatus; distance_km: number; }
export interface MapResponse { tiles_url: string | null; attribution: string; center: [number, number]; zoom: number; points: MapPoint[]; routes: MapRoute[]; source: string; }

export interface OptimizeLeg { from: string; to: string; distance_km: number; }
export interface OptimizeResponse {
  solved: boolean; solver: string; fell_back: boolean; objective: number; baseline: number;
  improvement_pct: number; order: number[]; legs: OptimizeLeg[]; detail: string;
  tour: { name: string; lat: number; lon: number }[];
  status: { plan?: Record<string, string>; solvers?: Record<string, { configured: boolean }> };
  source: string;
}

export interface ForecastPoint { ds: string; y?: number; yhat?: number; lower?: number; upper?: number; }
export interface ForecastResponse {
  history: ForecastPoint[]; forecast: ForecastPoint[];
  insights: { next_week_total?: number; stockout_risk_short?: string; stockout_risk_detail?: string; demand_pct_change_vs_prior_week?: number; historical_p90_daily?: number };
  source: string;
}

export interface ProcurementRow { carrier: string; score: number; on_time: number | null; current_share: number | null; recommended_share: number | null; }
export interface ProcurementResponse { carriers: ProcurementRow[]; impact: Record<string, number>; source: string; }

export interface OperationsResponse { kpis: Record<string, number>; status_counts: Record<string, number>; source: string; }

export interface WarehouseZone { zone: string; lat: number; lon: number; locations: number; utilization: number; }
export interface WarehouseResponse { zones: WarehouseZone[]; avg_utilization: number; hub_count: number; source: string; }

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
  lanes: Lane[]; delayed: DelayedShipment[]; carriers?: CarrierRow[]; source: string;
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

export interface Driver { reason: string; evidence: string; }
export interface DecisionImpact { cost_savings_yr?: number | null; stockout_risk_pct?: number | null; service_level_pct?: number | null; other?: string | null; }
export interface Recommendation {
  rec_key: string; source: string; category: string; title: string; action: string;
  drivers: Driver[]; confidence: number; confidence_basis: string; impact: DecisionImpact;
  status: string; created_ts?: string; decided_ts?: string; decided_by?: string; note?: string;
}
export interface DecisionsResponse {
  kpis: { pending: number; approved: number; rejected: number; approved_savings: number; avg_confidence: number | null };
  pending: Recommendation[]; history: Recommendation[]; source: string;
}
export interface AuditEntry { ts: string; actor: string; event: string; rec_key: string | null; details: string; }

export interface BacktestResponse {
  mape: number | null; mae: number | null; rmse: number | null; bias: number | null;
  accuracy: number | null; holdout_weeks: number; granularity: string;
  points: { ds: string; actual: number; predicted: number }[]; source: string;
}

export type DecisionStatus = "APPROVED" | "REJECTED" | "MODIFIED" | "ESCALATED";

export interface AdminResponse {
  providers: { capability: string; provider: string; model: string; configured: boolean }[];
  api_keys: { name: string; purpose: string; configured: boolean; masked: string }[];
  audit: AuditEntry[];
  users: { name: string; email: string; role: string; status: string }[];
  roles: { role: string; view: boolean; run: boolean; approve: boolean; admin: boolean }[];
  integrations: { name: string; kind: string; status: string }[];
  source: string;
}

export interface RepricingTicket {
  sku: string; current_price: number; recommended_price: number; uplift_pct: number;
  current_margin_pct: number; annual_impact: number;
}
export interface CommercialResponse {
  segments: { segment: string; orders: number; revenue: number; margin: number; margin_pct: number }[];
  kpis: { total_revenue: number; net_margin: number; net_margin_pct: number; revenue_leakage: number; underpriced_skus: number; repricing_upside: number };
  leakage: { freight: number; discount: number; total: number };
  waterfall: { label: string; value: number; kind: string }[];
  tickets: RepricingTicket[];
  assumptions: { aov: number; gross_margin_pct: number; target_margin_pct: number };
  source: string;
}
export interface EmailResponse { sku: string; subject: string; body: string; ticket: RepricingTicket | null; }

// ---- Endpoints --------------------------------------------------------------
export const api = {
  kpis: () => get<KpiResponse>("/api/kpis"),
  inventory: () => get<InventoryResponse>("/api/inventory"),
  logistics: () => get<LogisticsResponse>("/api/logistics"),
  logisticsMap: () => get<MapResponse>("/api/logistics/map"),
  optimizeRoute: () => get<OptimizeResponse>("/api/optimize/route"),
  forecast: () => get<ForecastResponse>("/api/forecast"),
  procurement: () => get<ProcurementResponse>("/api/procurement"),
  operations: () => get<OperationsResponse>("/api/operations"),
  warehouse: () => get<WarehouseResponse>("/api/warehouse"),
  reports: () => get<ReportsResponse>("/api/reports"),
  backtest: () => get<BacktestResponse>("/api/forecast/backtest"),
  decisions: () => get<DecisionsResponse>("/api/decisions"),
  decide: (rec_key: string, status: DecisionStatus, note = "") =>
    post<{ ok: boolean; rec_key: string; status: string }>("/api/decisions/decide", { rec_key, status, note }),
  audit: () => get<{ entries: AuditEntry[] }>("/api/audit"),
  admin: () => get<AdminResponse>("/api/admin"),
  commercial: () => get<CommercialResponse>("/api/commercial"),
  repricingEmail: (sku: string) => post<EmailResponse>("/api/commercial/email", { sku }),
  knowledgeAsk: (query: string) => post<KnowledgeAnswer>("/api/knowledge/ask", { query }),
  runWorkflow: (workflow = "full_control_tower", ai_enabled = false) =>
    post<WorkflowRun>("/api/agents/run", { workflow, ai_enabled }),
};
