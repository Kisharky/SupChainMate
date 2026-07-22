/**
 * Typed client for the SupChainMate FastAPI backend.
 * In dev, next.config.mjs rewrites /api/* to the Python server, so a relative
 * base works from the browser. Override with NEXT_PUBLIC_API_BASE if needed.
 */
import { tokenStore } from "@/auth/store";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

function authHeaders(): Record<string, string> {
  const t = tokenStore.access();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

/** Refresh the access token once when a request 401s (session persistence). */
async function tryRefresh(): Promise<boolean> {
  const refresh = tokenStore.refresh();
  if (!refresh) return false;
  const res = await fetch(`${BASE}/api/auth/refresh`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refresh }),
  });
  if (!res.ok) { tokenStore.clear(); return false; }
  const body = await res.json();
  tokenStore.set(body.access_token, body.refresh_token, body.user);
  return true;
}

async function request<T>(path: string, init: RequestInit, retry = true): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init, cache: "no-store",
    headers: { ...(init.headers || {}), ...authHeaders() },
  });
  if (res.status === 401 && retry && !path.startsWith("/api/auth/")) {
    if (await tryRefresh()) return request<T>(path, init, false);
    if (typeof window !== "undefined") window.location.href = "/login";
  }
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  return request<T>(path, { method: "GET" });
}
async function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
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
export interface AllocationAssignment { source: string; sink: string; units: number; cost: number; }
export interface CarrierAllocation {
  solved: boolean; solver: string; fell_back: boolean; objective: number; baseline: number;
  improvement_pct: number; lanes: string[]; assignments: AllocationAssignment[];
  detail: string; status: { plan?: Record<string, string> };
}
export interface ProcurementResponse { carriers: ProcurementRow[]; impact: Record<string, number>; optimization: CarrierAllocation | null; source: string; }

export interface OperationsResponse { kpis: Record<string, number>; status_counts: Record<string, number>; source: string; }

export interface WarehouseZone { zone: string; lat: number; lon: number; locations: number; utilization: number; }
export interface WarehouseResponse { zones: WarehouseZone[]; avg_utilization: number; hub_count: number; source: string; }

export interface InventoryRow {
  sku: string; abc: string; reorder_point: number; eoq: number;
  safety_stock: number; service_level: string; savings_yr: number;
}
export interface InventoryResponse {
  kpis: Record<string, unknown>; rows: InventoryRow[];
  allocation?: CarrierAllocation & { units_label?: string } | null;
  source: string;
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

// ---- Connectors & Integrations ----
export interface Connector {
  id: string; name: string; icon: string; description: string;
  connected: boolean; category: string; auth: string;
}
export interface ConnectorCategory { category: string; auth: string; connectors: Connector[]; }
export interface ConnectorsResponse {
  categories: ConnectorCategory[];
  summary: {
    active_connections: number; connected_systems: number; last_sync: string;
    data_health: number; failed_connections: number; daily_records: number;
  };
  sync: {
    last_sync: string; next_sync: string; records_imported: number; records_failed: number;
    duration_s: number; status: string; frequency: string; progress: number;
  };
  pipeline: { stage: string; detail: string; kind: string }[];
  source: string;
}
export interface ConnectorConfig {
  ok: boolean; connector_id: string; name: string; category: string;
  auth: string; connected: boolean; fields: string[]; source: string;
}
export interface ConnectorTest {
  ok: boolean; connector_id: string; name: string; status: string;
  message: string; latency_ms: number; source: string;
}

// ---- AI Digital Workers cockpit ----
export interface Worker {
  id: string; name: string; skill: string; domain: string; status: string;
  zero_touch_pct: number; tasks_today: number; exceptions: number; confidence: number; outputs: string[];
}
export interface WorkerTask {
  id: string; worker: string; worker_id: string; domain: string; task: string;
  state: string; state_label: string; state_status: string;
  confidence: number; impact_usd: number; minutes_ago: number;
}
export interface WorkersResponse {
  workers: Worker[];
  summary: {
    active_workers: number; total_workers: number; tasks_automated_today: number;
    zero_touch_pct: number; hours_saved_week: number; awaiting_approval: number; escalated: number;
  };
  queue: WorkerTask[];
  source: string;
}

// ---- Disruption & Risk Radar ----
export interface RadarSignal { layer: string; layer_name: string; severity: number; }
export interface RadarNode {
  id: string; name: string; type: string; lat: number; lon: number; region: string;
  risk_score: number; band: string; status: string; signals: Record<string, number>;
  convergence: number; top_signals: RadarSignal[];
}
export interface RadarLane {
  id: string; from_id: string; to_id: string; from: string; to: string;
  from_lat: number; from_lon: number; to_lat: number; to_lon: number;
  risk_score: number; band: string; status: string; convergence: number; categories: string[];
}
export interface RadarLayerEvent { node_id: string; node: string; lat: number; lon: number; severity: number; band: string; }
export interface RadarLayer {
  id: string; name: string; icon: string; color: string; active_events: number; events: RadarLayerEvent[];
}
export interface RadarAlert {
  id: string; scope: string; ref_id: string; name: string; region: string;
  convergence: number; categories: string[]; composite_score: number; band: string;
  status: string; critical: boolean; why: string; recommended_action: string;
}
export interface RadarResponse {
  index: {
    score: number; band: string; status: string; critical_alerts: number; converging_alerts: number;
    by_category: { id: string; name: string; severity: number; band: string }[];
    by_region: { region: string; score: number; band: string }[];
  };
  nodes: RadarNode[]; lanes: RadarLane[]; layers: RadarLayer[]; alerts: RadarAlert[];
  brief: string; converge_at: number; source: string;
}
export interface RadarNodeDetail {
  ok: boolean; node_id: string; name: string; type: string; region: string;
  risk_score: number; band: string; status: string; convergence: number;
  signals: { layer: string; layer_name: string; severity: number; active: boolean }[];
  why: string; recommended_action: string;
  lanes: { to: string; risk_score: number; band: string }[]; source: string;
}

// ---- Freight Operations (brokerage) ----
export interface CarrierRow {
  id: string; name: string; mc_number: string; dot_number: string;
  authority_status: string; authority_age_days: number; insurance_status: string;
  insurance_days_to_expiry: number; stage: string; flags: string[]; flag_count: number;
  risk_score: number; risk_severity: string; risk_status: string; recommendation: string;
}
export interface LoadMatch {
  carrier: string; carrier_id: string; fit_score: number; lane_loads: number;
  trucks_available: number; on_time_pct: number; risk_score: number;
}
export interface LoadRow {
  id: string; origin: string; destination: string; equipment: string;
  miles: number; pickup: string; weight_lbs: number; matches: LoadMatch[];
}
export interface TriageRow {
  id: string; from: string; subject: string; type: string; type_label: string;
  type_status: string; confidence: number; extracted: Record<string, string>;
  suggested_action: string; minutes_ago: number;
}
export interface FreightResponse {
  summary: {
    carriers_onboarded: number; pending_vetting: number; high_risk_carriers: number;
    open_loads: number; open_claims: number; triage_queue: number;
  };
  carriers: CarrierRow[];
  loads: LoadRow[];
  triage: TriageRow[];
  roadmap: { name: string; detail: string }[];
  source: string;
}
export interface CarrierDetail {
  ok: boolean; carrier_id: string; name: string; mc_number: string; dot_number: string;
  stage: string; risk_score: number; risk_status: string; recommendation: string;
  checks: { name: string; ok: boolean; detail: string }[];
  flags: { code: string; label: string }[]; source: string;
}
export interface QuoteResult {
  origin: string; destination: string; equipment: string; miles: number; transit_days: number;
  breakdown: { label: string; amount: number; basis: string }[];
  carrier_cost: number; margin_pct: number; all_in_rate: number; margin_usd: number; source: string;
}

// ---- Invoice & Document Intelligence ----
export interface DocRow {
  id: string; type: string; type_label: string; vendor: string; po_number: string;
  amount: number; extraction_confidence: number; match_status: string;
  discrepancy_count: number; status: string; hours_ago: number;
}
export interface DocumentsResponse {
  summary: {
    documents_processed: number; straight_through_pct: number; three_way_matched: number;
    exceptions: number; avg_confidence: number; value_in_flight: number;
  };
  queue: DocRow[];
  source: string;
}
export interface DocMatchLine {
  sku: string; description: string; po_qty: number; po_price: number; po_amount: number;
  invoice_qty: number; invoice_price: number; invoice_amount: number; receipt_qty: number; status: string;
}
export interface DocumentDetail {
  ok: boolean; doc_id: string; type_label: string; vendor: string; po_number: string;
  extraction_confidence: number; match_status: string; fields: Record<string, string>;
  lines: DocMatchLine[]; discrepancies: string[]; recommended_action: string; source: string;
}

// ---- Fraud & Anomaly Detection ----
export interface FraudAlert {
  id: string; type: string; type_label: string; icon: string; severity: string;
  severity_status: string; entity: string; detail: string; recommended_action: string;
  amount_at_risk: number; confidence: number; status: string; hours_ago: number;
}
export interface RiskEntity {
  name: string; kind: string; risk_score: number; tier: string; tier_status: string; top_factor: string;
}
export interface FraudResponse {
  summary: {
    open_alerts: number; high_severity: number; amount_at_risk: number;
    entities_flagged: number; duplicate_invoices: number; detection_accuracy: number;
  };
  checks: { name: string; coverage: number; status: string }[];
  alerts: FraudAlert[];
  entities: RiskEntity[];
  source: string;
}

// ---- Decision & Scenario Intelligence workspace ----
export interface BriefRisk { title: string; severity: "high" | "medium" | "low"; area: string; detail: string; }
export interface BriefDecision { title: string; action: string; confidence: number; impact_usd: number; area: string; }
export interface ExecBrief {
  summary: string; risks: BriefRisk[]; recommended: BriefDecision[];
  financial_impact: { at_risk_usd: number; opportunity_usd: number; net_usd: number };
  confidence: number; awaiting_approval: number;
  kpis: { health: number; on_time: number; supplier_health: number; inventory_value_m: number };
}
export interface WhatChanged {
  date: string; changes: string[]; new_risks: string[]; completed: string[];
  realized_savings: number; unresolved: { title: string; reason: string }[];
}
export interface TimelineItem { id: string; title: string; stage: string; confidence: number; impact_usd: number; status: string; outcome: string | null; ts: string; }
export interface Timeline { stages: string[]; counts: Record<string, number>; items: TimelineItem[]; }
// Real Planner (executive decision orchestrator)
export interface PlannerTask { capability: string; ok: boolean; summary: string; confidence: number; duration_ms: number; error: string | null; }
export interface PlannerDecision {
  objective: string; executive_summary: string; key_findings: string[];
  recommended_actions: { action: string; impact_usd: number; confidence: number; capability?: string }[];
  financial_impact: { identified_usd: number; actions: number };
  operational_impact: Record<string, number>; risks: string[]; confidence: number;
  evidence: string[]; kpis: { name: string; value: number | string }[];
  assumptions: string[]; next_steps: string[]; capabilities: string[];
  graph: string[][]; tasks: PlannerTask[]; run_id: string;
}
export interface CoaOption {
  id: string; name: string; implementation_cost: number; expected_savings: number;
  operational_risk: "low" | "medium" | "high"; service_level_impact: number; inventory_impact: string;
  execution_time: string; confidence: number; business_outcome: string; evidence: string[];
  optimization: string; roi: number; score: number;
}
export interface CoaResponse { issue: string; issue_key: string; options: CoaOption[]; recommended: string; }
export interface ScenarioImpact { financial_usd: number; service_pp: number; logistics_pp: number; inventory_pct: number; customers_affected: number; positive: boolean; }
export interface ScenarioResponse {
  kind: string; label: string; magnitude: number; impact: ScenarioImpact;
  before: Record<string, number>; after: Record<string, number>;
  mitigations: { action: string; effect: string; cost: string }[]; narrative: string;
}
export interface WorkspaceCatalog { scenarios: { key: string; label: string }[]; issues: { key: string; label: string }[]; }

export interface AuthTokens { access_token: string; refresh_token: string; token_type: string; expires_in: number; user: import("@/auth/store").AuthUser; }

// ---- Commercial Intelligence workspace ----
export interface CiBrief {
  total_revenue: number; true_operating_cost: number; gross_margin_pct: number;
  net_margin: number; net_margin_pct: number; revenue_leakage: number; profit_uplift: number;
  customers_action: number; accounts_total: number; summary: string;
  recommendations: { title: string; detail: string; impact_usd: number; confidence: number; account: string }[];
}
export interface CiRankRow { id: string; name: string; region: string; revenue: number; true_cost: number; net_margin: number; net_margin_pct: number; action: boolean; }
export interface CiWaterfall { label: string; value: number; kind: string; }
export interface CiProfitability { ranking: CiRankRow[]; waterfall: CiWaterfall[]; heatmap: { categories: string[]; rows: { account: string; cells: number[] }[] }; }
export interface CiCustomer {
  id: string; name: string; region: string; orders: number; revenue: number; cogs: number;
  serve_cost: number; true_cost: number; net_margin: number; net_margin_pct: number; gross_margin_pct: number;
  activities: Record<string, number>; freight: number; returns_pct: number; storage_util: number;
  dso: number; pay_on_time: number; sla_target: number; sla_actual: number; vol_trend: number;
  leakage: Record<string, number>; leakage_total: number; contract: Record<string, number | boolean>;
  insights: string[]; forecast_next_qtr_orders: number; revenue_gap: number;
  inventory_profile: { skus: number; days_cover: number; storage_util: number };
}
export interface CiLeakage {
  annual_leakage: number; recoverable: number; affected_customers: number;
  by_cause: { cause: string; amount: number; detail: string }[];
  items: { account: string; cause: string; cause_label: string; amount: number; root_cause: string; recoverable: number }[];
}
export interface CiContract {
  account: string; region: string; terms: Record<string, number>; contractual_pick: number; actual_pick: number;
  net_margin_pct: number; unprofitable: boolean; pick_underpriced: boolean; renewal_soon: boolean;
}
export interface CiContracts { contracts: CiContract[]; unprofitable_count: number; renewals_90d: number; }
export interface CiPricingRec {
  id: string; account: string; region: string; net_margin_pct: number; changes: Record<string, string>;
  profit_uplift: number; churn_risk_pct: number; confidence: number; negotiation: string; evidence: string[];
}
export interface CiPricing { recommendations: CiPricingRec[]; total_uplift: number; }
export interface CiRiskRow { id: string; account: string; region: string; scores: Record<string, number>; bands: Record<string, string>; overall: number; overall_band: string; }
export interface CiRisk { dimensions: string[]; rows: CiRiskRow[]; }

// ---- Decision Brain (long-term memory) ----
export interface BrainHit { id: string; kind: string; title: string; snippet: string; score: number; semantic: number; lexical: number; source: string; ts: string; }
export interface BrainStats { total: number; by_kind: Record<string, number>; embedder: string; dim: number; }
export interface BrainIngest { ingested: Record<string, number>; stats: BrainStats; }

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
  wsBrief: () => get<ExecBrief>("/api/workspace/brief"),
  wsChanged: () => get<WhatChanged>("/api/workspace/changed"),
  wsTimeline: () => get<Timeline>("/api/workspace/timeline"),
  wsCatalog: () => get<WorkspaceCatalog>("/api/workspace/catalog"),
  wsCoa: (issue: string) => post<CoaResponse>("/api/workspace/coa", { issue }),
  wsScenario: (kind: string, magnitude: number) => post<ScenarioResponse>("/api/workspace/scenario", { kind, magnitude }),
  plannerPlan: (request: string) => post<PlannerDecision>("/api/planner/plan", { request }),
  brainRecall: (query: string, kinds?: string[]) => post<{ query: string; results: BrainHit[] }>("/api/brain/recall", { query, kinds, top_k: 8 }),
  brainStats: () => get<BrainStats>("/api/brain/stats"),
  brainRemember: (title: string, content: string) => post<{ ok: boolean; id: string }>("/api/brain/remember", { title, content }),
  brainIngest: () => post<BrainIngest>("/api/brain/ingest", {}),
  // ---- auth ----
  login: (email: string, password: string) => post<AuthTokens>("/api/auth/login", { email, password }),
  me: () => get<import("@/auth/store").AuthUser>("/api/auth/me"),
  logout: () => post<{ ok: boolean }>("/api/auth/logout", {}),
  admin: () => get<AdminResponse>("/api/admin"),
  workers: () => get<WorkersResponse>("/api/workers"),
  fraud: () => get<FraudResponse>("/api/fraud"),
  documents: () => get<DocumentsResponse>("/api/documents"),
  documentDetail: (id: string) => get<DocumentDetail>(`/api/documents/${id}`),
  radar: () => get<RadarResponse>("/api/radar"),
  radarNode: (id: string) => get<RadarNodeDetail>(`/api/radar/node/${id}`),
  freight: () => get<FreightResponse>("/api/freight"),
  freightCarrier: (id: string) => get<CarrierDetail>(`/api/freight/carrier/${id}`),
  freightQuote: (origin: string, destination: string, equipment: string, miles = 0) =>
    post<QuoteResult>("/api/freight/quote", { origin, destination, equipment, miles }),
  connectors: () => get<ConnectorsResponse>("/api/connectors"),
  connectorConfig: (id: string) => get<ConnectorConfig>(`/api/connectors/config/${id}`),
  connectorTest: (id: string) => post<ConnectorTest>("/api/connectors/test", { connector_id: id }),
  ciBrief: () => get<CiBrief>("/api/commercial/brief"),
  ciProfitability: () => get<CiProfitability>("/api/commercial/profitability"),
  ciCustomer: (id: string) => get<CiCustomer>(`/api/commercial/customer/${id}`),
  ciLeakage: () => get<CiLeakage>("/api/commercial/leakage"),
  ciContracts: () => get<CiContracts>("/api/commercial/contracts"),
  ciPricing: () => get<CiPricing>("/api/commercial/pricing"),
  ciRisk: () => get<CiRisk>("/api/commercial/risk"),
  ciInvoice: (account_id: string, cause: string) => post<{ invoice_no: string; account: string; line_item: string; amount: number; detail: string; status: string }>("/api/commercial/invoice", { account_id, cause }),
  ciDecide: (item: string, action: string, note = "") => post<{ ok: boolean; item: string; status: string }>("/api/commercial/decide", { item, action, note }),
  knowledgeAsk: (query: string) => post<KnowledgeAnswer>("/api/knowledge/ask", { query }),
  runWorkflow: (workflow = "full_control_tower", ai_enabled = false) =>
    post<WorkflowRun>("/api/agents/run", { workflow, ai_enabled }),
};
