"use client";
/**
 * Connectors & Integrations — the enterprise "where does the data come from?"
 * workspace. Catalog of operational systems (ERP/WMS/TMS/cloud/db/BI/API/files),
 * a configuration panel, a sync dashboard, the ingestion pipeline, a mock data
 * upload, and a note on future AI-assisted connector generation.
 *
 * UI-only: connection tests and config are representative (labelled). The clean
 * seam lives in the backend (api/connectors.py) so real drivers plug in later.
 * No Planner / AI Router / Decision Brain / business logic is touched.
 */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { AdminNav } from "@/components/AdminNav";
import {
  Card, CardHead, KpiCard, Badge, Button, Alert, Progress, Modal,
  DataTable, Th, Td, EmptyState,
} from "@/components/ui/primitives";
import { api, Connector, ConnectorsResponse, ConnectorConfig, ConnectorTest } from "@/lib/api";

const compact = (n: number) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);

type TestState = ConnectorTest & { testing?: boolean };

export default function Connectors() {
  const [d, setD] = useState<ConnectorsResponse | null>(null);
  const [err, setErr] = useState(false);
  const [tests, setTests] = useState<Record<string, TestState>>({});
  const [selected, setSelected] = useState<Connector | null>(null);

  useEffect(() => { api.connectors().then(setD).catch(() => setErr(true)); }, []);

  const runTest = async (id: string) => {
    setTests((t) => ({ ...t, [id]: { ...(t[id] as TestState), testing: true } as TestState }));
    try {
      const res = await api.connectorTest(id);
      setTests((t) => ({ ...t, [id]: { ...res, testing: false } }));
    } catch {
      setTests((t) => ({ ...t, [id]: {
        ok: false, connector_id: id, name: id, status: "error",
        message: "Couldn't reach the API — start the FastAPI backend.", latency_ms: 0,
        source: "fallback", testing: false } }));
    }
  };

  const sum = d?.summary;

  return (
    <AppShell title="Connectors">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>
          Administration · Integrations {d?.source === "representative" ? "· representative catalog" : ""}
        </div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Enterprise Integrations</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem] max-w-2xl">
          Connect operational systems, cloud platforms, databases, and APIs to power enterprise decision intelligence.
        </p>
      </div>

      <AdminNav />

      {/* ---- Connection summary ---- */}
      <div className="grid gap-3 mb-6" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))" }}>
        <KpiCard label="Active Connections" value={sum?.active_connections ?? "—"} status="good" seed={3} />
        <KpiCard label="Connected Systems" value={sum?.connected_systems ?? "—"} status="info" seed={7} />
        <KpiCard label="Last Synchronisation" value={sum?.last_sync ?? "—"} status="good" seed={11} />
        <KpiCard label="Data Health" value={sum?.data_health ?? "—"} unit="%" status="good" seed={5} />
        <KpiCard label="Failed Connections" value={sum?.failed_connections ?? "—"} status={sum && sum.failed_connections > 0 ? "warning" : "good"} seed={9} />
        <KpiCard label="Daily Records Processed" value={sum ? compact(sum.daily_records) : "—"} status="info" seed={13} />
      </div>

      {err && <Alert status="critical" title="API unreachable">Start the FastAPI backend to load the connector catalog.</Alert>}

      {/* ---- Available connectors ---- */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-[1.25rem] font-semibold tracking-tight">Available Connectors</h2>
        <div className="text-[0.75rem] text-ink-3">{d?.categories.length ?? 0} categories</div>
      </div>

      {!d && !err && <Card><EmptyState kind="loading" /></Card>}

      <div className="flex flex-col gap-6">
        {(d?.categories ?? []).map((cat) => (
          <section key={cat.category}>
            <div className="flex items-center gap-2 mb-2.5">
              <h3 className="text-[0.9375rem] font-semibold">{cat.category}</h3>
              <span className="text-[0.6875rem] text-ink-3">· {cat.auth}</span>
              <span className="ml-auto text-[0.6875rem] text-ink-3">
                {cat.connectors.filter((c) => c.connected).length}/{cat.connectors.length} connected
              </span>
            </div>
            <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(280px,1fr))" }}>
              {cat.connectors.map((c) => {
                const t = tests[c.id];
                return (
                  <Card key={c.id} className="p-4 flex flex-col gap-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <span className="grid h-9 w-9 place-items-center rounded-md text-[16px] flex-none"
                          style={{ background: "var(--panel-2)", border: "1px solid var(--hairline)", color: "var(--accent)" }}>{c.icon}</span>
                        <div className="min-w-0">
                          <div className="font-semibold text-[0.875rem] truncate">{c.name}</div>
                          <div className="text-[0.6875rem] text-ink-3">{c.auth}</div>
                        </div>
                      </div>
                      <Badge status={c.connected ? "good" : "neutral"}>{c.connected ? "Connected" : "Not Connected"}</Badge>
                    </div>
                    <p className="text-[0.8125rem] text-ink-2 leading-snug min-h-[2.4em]">{c.description}</p>

                    {t && !t.testing && (
                      <div className="text-[0.75rem] rounded border px-2.5 py-1.5"
                        style={{ borderColor: "var(--hairline)", background: "var(--panel-2)",
                                 color: t.ok ? "var(--good)" : "var(--critical)" }}>
                        {t.ok ? "✓" : "✕"} {t.message} {t.ok && <span className="text-ink-3">· {t.latency_ms} ms</span>}
                      </div>
                    )}

                    <div className="flex gap-2 mt-auto">
                      <Button sm variant="secondary" onClick={() => setSelected(c)}>Configure</Button>
                      <Button sm variant="ghost" onClick={() => runTest(c.id)} disabled={t?.testing}>
                        {t?.testing ? "Testing…" : "Test Connection"}
                      </Button>
                    </div>
                  </Card>
                );
              })}
            </div>
          </section>
        ))}
      </div>

      {/* ---- Sync dashboard + pipeline ---- */}
      <div className="grid gap-4 mt-8 items-start" style={{ gridTemplateColumns: "1fr 1.25fr" }}>
        <SyncDashboard sync={d?.sync} />
        <PipelineCard pipeline={d?.pipeline} />
      </div>

      {/* ---- Upload data ---- */}
      <div className="mt-4"><UploadCard /></div>

      {/* ---- Future AI integration ---- */}
      <Card className="mt-4">
        <CardHead title="AI-assisted connector generation" hint="roadmap · informational" />
        <div className="p-[18px]">
          <Alert status="info" title="Future: schemas mapped and validated by the AI Router">
            In a future release, pointing SupChainMate at a new source will let the existing
            provider-agnostic <b>AI Router</b> infer the mapping to the canonical supply-chain
            model, propose transformations, and validate the schema — turning a multi-day
            integration into a reviewed, one-click connector. No new AI system is introduced;
            it reuses the router already powering the platform. This section is informational only.
          </Alert>
        </div>
      </Card>

      {/* ---- Configuration panel ---- */}
      <ConfigPanel connector={selected} onClose={() => setSelected(null)} onTest={runTest} test={selected ? tests[selected.id] : undefined} />
    </AppShell>
  );
}

/* ---- Sync dashboard ---- */
function SyncDashboard({ sync }: { sync?: ConnectorsResponse["sync"] }) {
  return (
    <Card>
      <CardHead title="Sync dashboard" hint={sync?.frequency ?? ""} right={sync && <Badge status={sync.status === "healthy" ? "good" : "warning"}>{sync.status}</Badge>} />
      <div className="p-[18px] flex flex-col gap-3">
        {!sync ? <EmptyState kind="loading" /> : (<>
          <div className="grid gap-3" style={{ gridTemplateColumns: "1fr 1fr" }}>
            {[
              { l: "Last sync", v: sync.last_sync },
              { l: "Next scheduled", v: sync.next_sync },
              { l: "Records imported", v: sync.records_imported.toLocaleString() },
              { l: "Records failed", v: sync.records_failed.toLocaleString(), warn: sync.records_failed > 0 },
              { l: "Duration", v: `${sync.duration_s}s` },
              { l: "Status", v: sync.status },
            ].map((m) => (
              <div key={m.l} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                <div className="eyebrow">{m.l}</div>
                <div className="text-[1.05rem] font-bold tnum mt-0.5" style={{ color: m.warn ? "var(--warning)" : "var(--ink)" }}>{m.v}</div>
              </div>
            ))}
          </div>
          <div>
            <div className="flex justify-between text-[0.75rem] text-ink-3 mb-1">
              <span>Last cycle</span><span>{sync.progress}%</span>
            </div>
            <Progress value={sync.progress} status="good" />
          </div>
        </>)}
      </div>
    </Card>
  );
}

/* ---- Data pipeline flow ---- */
const PIPE_COLOR: Record<string, string> = {
  source: "var(--info)", process: "var(--text-2)", intelligence: "var(--accent)", output: "var(--good)",
};
function PipelineCard({ pipeline }: { pipeline?: ConnectorsResponse["pipeline"] }) {
  return (
    <Card>
      <CardHead title="Data pipeline" hint="source → decision" />
      <div className="p-[18px]">
        {!pipeline ? <EmptyState kind="loading" /> : (
          <div className="flex flex-col md:flex-row md:items-stretch gap-2">
            {pipeline.map((s, i) => (
              <div key={s.stage} className="flex md:flex-col items-center md:flex-1 gap-2">
                <div className="flex-1 w-full rounded-lg border p-3 text-center"
                  style={{ borderColor: "var(--hairline)", background: "var(--panel-2)",
                           borderTop: `2px solid ${PIPE_COLOR[s.kind] ?? "var(--hairline)"}` }}>
                  <div className="text-[0.8125rem] font-semibold">{s.stage}</div>
                  <div className="text-[0.6875rem] text-ink-3 mt-1 leading-snug">{s.detail}</div>
                </div>
                {i < pipeline.length - 1 && (
                  <span className="text-ink-3 text-[13px] flex-none md:rotate-90" aria-hidden>↓</span>
                )}
              </div>
            ))}
          </div>
        )}
        <div className="text-[0.6875rem] text-ink-3 mt-3">
          Every connected source flows through validation and transformation into the Decision Brain and Planner —
          the same layers that already power the executive decisions.
        </div>
      </div>
    </Card>
  );
}

/* ---- Upload data (mock UI) ---- */
const MOCK_SCHEMA = [
  { column: "order_id", type: "string", nullable: false, mapped: "order.id" },
  { column: "sku", type: "string", nullable: false, mapped: "item.sku" },
  { column: "quantity", type: "integer", nullable: false, mapped: "item.qty" },
  { column: "unit_price", type: "decimal", nullable: false, mapped: "item.price" },
  { column: "order_date", type: "date", nullable: false, mapped: "order.date" },
  { column: "customer_region", type: "string", nullable: true, mapped: "customer.region" },
  { column: "notes", type: "string", nullable: true, mapped: "—" },
];
const MOCK_PREVIEW = [
  ["ORD-10241", "SKU-7781", "12", "24.90", "2026-07-18", "SP"],
  ["ORD-10242", "SKU-3355", "4", "88.00", "2026-07-18", "RJ"],
  ["ORD-10243", "SKU-7781", "9", "24.90", "2026-07-19", "MG"],
];
function UploadCard() {
  const [file, setFile] = useState<string | null>(null);
  const mapped = MOCK_SCHEMA.filter((s) => s.mapped !== "—").length;

  return (
    <Card>
      <CardHead title="Upload data" hint="CSV · Excel · JSON" />
      <div className="p-[18px]">
        {!file ? (
          <div className="rounded-lg border border-dashed p-8 text-center" style={{ borderColor: "var(--hairline-strong)" }}>
            <div className="text-[1.5rem] text-ink-3" aria-hidden>⇪</div>
            <div className="text-[0.9375rem] font-semibold mt-2">Drop a file or choose a sample</div>
            <div className="text-[0.8125rem] text-ink-3 mt-1">We detect the schema, preview rows, and map to the canonical model. (Demo — no data is ingested.)</div>
            <div className="flex gap-2 justify-center mt-3 flex-wrap">
              {["orders.csv", "inventory.xlsx", "shipments.json"].map((f) => (
                <Button key={f} sm variant="secondary" onClick={() => setFile(f)}>{f}</Button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div className="flex items-center gap-2">
                <Badge status="good">Validated</Badge>
                <span className="font-mono text-[0.8125rem]">{file}</span>
                <span className="text-[0.75rem] text-ink-3">· {MOCK_PREVIEW.length * 431} rows · {MOCK_SCHEMA.length} columns</span>
              </div>
              <Button sm variant="ghost" onClick={() => setFile(null)}>Choose another</Button>
            </div>

            <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(140px,1fr))" }}>
              {[
                { l: "Schema detected", v: `${MOCK_SCHEMA.length} cols`, s: "good" },
                { l: "Mapped to model", v: `${mapped}/${MOCK_SCHEMA.length}`, s: "good" },
                { l: "Unmapped", v: `${MOCK_SCHEMA.length - mapped}`, s: "warning" },
                { l: "Validation", v: "Passed", s: "good" },
              ].map((m) => (
                <div key={m.l} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                  <div className="eyebrow">{m.l}</div>
                  <div className="text-[1rem] font-bold mt-0.5" style={{ color: m.s === "warning" ? "var(--warning)" : "var(--good)" }}>{m.v}</div>
                </div>
              ))}
            </div>

            <div>
              <div className="eyebrow mb-1.5">Preview</div>
              <DataTable head={<>{MOCK_SCHEMA.slice(0, 6).map((s) => <Th key={s.column}>{s.column}</Th>)}</>}>
                {MOCK_PREVIEW.map((row, i) => (
                  <tr key={i}>{row.map((cell, j) => <Td key={j}>{cell}</Td>)}</tr>
                ))}
              </DataTable>
            </div>

            <div>
              <div className="eyebrow mb-1.5">Schema & mapping summary</div>
              <DataTable head={<><Th>Column</Th><Th>Type</Th><Th>Nullable</Th><Th>Canonical field</Th></>}>
                {MOCK_SCHEMA.map((s) => (
                  <tr key={s.column}>
                    <Td strong><span className="font-mono text-[0.75rem]">{s.column}</span></Td>
                    <Td>{s.type}</Td>
                    <Td>{s.nullable ? "yes" : "no"}</Td>
                    <Td>{s.mapped === "—"
                      ? <Badge status="warning">unmapped</Badge>
                      : <span className="font-mono text-[0.75rem] text-ink-2">{s.mapped}</span>}</Td>
                  </tr>
                ))}
              </DataTable>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

/* ---- Configuration panel (modal) ---- */
const FREQ_OPTS = ["Real-time", "Every 5 minutes", "Every 15 minutes", "Hourly", "Daily"];
const ENV_OPTS = ["Production", "Staging", "Sandbox"];

function ConfigPanel({ connector, onClose, onTest, test }:
  { connector: Connector | null; onClose: () => void; onTest: (id: string) => void; test?: TestState }) {
  const [cfg, setCfg] = useState<ConnectorConfig | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setCfg(null); setSaved(false);
    if (connector) api.connectorConfig(connector.id).then(setCfg).catch(() => setCfg(null));
  }, [connector]);

  if (!connector) return null;

  const fieldType = (label: string): "password" | "freq" | "env" | "text" =>
    /password/i.test(label) ? "password" : /frequency/i.test(label) ? "freq"
      : /environment/i.test(label) ? "env" : "text";

  return (
    <Modal open={!!connector} onClose={onClose} wide
      title={`Configure · ${connector.name}`} subtitle={`${connector.category} · authentication: ${connector.auth}`}
      footer={<>
        <Button variant="danger" sm onClick={onClose}>Disconnect</Button>
        <Button variant="secondary" sm onClick={() => onTest(connector.id)} disabled={test?.testing}>
          {test?.testing ? "Testing…" : "Test Connection"}
        </Button>
        <Button variant="primary" sm onClick={() => { setSaved(true); }}>Save</Button>
      </>}>
      <div className="flex items-center gap-2 mb-4">
        <Badge status={connector.connected ? "good" : "neutral"}>{connector.connected ? "Connected" : "Not Connected"}</Badge>
        <span className="text-[0.75rem] text-ink-3">Fields below are representative — no system is contacted.</span>
      </div>

      {!cfg ? <EmptyState kind="loading" /> : (
        <div className="grid gap-3" style={{ gridTemplateColumns: "1fr 1fr" }}>
          {cfg.fields.map((label) => {
            const type = fieldType(label);
            return (
              <label key={label} className="flex flex-col gap-1">
                <span className="text-[0.75rem] font-medium text-ink-2">{label}</span>
                {type === "freq" || type === "env" ? (
                  <select className="rounded-sm border bg-[var(--panel-2)] px-2.5 py-1.5 text-[0.8125rem] text-ink"
                    style={{ borderColor: "var(--hairline-strong)" }} defaultValue={type === "env" ? "Production" : "Every 15 minutes"}>
                    {(type === "freq" ? FREQ_OPTS : ENV_OPTS).map((o) => <option key={o}>{o}</option>)}
                  </select>
                ) : (
                  <input type={type === "password" ? "password" : "text"}
                    placeholder={type === "password" ? "••••••••" : label}
                    defaultValue={type === "password" ? "supchain-secret" : ""}
                    className="rounded-sm border bg-[var(--panel-2)] px-2.5 py-1.5 text-[0.8125rem] text-ink"
                    style={{ borderColor: "var(--hairline-strong)" }} />
                )}
              </label>
            );
          })}
        </div>
      )}

      {test && !test.testing && (
        <div className="mt-4 rounded border px-3 py-2 text-[0.8125rem]"
          style={{ borderColor: "var(--hairline)", background: "var(--panel-2)", color: test.ok ? "var(--good)" : "var(--critical)" }}>
          {test.ok ? "✓" : "✕"} {test.message}{test.ok && <span className="text-ink-3"> · {test.latency_ms} ms</span>}
        </div>
      )}
      {saved && <div className="mt-3"><Alert status="good" title="Configuration saved">Representative save — wire a secrets store to persist real credentials.</Alert></div>}
    </Modal>
  );
}
