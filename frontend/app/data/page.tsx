"use client";
/**
 * Data Hub — enterprise data onboarding. Drag & drop CSV/Excel/JSON, AI detects
 * the dataset, map columns, validate, then import: the file is indexed into the
 * Knowledge/RAG store and the Decision Brain (real, offline) so the Knowledge
 * Center and Planner immediately see it. Orchestrates existing services only —
 * no business module, Brain, Planner, or AI Router logic is modified.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import {
  Card, CardHead, KpiCard, Badge, Button, Progress, Modal, DataTable, Th, Td,
  EmptyState, Alert, Skeleton,
} from "@/components/ui/primitives";
import { api, UploadResult, Dataset, DatasetsResponse, QualityResponse, PreviewResult, BrainOptions } from "@/lib/api";

const DATASET_TYPES = [
  "Inventory", "Purchase Orders", "Sales Orders", "Shipments", "Suppliers", "Customers",
  "Products", "Warehouse Locations", "Forecast Data", "Demand History", "Production Orders",
  "Transport Costs", "Carrier Performance",
];
const CONNECTORS = [
  { name: "CSV Upload", active: true }, { name: "Excel Upload", active: true }, { name: "JSON Upload", active: true },
  { name: "SAP", active: false }, { name: "Oracle ERP", active: false }, { name: "Dynamics 365", active: false },
  { name: "NetSuite", active: false }, { name: "Odoo", active: false },
];
const BRAIN_OPTS: { key: string; label: string }[] = [
  { key: "index_docs", label: "Index uploaded documents" },
  { key: "learn_suppliers", label: "Learn supplier relationships" },
  { key: "learn_inventory", label: "Learn inventory history" },
  { key: "learn_procurement", label: "Learn procurement history" },
  { key: "semantic_search", label: "Enable semantic search" },
];
const ACCEPT = ".csv,.xlsx,.xls,.json";
const fmtDate = (t: number | null) => (t ? new Date(t * 1000).toLocaleString() : "—");
type Toast = { id: number; msg: string; status: "good" | "warning" | "critical" | "info" };

export default function DataHub() {
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [quality, setQuality] = useState<QualityResponse | null>(null);
  const [staged, setStaged] = useState<UploadResult[]>([]);
  const [uploading, setUploading] = useState(false);
  const [drag, setDrag] = useState(false);
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [confirmDel, setConfirmDel] = useState<Dataset | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  const toast = useCallback((msg: string, status: Toast["status"] = "good") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, msg, status }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3600);
  }, []);

  const refresh = useCallback(() => {
    api.dataDatasets().then((d: DatasetsResponse) => setDatasets(d.datasets)).catch(() => setDatasets([]));
    api.dataQuality().then(setQuality).catch(() => setQuality(null));
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const onFiles = useCallback(async (files: FileList | File[]) => {
    setUploading(true);
    for (const file of Array.from(files)) {
      try {
        const res = await api.dataUpload(file);
        if (res.ok) { setStaged((s) => [...s, res]); toast(`Detected: ${res.dataset.type_label} in ${file.name}`, "info"); }
        else toast(res.error || `Could not parse ${file.name}`, "critical");
      } catch { toast(`Upload failed for ${file.name}`, "critical"); }
    }
    setUploading(false);
  }, [toast]);

  const k = quality?.kpis;

  return (
    <AppShell title="Data Hub">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>Enterprise data onboarding · live</div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Data Hub</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem] max-w-2xl">
          Bring your own operational data in — no code. Upload, let AI detect and map it, validate, then import:
          it&apos;s indexed straight into the Knowledge Center and Decision Brain.
        </p>
      </div>

      {/* ---- Data quality KPIs ---- */}
      <div className="grid gap-3 mb-6" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))" }}>
        {quality ? <>
          <KpiCard label="Data Quality" value={k!.data_quality} unit="%" status={k!.data_quality >= 80 ? "good" : "warning"} seed={3} />
          <KpiCard label="Duplicate Rate" value={k!.duplicate_rate} unit="%" status={k!.duplicate_rate > 5 ? "warning" : "good"} seed={6} />
          <KpiCard label="Missing Fields" value={k!.missing_rate} unit="%" status={k!.missing_rate > 10 ? "warning" : "good"} seed={9} />
          <KpiCard label="Datasets" value={k!.datasets} status="info" seed={2} />
          <KpiCard label="Total Rows" value={k!.total_rows.toLocaleString()} status="info" seed={5} />
          <KpiCard label="Last Refresh" value={k!.last_refresh ? "live" : "—"} status="good" seed={8} />
        </> : Array.from({ length: 6 }).map((_, i) => (
          <Card key={i} className="p-4"><Skeleton w="60%" /><div className="mt-3"><Skeleton w="40%" h="1.6rem" /></div></Card>
        ))}
      </div>

      {/* ---- Upload ---- */}
      <Card className="mb-6">
        <CardHead title="Upload data" hint="CSV · Excel · JSON · multiple files" />
        <div className="p-[18px]">
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files); }}
            onClick={() => fileRef.current?.click()}
            className="rounded-lg border-2 border-dashed p-10 text-center cursor-pointer transition"
            style={{ borderColor: drag ? "var(--accent)" : "var(--hairline-strong)", background: drag ? "color-mix(in srgb,var(--accent) 8%,transparent)" : "transparent" }}>
            <div className="text-[1.75rem] text-ink-3" aria-hidden>⇪</div>
            <div className="text-[0.9375rem] font-semibold mt-2">{uploading ? "Uploading…" : "Drop files here or click to browse"}</div>
            <div className="text-[0.8125rem] text-ink-3 mt-1">Inventory, POs, sales, shipments, suppliers, customers, products, forecasts, demand, and more.</div>
            <input ref={fileRef} type="file" accept={ACCEPT} multiple className="hidden"
              onChange={(e) => { if (e.target.files?.length) onFiles(e.target.files); e.target.value = ""; }} />
          </div>
          <div className="flex flex-wrap gap-1.5 mt-3">
            {DATASET_TYPES.map((t) => <span key={t} className="rounded-full border px-2 py-0.5 text-[0.6875rem] text-ink-3" style={{ borderColor: "var(--hairline)" }}>{t}</span>)}
          </div>
        </div>
      </Card>

      {/* ---- Staged uploads (detect → map → validate → import) ---- */}
      {staged.map((u) => (
        <StagedCard key={u.dataset.id} upload={u} toast={toast}
          onDone={() => { setStaged((s) => s.filter((x) => x.dataset.id !== u.dataset.id)); refresh(); }}
          onCancel={() => { setStaged((s) => s.filter((x) => x.dataset.id !== u.dataset.id)); api.dataDelete(u.dataset.id).catch(() => {}); }} />
      ))}

      {/* ---- Data sources ---- */}
      <Card className="mb-6">
        <CardHead title="Data sources" hint="upload now · connectors coming soon" />
        <div className="p-[18px] grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(180px,1fr))" }}>
          {CONNECTORS.map((c) => (
            <div key={c.name} className="rounded-lg border p-3 flex items-center justify-between gap-2"
              style={{ borderColor: "var(--hairline)", background: c.active ? "var(--panel-2)" : "transparent", opacity: c.active ? 1 : 0.6 }}>
              <span className="text-[0.8125rem] font-medium">{c.name}</span>
              <Badge status={c.active ? "good" : "neutral"}>{c.active ? "Active" : "Soon"}</Badge>
            </div>
          ))}
        </div>
      </Card>

      {/* ---- Imported datasets ---- */}
      <Card className="mb-6">
        <CardHead title="Imported datasets" hint={datasets ? `${datasets.length} total` : ""} />
        <DataTable head={<>
          <Th>Dataset</Th><Th>Type</Th><Th num>Rows</Th><Th>Imported by</Th><Th>Date</Th><Th>Status</Th><Th>Actions</Th>
        </>}>
          {datasets?.map((ds) => (
            <tr key={ds.id}>
              <Td strong>{ds.name}<div className="text-[0.6875rem] text-ink-3 font-normal font-mono">{ds.filename}</div></Td>
              <Td>{ds.type_label}</Td>
              <Td num>{ds.rows.toLocaleString()}</Td>
              <Td>{ds.imported_by || "—"}</Td>
              <Td>{fmtDate(ds.imported_at ?? ds.created_at)}</Td>
              <Td>
                <Badge status={ds.status === "imported" ? "good" : "warning"}>{ds.status}</Badge>
                {ds.indexed ? <span className="text-[0.625rem] ml-1" style={{ color: "var(--accent)" }}>indexed</span> : null}
              </Td>
              <Td>
                <div className="flex gap-1.5">
                  <Button sm variant="ghost" onClick={() => api.dataPreview(ds.id).then(setPreview)}>Preview</Button>
                  <Button sm variant="ghost" onClick={() => api.dataIndex(ds.id, allOpts()).then((r) => toast(`Reindexed · ${r.index.documents} doc, ${r.index.entities} entities`, "good")).then(refresh)}>Reindex</Button>
                  <Button sm variant="ghost" onClick={() => { window.location.href = `/api/data/download/${ds.id}`; }}>Download</Button>
                  <Button sm variant="danger" onClick={() => setConfirmDel(ds)}>Delete</Button>
                </div>
              </Td>
            </tr>
          ))}
          {datasets && datasets.length === 0 && <tr><Td>No datasets yet — upload one above.</Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td></tr>}
          {!datasets && <tr><Td>Loading…</Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td></tr>}
        </DataTable>
      </Card>

      {/* ---- Data quality dashboard ---- */}
      <Card>
        <CardHead title="Data quality dashboard" hint="across imported datasets" />
        <div className="p-[18px] grid gap-5" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <div>
            <div className="eyebrow mb-2">Import history</div>
            {quality && quality.history.length > 0 ? (
              <div className="flex flex-col gap-2">
                {quality.history.map((h, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-[0.75rem] mb-1"><span className="text-ink-2">{h.name}</span><span className="tnum text-ink-3">{h.rows.toLocaleString()} rows</span></div>
                    <Progress value={Math.min(100, Math.round(100 * h.rows / Math.max(1, Math.max(...quality.history.map((x) => x.rows)))))} status="info" />
                  </div>
                ))}
              </div>
            ) : <EmptyState title="No imports yet" hint="Import a dataset to see history." />}
          </div>
          <div>
            <div className="eyebrow mb-2">Data completeness</div>
            {quality && quality.completeness.length > 0 ? (
              <div className="flex flex-col gap-2">
                {quality.completeness.map((c, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-[0.75rem] mb-1"><span className="text-ink-2">{c.name}</span><span className="tnum text-ink-3">{c.completeness}%</span></div>
                    <Progress value={c.completeness} status={c.completeness >= 90 ? "good" : c.completeness >= 70 ? "warning" : "critical"} />
                  </div>
                ))}
              </div>
            ) : <EmptyState title="No data yet" />}
          </div>
        </div>
      </Card>

      {/* ---- Preview modal ---- */}
      <Modal open={!!preview} onClose={() => setPreview(null)} wide title={preview ? `Preview · ${preview.name}` : "Preview"}
        subtitle={preview ? `${preview.rows.length} rows shown` : undefined}>
        {preview && (
          <DataTable head={<>{preview.columns.map((c) => <Th key={c}>{c}{preview.mapping[c] ? <span className="text-[0.625rem] block font-normal" style={{ color: "var(--accent)" }}>→ {preview.mapping[c]}</span> : null}</Th>)}</>}>
            {preview.rows.map((row, i) => (
              <tr key={i}>{preview.columns.map((c) => <Td key={c}>{String(row[c] ?? "")}</Td>)}</tr>
            ))}
          </DataTable>
        )}
      </Modal>

      {/* ---- Delete confirm ---- */}
      <Modal open={!!confirmDel} onClose={() => setConfirmDel(null)}
        title="Delete dataset?" subtitle={confirmDel?.name}
        footer={<>
          <Button sm variant="secondary" onClick={() => setConfirmDel(null)}>Cancel</Button>
          <Button sm variant="danger" onClick={() => { const id = confirmDel!.id; setConfirmDel(null); api.dataDelete(id).then(() => { toast("Dataset deleted (import undone)", "warning"); refresh(); }); }}>Delete</Button>
        </>}>
        <p className="text-[0.875rem] text-ink-2">This removes the dataset and its file. Documents already indexed into the Knowledge Center / Decision Brain remain until reindexed.</p>
      </Modal>

      {/* ---- Toasts ---- */}
      <div className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2">
        {toasts.map((t) => (
          <div key={t.id} className="rounded-lg border px-3.5 py-2.5 text-[0.8125rem] shadow-card flex items-center gap-2 bg-panel"
            style={{ borderColor: "var(--hairline)", borderLeftWidth: 3, borderLeftColor: `var(--${t.status})` }}>
            <span style={{ color: `var(--${t.status})` }}>●</span>{t.msg}
          </div>
        ))}
      </div>
    </AppShell>
  );
}

function allOpts(): BrainOptions {
  return Object.fromEntries(BRAIN_OPTS.map((o) => [o.key, true]));
}

function StagedCard({ upload, onDone, onCancel, toast }: {
  upload: UploadResult; onDone: () => void; onCancel: () => void;
  toast: (m: string, s?: "good" | "warning" | "critical" | "info") => void;
}) {
  const [mapping, setMapping] = useState<Record<string, string>>(upload.dataset.mapping);
  const [opts, setOpts] = useState<BrainOptions>(allOpts());
  const [importing, setImporting] = useState(false);
  const v = upload.dataset.validation;
  const ds = upload.dataset;

  const runImport = async () => {
    setImporting(true);
    try {
      const res = await api.dataImport(ds.id, opts, mapping);
      if (res.ok) toast(`Imported ${ds.name} · indexed ${res.index.documents} doc, ${res.index.entities} entities`, "good");
      else toast(res.error || "Import failed", "critical");
      onDone();
    } catch { toast("Import failed", "critical"); setImporting(false); }
  };

  return (
    <Card className="mb-4" style={{ borderColor: "color-mix(in srgb,var(--accent) 40%,var(--hairline))" }}>
      <CardHead title={`Onboard · ${ds.filename}`} hint={`${ds.rows} rows`} right={<Button sm variant="ghost" onClick={onCancel}>Discard</Button>} />
      <div className="p-[18px] flex flex-col gap-4">
        {/* AI detection */}
        <Alert status="info" title={upload.detection_message}>
          Detected type <b>{ds.type_label}</b> · confidence <b>{ds.confidence}%</b>. Review the mapping and validation below.
        </Alert>

        <div className="grid gap-4" style={{ gridTemplateColumns: "1.3fr 1fr" }}>
          {/* Column mapping */}
          <div>
            <div className="eyebrow mb-1.5">Column mapping</div>
            <DataTable head={<><Th>Source column</Th><Th>{" "}</Th><Th>Canonical field</Th></>}>
              {ds.columns.map((c) => (
                <tr key={c}>
                  <Td strong><span className="font-mono text-[0.75rem]">{c}</span></Td>
                  <Td>→</Td>
                  <Td>
                    <select value={mapping[c] ?? ""} onChange={(e) => setMapping((m) => ({ ...m, [c]: e.target.value }))}
                      className="rounded-sm border bg-[var(--panel-2)] px-2 py-1 text-[0.75rem] text-ink" style={{ borderColor: "var(--hairline-strong)" }}>
                      <option value="">— unmapped —</option>
                      {upload.canonical_fields.map((f) => <option key={f} value={f}>{f}</option>)}
                    </select>
                  </Td>
                </tr>
              ))}
            </DataTable>
          </div>

          {/* Validation */}
          <div>
            <div className="eyebrow mb-1.5">Validation</div>
            <div className="rounded-lg border p-3" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[0.8125rem]">Health score</span>
                <Badge status={v.health_score >= 80 ? "good" : v.health_score >= 60 ? "warning" : "critical"}>{v.health_score}</Badge>
              </div>
              <Progress value={v.health_score} status={v.health_score >= 80 ? "good" : v.health_score >= 60 ? "warning" : "critical"} />
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 mt-3 text-[0.75rem]">
                {[["Rows imported", v.rows], ["Missing values", v.missing_values], ["Duplicates", v.duplicate_records],
                  ["Invalid dates", v.invalid_dates], ["Unknown SKUs", v.unknown_skus], ["Invalid supplier IDs", v.invalid_supplier_ids]].map(([l, n]) => (
                  <div key={l as string} className="flex justify-between"><span className="text-ink-3">{l}</span><span className="tnum text-ink">{n as number}</span></div>
                ))}
              </div>
              {v.warnings.length > 0 && <div className="mt-2 text-[0.6875rem] text-ink-3">{v.warnings.join(" · ")}</div>}
            </div>
          </div>
        </div>

        {/* Decision Brain options */}
        <div>
          <div className="eyebrow mb-1.5">Decision Brain indexing</div>
          <div className="flex flex-wrap gap-x-5 gap-y-2">
            {BRAIN_OPTS.map((o) => (
              <label key={o.key} className="flex items-center gap-2 text-[0.8125rem] cursor-pointer">
                <input type="checkbox" checked={opts[o.key] ?? false} onChange={(e) => setOpts((s) => ({ ...s, [o.key]: e.target.checked }))} />
                {o.label}
              </label>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2 justify-end">
          {v.errors.length > 0 && <span className="text-[0.75rem] text-critical mr-auto">{v.errors.join(" · ")}</span>}
          <Button sm variant="ghost" onClick={onCancel}>Cancel</Button>
          <Button sm variant="primary" onClick={runImport} disabled={importing || v.errors.length > 0}>
            {importing ? "Importing…" : "Import & index"}
          </Button>
        </div>
        {importing && <Progress value={100} status="good" className="animate-pulse" />}
      </div>
    </Card>
  );
}
