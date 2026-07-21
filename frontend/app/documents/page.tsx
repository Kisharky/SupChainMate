"use client";
/**
 * Invoice & Document Intelligence — extraction + three-way match (PO ↔ Invoice
 * ↔ Receipt) for AP audit. Matched documents auto-approve; exceptions open a
 * review with the discrepancy and route to a human. Representative extraction
 * (labelled) over the AI Router's OCR/RAG capability. No business logic changed.
 */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import {
  Card, CardHead, KpiCard, Badge, Button, Modal, DataTable, Th, Td, EmptyState, Alert,
} from "@/components/ui/primitives";
import { api, DocumentsResponse, DocumentDetail, DocRow } from "@/lib/api";

const compact = (n: number) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);

const MATCH: Record<string, { label: string; status: "good" | "warning" | "critical" }> = {
  matched: { label: "3-way matched", status: "good" },
  partial: { label: "Partial match", status: "warning" },
  exception: { label: "Exception", status: "critical" },
};

export default function Documents() {
  const [d, setD] = useState<DocumentsResponse | null>(null);
  const [err, setErr] = useState(false);
  const [open, setOpen] = useState<DocRow | null>(null);

  useEffect(() => { api.documents().then(setD).catch(() => setErr(true)); }, []);
  const s = d?.summary;

  return (
    <AppShell title="Documents">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>
          AP automation {d?.source === "representative" ? "· representative extraction" : ""}
        </div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Invoice &amp; Document Intelligence</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem] max-w-2xl">
          Reads invoices, bills of lading, and receipts, then runs a three-way match against the PO — auto-approving clean
          documents and flagging over-billing or quantity mismatches before payment.
        </p>
      </div>

      {/* ---- Summary ---- */}
      <div className="grid gap-3 mb-6" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))" }}>
        <KpiCard label="Documents Processed" value={s?.documents_processed ?? "—"} status="info" seed={3} />
        <KpiCard label="Straight-Through" value={s?.straight_through_pct ?? "—"} unit="%" status="good" seed={6} />
        <KpiCard label="3-Way Matched" value={s?.three_way_matched ?? "—"} status="good" seed={2} />
        <KpiCard label="Exceptions" value={s?.exceptions ?? "—"} status={s && s.exceptions > 0 ? "warning" : "good"} seed={9} />
        <KpiCard label="Avg Extraction Conf." value={s?.avg_confidence ?? "—"} unit="%" status="good" seed={5} />
        <KpiCard label="Value in Flight" prefix="$" value={s ? compact(s.value_in_flight) : "—"} status="info" seed={8} />
      </div>

      {err && <Alert status="critical" title="API unreachable">Start the FastAPI backend to load the document queue.</Alert>}

      {/* ---- Document queue ---- */}
      <Card>
        <CardHead title="Document queue" hint="exceptions first"
          right={s && <span className="text-[0.75rem] text-ink-3">{s.exceptions} need review</span>} />
        <DataTable head={<>
          <Th>Document</Th><Th>Vendor</Th><Th>PO</Th><Th num>Amount</Th>
          <Th num>Extraction</Th><Th>Match</Th><Th>{" "}</Th>
        </>}>
          {(d?.queue ?? []).map((doc) => (
            <tr key={doc.id} className="hover:bg-[color-mix(in_srgb,var(--accent)_6%,transparent)]">
              <Td strong>{doc.type_label}<div className="text-[0.6875rem] text-ink-3 font-normal font-mono">{doc.id} · {doc.hours_ago}h ago</div></Td>
              <Td>{doc.vendor}</Td>
              <Td><span className="font-mono text-[0.75rem]">{doc.po_number}</span></Td>
              <Td num>${doc.amount.toLocaleString()}</Td>
              <Td num>{doc.extraction_confidence}%</Td>
              <Td>
                <Badge status={MATCH[doc.match_status].status}>{MATCH[doc.match_status].label}</Badge>
                {doc.discrepancy_count > 0 && <span className="text-[0.6875rem] text-ink-3 ml-1.5">{doc.discrepancy_count} issue{doc.discrepancy_count > 1 ? "s" : ""}</span>}
              </Td>
              <Td><Button sm variant={doc.match_status === "matched" ? "ghost" : "secondary"} onClick={() => setOpen(doc)}>Review</Button></Td>
            </tr>
          ))}
          {!d && !err && <tr><Td>Loading…</Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td><Td> </Td></tr>}
        </DataTable>
        <div className="px-[18px] py-3 text-[0.6875rem] text-ink-3 border-t" style={{ borderColor: "var(--hairline)" }}>
          Extraction confidence and the three-way match come from the AI Router&apos;s OCR/RAG capability. Clean matches
          auto-post to the audit trail; exceptions open below for a human call.
        </div>
      </Card>

      <MatchModal doc={open} onClose={() => setOpen(null)} />
    </AppShell>
  );
}

function MatchModal({ doc, onClose }: { doc: DocRow | null; onClose: () => void }) {
  const [det, setDet] = useState<DocumentDetail | null>(null);
  useEffect(() => {
    setDet(null);
    if (doc) api.documentDetail(doc.id).then(setDet).catch(() => setDet(null));
  }, [doc]);
  if (!doc) return null;

  return (
    <Modal open={!!doc} onClose={onClose} wide
      title={`${doc.type_label} · ${doc.vendor}`}
      subtitle={`${doc.id} · PO ${doc.po_number} · extraction ${doc.extraction_confidence}% confidence`}
      footer={<>
        <Button variant="danger" sm onClick={onClose}>Escalate</Button>
        <Button variant="primary" sm onClick={onClose}>{doc.match_status === "matched" ? "Approve payment" : "Approve with note"}</Button>
      </>}>
      {!det ? <EmptyState kind="loading" /> : !det.ok ? (
        <EmptyState kind="error" title="Couldn't load document detail" />
      ) : (
        <div className="flex flex-col gap-4">
          {/* extracted fields */}
          <div>
            <div className="eyebrow mb-1.5">Extracted fields</div>
            <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))" }}>
              {Object.entries(det.fields).map(([k, v]) => (
                <div key={k} className="rounded border p-2" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                  <div className="eyebrow">{k}</div>
                  <div className="text-[0.8125rem] font-semibold mt-0.5">{v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* three-way match */}
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <div className="eyebrow">Three-way match — PO ↔ Invoice ↔ Receipt</div>
              <Badge status={MATCH[det.match_status].status}>{MATCH[det.match_status].label}</Badge>
            </div>
            <DataTable head={<>
              <Th>Line</Th><Th num>PO qty</Th><Th num>Inv qty</Th><Th num>Rec qty</Th>
              <Th num>PO price</Th><Th num>Inv price</Th><Th>Match</Th>
            </>}>
              {det.lines.map((ln) => (
                <tr key={ln.sku}>
                  <Td strong>{ln.sku}<div className="text-[0.6875rem] text-ink-3 font-normal">{ln.description}</div></Td>
                  <Td num>{ln.po_qty}</Td>
                  <Td num><span style={ln.invoice_qty !== ln.po_qty ? { color: "var(--critical)", fontWeight: 600 } : undefined}>{ln.invoice_qty}</span></Td>
                  <Td num>{ln.receipt_qty}</Td>
                  <Td num>${ln.po_price.toFixed(2)}</Td>
                  <Td num><span style={Math.abs(ln.invoice_price - ln.po_price) >= 0.01 ? { color: "var(--critical)", fontWeight: 600 } : undefined}>${ln.invoice_price.toFixed(2)}</span></Td>
                  <Td>{ln.status === "matched" ? <span style={{ color: "var(--good)" }}>✓</span> : <Badge status="critical">mismatch</Badge>}</Td>
                </tr>
              ))}
            </DataTable>
          </div>

          {det.discrepancies.length > 0 && (
            <Alert status="warning" title={`${det.discrepancies.length} discrepancy(ies) found`}>
              <ul className="list-disc ml-4 mt-1">{det.discrepancies.map((x, i) => <li key={i}>{x}</li>)}</ul>
            </Alert>
          )}
          <div className="text-[0.8125rem] text-ink-2">
            <span className="text-ink-3">Recommended:</span> {det.recommended_action}
          </div>
        </div>
      )}
    </Modal>
  );
}
