"use client";
/** Knowledge Center — RAG Q&A + the Decision Brain (long-term memory). */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, Button, Badge } from "@/components/ui/primitives";
import { api, KnowledgeAnswer, BrainHit, BrainStats } from "@/lib/api";

const SUGGESTIONS = [
  "What is our expedite policy when a critical SKU drops below 3 days of cover?",
  "Supplier lead-time SLA?",
  "Safety-stock formula",
  "Returns policy AU",
];

export default function Knowledge() {
  const [q, setQ] = useState(SUGGESTIONS[0]);
  const [ans, setAns] = useState<KnowledgeAnswer | null>(null);
  const [loading, setLoading] = useState(false);

  const ask = async (query: string) => {
    setQ(query); setLoading(true);
    try { setAns(await api.knowledgeAsk(query)); }
    catch { setAns(null); }
    finally { setLoading(false); }
  };

  return (
    <AppShell title="Knowledge Center">
      <div className="mb-4">
        <div className="text-[0.75rem] uppercase tracking-[.16em] font-semibold" style={{ color: "var(--accent)" }}>Retrieval-augmented · cites sources</div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Knowledge Center</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Ask across your policies, contracts, and SOPs — answers cite the documents they came from.</p>
      </div>

      <div className="flex gap-2.5 items-center rounded border p-2 pl-3.5 shadow-card bg-panel" style={{ borderColor: "var(--hairline-strong)" }}>
        <span style={{ color: "var(--accent)", fontSize: 16 }}>◍</span>
        <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask(q)}
          className="flex-1 bg-transparent outline-none text-[0.9375rem] text-ink" placeholder="Ask a question…" />
        <Button variant="primary" sm onClick={() => ask(q)} disabled={loading}>{loading ? "…" : "Ask"}</Button>
      </div>
      <div className="flex gap-2.5 flex-wrap mt-2.5">
        {SUGGESTIONS.map((s) => <button key={s} onClick={() => ask(s)}><Badge status="neutral">{s}</Badge></button>)}
      </div>

      <Card className="mt-4">
        <CardHead title="Answer" hint={ans ? `${ans.retriever ?? "hybrid"} · ${ans.source}` : "ask to begin"} />
        <div className="p-[18px]">
          {loading && <p className="text-ink-3 text-[0.9375rem]">Retrieving…</p>}
          {!loading && ans && (
            <>
              <p className="text-[0.9375rem] leading-relaxed text-ink whitespace-pre-wrap">{ans.answer}</p>
              {ans.citations?.length > 0 && (
                <div className="flex gap-2 flex-wrap mt-3.5 pt-3 border-t" style={{ borderColor: "var(--hairline)" }}>
                  {ans.citations.map((c, i) => (
                    <span key={i} className="inline-flex items-center gap-1.5 rounded text-[11px] font-semibold px-1.5 py-0.5"
                      style={{ color: "var(--info)", background: "var(--info-bg)", border: "1px solid color-mix(in srgb,var(--info) 30%,transparent)" }}>
                      ◈ {c.source ?? c.name ?? c.ref ?? `Source ${i + 1}`}
                    </span>
                  ))}
                </div>
              )}
              {ans.confidence != null && (
                <div className="mt-3"><Badge status="good">Confidence · {ans.confidence}</Badge></div>
              )}
            </>
          )}
          {!loading && !ans && <p className="text-ink-3 text-[0.9375rem]">Ask a question above to see a grounded, cited answer.</p>}
        </div>
      </Card>

      <DecisionBrain />
    </AppShell>
  );
}

const KIND_COLOR: Record<string, "good" | "warning" | "critical" | "info" | "neutral"> = {
  decision: "info", knowledge: "good", recommendation: "warning", outcome: "good",
  feedback: "neutral", approval: "info", entity: "neutral",
};
const KINDS = ["decision", "knowledge", "recommendation", "approval", "feedback"];

function DecisionBrain() {
  const [q, setQ] = useState("expedite policy for critical SKUs");
  const [kind, setKind] = useState<string | null>(null);
  const [hits, setHits] = useState<BrainHit[] | null>(null);
  const [stats, setStats] = useState<BrainStats | null>(null);
  const [busy, setBusy] = useState(false);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");

  const loadStats = () => { api.brainStats().then(setStats).catch(() => {}); };
  useEffect(() => { loadStats(); }, []);

  const recall = async () => {
    setBusy(true);
    try { setHits((await api.brainRecall(q, kind ? [kind] : undefined)).results); }
    finally { setBusy(false); }
  };
  const remember = async () => {
    if (!title || !body) return;
    await api.brainRemember(title, body);
    setTitle(""); setBody(""); loadStats(); recall();
  };
  const ingest = async () => { await api.brainIngest(); loadStats(); };

  return (
    <Card className="mt-4">
      <CardHead title="Decision Brain — long-term memory"
        hint={stats ? `${stats.total} memories · ${stats.embedder} · offline` : "semantic memory"}
        right={<Button variant="ghost" sm onClick={ingest}>⟳ Sync existing</Button>} />
      <div className="p-[18px]">
        {/* memory stats by kind */}
        {stats && (
          <div className="flex gap-2 flex-wrap mb-3">
            {Object.entries(stats.by_kind).map(([k, n]) => (
              <span key={k} className="inline-flex items-center gap-1.5"><Badge status={KIND_COLOR[k] ?? "neutral"}>{k} · {n}</Badge></span>
            ))}
          </div>
        )}
        {/* recall */}
        <div className="flex gap-2 items-center rounded border p-1.5 pl-3" style={{ borderColor: "var(--hairline-strong)", background: "var(--bg-sunken)" }}>
          <span style={{ color: "var(--accent)" }}>◈</span>
          <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && recall()}
            className="flex-1 bg-transparent outline-none text-[0.875rem] text-ink" placeholder="Recall past decisions, knowledge, feedback…" />
          <Button variant="primary" sm onClick={recall} disabled={busy}>{busy ? "…" : "Recall"}</Button>
        </div>
        <div className="flex gap-1.5 flex-wrap mt-2">
          <button onClick={() => setKind(null)}><Badge status={kind === null ? "info" : "neutral"}>all</Badge></button>
          {KINDS.map((k) => <button key={k} onClick={() => setKind(k)}><Badge status={kind === k ? "info" : "neutral"}>{k}</Badge></button>)}
        </div>

        {hits && (
          <div className="mt-3 flex flex-col gap-2">
            {hits.map((h) => (
              <div key={h.id} className="rounded border p-2.5" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
                <div className="flex items-center justify-between gap-2">
                  <div className="text-[0.8125rem] font-semibold flex items-center gap-2"><Badge status={KIND_COLOR[h.kind] ?? "neutral"}>{h.kind}</Badge>{h.title}</div>
                  <span className="text-[0.6875rem] text-ink-3 tnum">score {h.score.toFixed(2)} · sem {h.semantic.toFixed(2)}</span>
                </div>
                <div className="text-[0.75rem] text-ink-3 mt-1">{h.snippet}</div>
              </div>
            ))}
            {hits.length === 0 && <p className="text-ink-3 text-[0.8125rem]">No memory matched — sync existing state or add knowledge below.</p>}
          </div>
        )}

        {/* add to memory */}
        <div className="mt-4 pt-3 border-t" style={{ borderColor: "var(--hairline)" }}>
          <div className="eyebrow mb-2">Teach the Brain</div>
          <div className="flex gap-2 flex-wrap">
            <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Title (e.g. SOP §3.1)"
              className="bg-[var(--bg-sunken)] border rounded-sm px-3 py-2 text-[0.8125rem] text-ink outline-none" style={{ borderColor: "var(--hairline-strong)", width: 200 }} />
            <input value={body} onChange={(e) => setBody(e.target.value)} onKeyDown={(e) => e.key === "Enter" && remember()} placeholder="Knowledge, policy, contract term, note…"
              className="flex-1 bg-[var(--bg-sunken)] border rounded-sm px-3 py-2 text-[0.8125rem] text-ink outline-none" style={{ borderColor: "var(--hairline-strong)", minWidth: 240 }} />
            <Button variant="secondary" sm onClick={remember}>+ Remember</Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
