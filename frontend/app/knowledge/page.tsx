"use client";
/** Knowledge Center — RAG interface backed by ai/rag.py (live). */
import { useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, CardHead, Button, Badge } from "@/components/ui/primitives";
import { api, KnowledgeAnswer } from "@/lib/api";

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
    </AppShell>
  );
}
