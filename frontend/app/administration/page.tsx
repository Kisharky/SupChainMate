"use client";
/** Administration — users, roles, API keys (masked), AI providers, ERP, audit. */
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { AdminNav } from "@/components/AdminNav";
import { Card, CardHead, Badge, DataTable, Th, Td } from "@/components/ui/primitives";
import { api, AdminResponse } from "@/lib/api";

export default function Administration() {
  const [d, setD] = useState<AdminResponse | null>(null);
  useEffect(() => { api.admin().then(setD).catch(() => setD(null)); }, []);
  const check = (b: boolean) => <span style={{ color: b ? "var(--good)" : "var(--text-3)" }}>{b ? "✓" : "—"}</span>;

  return (
    <AppShell title="Administration">
      <div className="mb-4">
        <div className="eyebrow" style={{ color: "var(--accent)" }}>System governance · {d?.source === "live" ? "live" : "loading"}</div>
        <h1 className="text-[2.25rem] tracking-tight leading-none mt-2 font-semibold">Administration</h1>
        <p className="text-ink-2 mt-1.5 text-[0.9375rem]">Users, roles, credentials, AI providers, and integrations — with a full audit trail.</p>
      </div>

      <AdminNav />

      <div className="grid gap-4 items-start" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Card id="users" className="scroll-mt-24">
          <CardHead title="Users" hint={`${d?.users.length ?? 0} members`} />
          <DataTable head={<><Th>Name</Th><Th>Role</Th><Th>Status</Th></>}>
            {(d?.users ?? []).map((u) => (
              <tr key={u.email}>
                <Td strong>{u.name}<div className="text-[0.6875rem] text-ink-3 font-normal">{u.email}</div></Td>
                <Td>{u.role}</Td>
                <Td><Badge status={u.status === "active" ? "good" : "warning"}>{u.status}</Badge></Td>
              </tr>
            ))}
          </DataTable>
        </Card>

        <Card id="roles" className="scroll-mt-24">
          <CardHead title="Roles & permissions" hint="RBAC" />
          <DataTable head={<><Th>Role</Th><Th>View</Th><Th>Run</Th><Th>Approve</Th><Th>Admin</Th></>}>
            {(d?.roles ?? []).map((r) => (
              <tr key={r.role}>
                <Td strong>{r.role}</Td>
                <Td>{check(r.view)}</Td><Td>{check(r.run)}</Td><Td>{check(r.approve)}</Td><Td>{check(r.admin)}</Td>
              </tr>
            ))}
          </DataTable>
        </Card>

        <Card id="settings" className="scroll-mt-24">
          <CardHead title="API keys" hint="values never exposed" />
          <DataTable head={<><Th>Key</Th><Th>Purpose</Th><Th>Status</Th></>}>
            {(d?.api_keys ?? []).map((k) => (
              <tr key={k.name}>
                <Td strong><span className="font-mono text-[0.75rem]">{k.name}</span></Td>
                <Td>{k.purpose}</Td>
                <Td>{k.configured
                  ? <span className="inline-flex items-center gap-2"><Badge status="good">configured</Badge><span className="font-mono text-ink-3">{k.masked}</span></span>
                  : <Badge status="neutral">not set</Badge>}</Td>
              </tr>
            ))}
          </DataTable>
        </Card>

        <Card>
          <CardHead title="AI providers" hint="capability → model" />
          <DataTable head={<><Th>Capability</Th><Th>Model</Th><Th>Status</Th></>}>
            {(d?.providers ?? []).map((p) => (
              <tr key={p.capability}>
                <Td strong>{p.capability}</Td>
                <Td><span className="font-mono text-[0.75rem]">{p.model}</span></Td>
                <Td><Badge status={p.configured ? "good" : "warning"}>{p.configured ? "ready" : "no key"}</Badge></Td>
              </tr>
            ))}
          </DataTable>
        </Card>
      </div>

      <div className="grid gap-4 mt-4 items-start" style={{ gridTemplateColumns: "1fr 1.4fr" }}>
        <Card>
          <CardHead title="ERP & WMS integrations" />
          <div className="p-[18px] flex flex-col gap-2">
            {(d?.integrations ?? []).map((it) => (
              <div key={it.name} className="flex items-center justify-between rounded border p-2.5 bg-[var(--panel-2)]" style={{ borderColor: "var(--hairline)" }}>
                <div><span className="text-[0.875rem] font-medium">{it.name}</span> <span className="text-[0.6875rem] text-ink-3 ml-1">{it.kind}</span></div>
                <Badge status="neutral">{it.status.replace("_", " ")}</Badge>
              </div>
            ))}
          </div>
        </Card>
        <Card id="audit" className="scroll-mt-24">
          <CardHead title="Audit log" hint="immutable" />
          <div className="px-[18px] py-1.5 max-h-[320px] overflow-y-auto">
            {(d?.audit ?? []).slice(0, 40).map((e, i) => (
              <div key={i} className="flex gap-3 py-2 border-b last:border-0 text-[0.75rem]" style={{ borderColor: "var(--hairline)" }}>
                <span className="font-mono text-ink-3 whitespace-nowrap">{e.ts?.slice(5, 16)}</span>
                <span className="text-ink-2"><b className="text-ink">{e.actor}</b> · {e.event.replace(/_/g, " ")}{e.details ? ` — ${e.details}` : ""}</span>
              </div>
            ))}
            {(!d?.audit || d.audit.length === 0) && <p className="text-ink-3 text-[0.8125rem] py-3">No audit entries yet.</p>}
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
