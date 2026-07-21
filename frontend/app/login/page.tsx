"use client";
/** Login — enterprise sign-in. Demo accounts listed for the portfolio walkthrough. */
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/auth/context";

const DEMO = [
  ["Admin", "admin@supchainmate.io"],
  ["Executive", "exec@supchainmate.io"],
  ["Supply Chain Mgr", "scm@supchainmate.io"],
  ["Planner", "planner@supchainmate.io"],
  ["Warehouse Mgr", "warehouse@supchainmate.io"],
  ["Read Only", "viewer@supchainmate.io"],
];

export default function Login() {
  const { user, login } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("exec@supchainmate.io");
  const [password, setPassword] = useState("supchain123");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { if (user) router.replace("/"); }, [user, router]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setError(null);
    try { await login(email, password); router.replace("/"); }
    catch { setError("Invalid email or password."); }
    finally { setBusy(false); }
  };

  return (
    <div className="min-h-screen grid place-items-center px-4" style={{ background: "var(--bg)" }}>
      <div className="w-full max-w-[400px]">
        <div className="flex items-center gap-2.5 mb-6 justify-center">
          <div className="grid h-9 w-9 place-items-center rounded-md font-extrabold text-[17px]"
            style={{ background: "linear-gradient(145deg,var(--emerald-500),var(--emerald-600))", color: "var(--accent-ink)" }}>S</div>
          <div>
            <div className="font-bold tracking-tight text-[17px]">SupChainMate</div>
            <div className="text-[10px] uppercase tracking-[.14em] text-ink-3">Decision Intelligence</div>
          </div>
        </div>

        <form onSubmit={submit} className="rounded-lg border p-6 shadow-card" style={{ borderColor: "var(--hairline)", background: "var(--panel)" }}>
          <h1 className="text-[1.25rem] font-semibold mb-1">Sign in</h1>
          <p className="text-ink-3 text-[0.8125rem] mb-4">Access the enterprise control plane.</p>

          <label className="text-[0.75rem] font-semibold text-ink-2">Email</label>
          <input value={email} onChange={(e) => setEmail(e.target.value)} type="email" required
            className="w-full mt-1 mb-3 bg-[var(--bg-sunken)] border rounded-sm px-3 py-2 text-[0.875rem] text-ink outline-none"
            style={{ borderColor: "var(--hairline-strong)" }} />

          <label className="text-[0.75rem] font-semibold text-ink-2">Password</label>
          <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" required
            className="w-full mt-1 mb-4 bg-[var(--bg-sunken)] border rounded-sm px-3 py-2 text-[0.875rem] text-ink outline-none"
            style={{ borderColor: "var(--hairline-strong)" }} />

          {error && <div className="text-[0.8125rem] mb-3" style={{ color: "var(--critical)" }}>{error}</div>}

          <button type="submit" disabled={busy}
            className="w-full py-2.5 rounded-sm font-semibold text-[0.875rem] disabled:opacity-50"
            style={{ background: "var(--emerald-500)", color: "var(--accent-ink)" }}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <div className="mt-4 rounded-lg border p-3" style={{ borderColor: "var(--hairline)", background: "var(--panel-2)" }}>
          <div className="text-[10px] uppercase tracking-wider text-ink-3 mb-2">Demo accounts · password <b className="text-ink-2">supchain123</b></div>
          <div className="grid grid-cols-2 gap-1.5">
            {DEMO.map(([role, mail]) => (
              <button key={mail} onClick={() => { setEmail(mail); setPassword("supchain123"); }}
                className="text-left rounded border px-2 py-1.5 text-[0.6875rem]" style={{ borderColor: "var(--hairline)" }}>
                <div className="text-ink font-medium">{role}</div>
                <div className="text-ink-3 truncate">{mail}</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
