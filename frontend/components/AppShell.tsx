"use client";
/**
 * AppShell — the persistent command shell: 236px left nav rail + topbar.
 * One consistent brand ("SupChainMate") and one navigation across every
 * screen (the Stitch mockups drifted on both; this is the canonical IA).
 * Note: no standalone "AI" destination — AI lives inside the workflows.
 */
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ReactNode, useEffect, useState } from "react";
import { useAuth } from "@/auth/context";

const NAV: { label: string; href: string; icon: string; badge?: number; perm: string }[] = [
  { label: "Dashboard", href: "/", icon: "◧", perm: "dashboard" },
  { label: "Intelligence", href: "/workspace", icon: "◆", perm: "intelligence" },
  { label: "Workforce", href: "/workforce", icon: "❖", perm: "intelligence" },
  { label: "Operations", href: "/operations", icon: "⬒", perm: "operations" },
  { label: "Forecasting", href: "/forecasting", icon: "◡", perm: "forecasting" },
  { label: "Inventory", href: "/inventory", icon: "▦", badge: 7, perm: "inventory" },
  { label: "Procurement", href: "/procurement", icon: "◈", perm: "procurement" },
  { label: "Commercial", href: "/commercial", icon: "◆", perm: "commercial" },
  { label: "Warehouse", href: "/warehouse", icon: "▤", perm: "warehouse" },
  { label: "Logistics", href: "/logistics", icon: "◎", badge: 3, perm: "logistics" },
  { label: "Decisions", href: "/decisions", icon: "◇", perm: "decisions" },
  { label: "Fraud & Risk", href: "/fraud", icon: "⚑", perm: "operations" },
  { label: "Knowledge", href: "/knowledge", icon: "◍", perm: "knowledge" },
  { label: "Reports", href: "/reports", icon: "▥", perm: "reports" },
];

export function AppShell({ title, children }: { title: string; children: ReactNode }) {
  const pathname = usePathname();
  const { user, can, logout } = useAuth();
  const [theme, setTheme] = useState<"dark" | "light" | null>(null);
  // Role-based navigation: only show sections the user is permitted to see.
  const nav = user ? NAV.filter((n) => can(n.perm)) : NAV;
  const canAdmin = !user || can("administration");

  useEffect(() => {
    const saved = (localStorage.getItem("scm-theme") as "dark" | "light") || "dark";
    setTheme(saved);
    document.documentElement.setAttribute("data-theme", saved);
  }, []);
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("scm-theme", next);
  };

  return (
    <div className="grid min-h-screen" style={{ gridTemplateColumns: "236px 1fr" }}>
      <aside className="sticky top-0 h-screen overflow-y-auto border-r p-3.5 flex flex-col gap-1 bg-rail"
        style={{ borderColor: "var(--hairline)" }}>
        <div className="flex items-center gap-2.5 px-2 pt-1 pb-4">
          <div className="grid h-7 w-7 place-items-center rounded-md font-extrabold text-[15px]"
            style={{ background: "linear-gradient(145deg,var(--emerald-500),var(--emerald-600))", color: "var(--accent-ink)" }}>S</div>
          <div>
            <div className="font-bold tracking-tight text-[15px]">SupChainMate</div>
            <div className="text-[10px] uppercase tracking-[.14em] text-ink-3">Decision Intelligence</div>
          </div>
        </div>
        {nav.map((n) => {
          const active = pathname === n.href;
          return (
            <Link key={n.href} href={n.href}
              className="flex items-center gap-3 rounded-sm px-2.5 py-2 text-[0.8125rem] font-medium border transition"
              style={active
                ? { background: "color-mix(in srgb,var(--accent) 14%,transparent)", color: "var(--text)", borderColor: "color-mix(in srgb,var(--accent) 34%,transparent)" }
                : { color: "var(--text-2)", borderColor: "transparent" }}>
              <span className="w-4 text-center text-[13px]" style={{ color: active ? "var(--accent)" : "var(--text-3)" }}>{n.icon}</span>
              {n.label}
              {n.badge && <span className="ml-auto rounded-full bg-critical px-1.5 text-[10px] font-bold text-white">{n.badge}</span>}
            </Link>
          );
        })}
        <div className="mt-auto pt-4 border-t" style={{ borderColor: "var(--hairline)" }}>
          {canAdmin && (() => {
            const inAdmin = pathname.startsWith("/administration");
            const subItems = [
              { label: "Users", href: "/administration#users" },
              { label: "Roles", href: "/administration#roles" },
              { label: "Audit Logs", href: "/administration#audit" },
              { label: "Settings", href: "/administration#settings" },
              { label: "Connectors", href: "/administration/connectors" },
            ];
            return (
              <div>
                <Link href="/administration"
                  className="flex items-center gap-3 rounded-sm px-2.5 py-2 text-[0.8125rem]"
                  style={{ color: inAdmin ? "var(--text)" : "var(--text-2)" }}>
                  <span className="w-4 text-center" style={{ color: inAdmin ? "var(--accent)" : "var(--text-3)" }}>⚙</span> Administration
                </Link>
                {inAdmin && (
                  <div className="ml-4 mt-0.5 flex flex-col gap-0.5 border-l pl-2.5" style={{ borderColor: "var(--hairline)" }}>
                    {subItems.map((s) => {
                      const active = s.href === "/administration/connectors" && pathname === "/administration/connectors";
                      return (
                        <Link key={s.label} href={s.href}
                          className="rounded-sm px-2 py-1.5 text-[0.75rem] transition"
                          style={{ color: active ? "var(--accent)" : "var(--text-3)" }}>
                          {s.label}
                        </Link>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })()}
          {user && (
            <div className="flex items-center gap-2 px-2.5 py-2">
              <div className="grid h-7 w-7 place-items-center rounded-full text-[11px] font-bold flex-none"
                style={{ background: "var(--panel-2)", color: "var(--text-2)", border: "1px solid var(--hairline)" }}>
                {user.name.split(" ").map((w) => w[0]).slice(0, 2).join("")}
              </div>
              <div className="min-w-0">
                <div className="text-[0.75rem] text-ink font-medium truncate">{user.name}</div>
                <div className="text-[10px] text-ink-3 truncate">{user.role}</div>
              </div>
              <button onClick={logout} title="Sign out" className="ml-auto text-ink-3 hover:text-ink text-[13px]">⏻</button>
            </div>
          )}
          <div className="flex items-center gap-3 px-2.5 py-1 text-[11px] text-ink-3">
            <span style={{ color: "var(--good)" }}>●</span> AI Router · operational
          </div>
        </div>
      </aside>

      <div className="min-w-0 flex flex-col">
        <div className="sticky top-0 z-10 flex items-center justify-between gap-4 border-b px-6 py-3 backdrop-blur"
          style={{ borderColor: "var(--hairline)", background: "color-mix(in srgb,var(--bg) 82%,transparent)" }}>
          <div className="flex items-center gap-2 text-[0.8125rem] text-ink-3">
            SupChainMate <span>/</span> <b className="text-ink font-semibold">{title}</b>
          </div>
          <button onClick={toggle}
            className="inline-flex items-center gap-2 rounded-sm border bg-panel px-3 py-1.5 text-[0.8125rem] font-semibold text-ink-2"
            style={{ borderColor: "var(--hairline)" }}>◑ Theme</button>
        </div>
        <div className="mx-auto w-full max-w-[1280px] px-6 pb-16 pt-5">{children}</div>
      </div>
    </div>
  );
}
