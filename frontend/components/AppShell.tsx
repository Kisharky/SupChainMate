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

type NavItem = { label: string; href: string; icon: string; badge?: number; perm: string };

// Grouped IA — same destinations, organised into scannable clusters (the flat
// 18-item rail was overwhelming). Groups with no visible items are hidden, and
// role-based permissions still filter each item.
const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  { label: "", items: [
    { label: "Dashboard", href: "/", icon: "◧", perm: "dashboard" },
  ] },
  { label: "Decisions & AI", items: [
    { label: "Intelligence", href: "/workspace", icon: "◆", perm: "intelligence" },
    { label: "Decisions", href: "/decisions", icon: "◇", perm: "decisions" },
    { label: "Workforce", href: "/workforce", icon: "❖", perm: "intelligence" },
    { label: "Knowledge", href: "/knowledge", icon: "◍", perm: "knowledge" },
  ] },
  { label: "Supply Chain", items: [
    { label: "Operations", href: "/operations", icon: "⬒", perm: "operations" },
    { label: "Forecasting", href: "/forecasting", icon: "◡", perm: "forecasting" },
    { label: "Inventory", href: "/inventory", icon: "▦", badge: 7, perm: "inventory" },
    { label: "Procurement", href: "/procurement", icon: "◈", perm: "procurement" },
    { label: "Warehouse", href: "/warehouse", icon: "▤", perm: "warehouse" },
    { label: "Logistics", href: "/logistics", icon: "◎", badge: 3, perm: "logistics" },
    { label: "Freight Ops", href: "/freight", icon: "⛁", perm: "operations" },
  ] },
  { label: "Commercial", items: [
    { label: "Commercial", href: "/commercial", icon: "◆", perm: "commercial" },
    { label: "Customers", href: "/customers", icon: "◐", perm: "commercial" },
    { label: "Documents", href: "/documents", icon: "❑", perm: "operations" },
  ] },
  { label: "Risk & Trust", items: [
    { label: "Risk Radar", href: "/radar", icon: "◉", perm: "operations" },
    { label: "Fraud & Risk", href: "/fraud", icon: "⚑", perm: "operations" },
  ] },
  { label: "Insights", items: [
    { label: "Reports", href: "/reports", icon: "▥", perm: "reports" },
  ] },
];

function NavLink({ item, active }: { item: NavItem; active: boolean }) {
  return (
    <Link href={item.href}
      className="flex items-center gap-3 rounded-sm px-2.5 py-2 text-[0.8125rem] font-medium border transition"
      style={active
        ? { background: "color-mix(in srgb,var(--accent) 14%,transparent)", color: "var(--text)", borderColor: "color-mix(in srgb,var(--accent) 34%,transparent)" }
        : { color: "var(--text-2)", borderColor: "transparent" }}>
      <span className="w-4 text-center text-[13px]" style={{ color: active ? "var(--accent)" : "var(--text-3)" }}>{item.icon}</span>
      {item.label}
      {item.badge && <span className="ml-auto rounded-full bg-critical px-1.5 text-[10px] font-bold text-white">{item.badge}</span>}
    </Link>
  );
}

export function AppShell({ title, children }: { title: string; children: ReactNode }) {
  const pathname = usePathname();
  const { user, can, logout } = useAuth();
  const [theme, setTheme] = useState<"dark" | "light" | null>(null);
  // Role-based navigation: only show groups/items the user is permitted to see.
  const groups = NAV_GROUPS
    .map((g) => ({ ...g, items: user ? g.items.filter((n) => can(n.perm)) : g.items }))
    .filter((g) => g.items.length > 0);
  const canAdmin = !user || can("administration");
  const canData = !user || can("data_hub");

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
        {groups.map((g, gi) => (
          <div key={g.label || `g${gi}`} className="flex flex-col gap-0.5">
            {g.label && (
              <div className="px-2.5 pt-3 pb-1 text-[10px] uppercase tracking-[.14em] font-semibold text-ink-3">{g.label}</div>
            )}
            {g.items.map((n) => <NavLink key={n.href} item={n} active={pathname === n.href} />)}
          </div>
        ))}
        <div className="mt-auto pt-4 border-t flex flex-col gap-0.5" style={{ borderColor: "var(--hairline)" }}>
          <div className="px-2.5 pb-1 text-[10px] uppercase tracking-[.14em] font-semibold text-ink-3">Data &amp; Admin</div>
          {canData && <NavLink item={{ label: "Data Hub", href: "/data", icon: "⊞", perm: "data_hub" }} active={pathname === "/data"} />}
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
