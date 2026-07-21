"use client";
/**
 * AdminNav — the Administration section's sub-navigation tab strip.
 * Shared by every Administration surface so the IA stays consistent:
 *   Users · Roles · Audit Logs · Settings · Connectors
 * Users/Roles/Audit/Settings anchor into the combined Administration page;
 * Connectors is its own enterprise workspace.
 */
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const TABS = [
  { label: "Users", href: "/administration#users", hash: "#users" },
  { label: "Roles", href: "/administration#roles", hash: "#roles" },
  { label: "Audit Logs", href: "/administration#audit", hash: "#audit" },
  { label: "Settings", href: "/administration#settings", hash: "#settings" },
  { label: "Connectors", href: "/administration/connectors", hash: "" },
];

export function AdminNav() {
  const pathname = usePathname();
  const [hash, setHash] = useState("#users");
  useEffect(() => {
    const sync = () => setHash(window.location.hash || "#users");
    sync();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, [pathname]);

  const isActive = (t: (typeof TABS)[number]) =>
    t.label === "Connectors"
      ? pathname === "/administration/connectors"
      : pathname === "/administration" && hash === t.hash;

  return (
    <nav className="flex flex-wrap gap-1 border-b mb-5" style={{ borderColor: "var(--hairline)" }}>
      {TABS.map((t) => {
        const active = isActive(t);
        return (
          <Link key={t.label} href={t.href} onClick={() => t.hash && setHash(t.hash)}
            className="relative px-3 py-2 text-[0.8125rem] font-medium transition -mb-px border-b-2"
            style={active
              ? { color: "var(--text)", borderColor: "var(--accent)" }
              : { color: "var(--text-3)", borderColor: "transparent" }}>
            {t.label}
          </Link>
        );
      })}
    </nav>
  );
}
