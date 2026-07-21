"use client";
/**
 * frontend/auth/guard.tsx — protects the app shell. Unauthenticated users are
 * redirected to /login; unknown/forbidden routes show an access notice.
 */
import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { ReactNode } from "react";
import { useAuth } from "@/auth/context";

export function RouteGuard({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!loading && !user && pathname !== "/login") router.replace("/login");
  }, [loading, user, pathname, router]);

  if (loading) {
    return <div className="min-h-screen grid place-items-center text-ink-3 text-sm">Loading…</div>;
  }
  if (!user && pathname !== "/login") return null;
  return <>{children}</>;
}
