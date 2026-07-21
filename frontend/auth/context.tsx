"use client";
/**
 * frontend/auth/context.tsx — the auth context: current user, login, logout, and
 * permission checks. Restores the session on mount (from persisted tokens) so a
 * reload keeps the user logged in.
 */
import { createContext, useCallback, useContext, useEffect, useState, ReactNode } from "react";
import { api } from "@/lib/api";
import { AuthUser, tokenStore } from "@/auth/store";

interface AuthState {
  user: AuthUser | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  can: (permission: string) => boolean;
}

const Ctx = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Restore session: trust the cached user, then verify against /me.
    const cached = tokenStore.user();
    if (cached) setUser(cached);
    if (tokenStore.access()) {
      api.me().then(setUser).catch(() => { tokenStore.clear(); setUser(null); })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const t = await api.login(email, password);
    tokenStore.set(t.access_token, t.refresh_token, t.user);
    setUser(t.user);
  }, []);

  const logout = useCallback(async () => {
    try { await api.logout(); } catch { /* ignore */ }
    tokenStore.clear();
    setUser(null);
    if (typeof window !== "undefined") window.location.href = "/login";
  }, []);

  const can = useCallback((permission: string) => !!user?.permissions.includes(permission), [user]);

  return <Ctx.Provider value={{ user, loading, login, logout, can }}>{children}</Ctx.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
