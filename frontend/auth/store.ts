/**
 * frontend/auth/store.ts — token + session persistence in localStorage so the
 * session survives reloads. Read synchronously by the API client to attach the
 * bearer token; mutated by the AuthProvider on login / refresh / logout.
 */
export interface AuthUser {
  id: number;
  email: string;
  name: string;
  role: string;
  permissions: string[];
}

const ACCESS = "scm.access";
const REFRESH = "scm.refresh";
const USER = "scm.user";

const isBrowser = typeof window !== "undefined";

export const tokenStore = {
  access: (): string | null => (isBrowser ? localStorage.getItem(ACCESS) : null),
  refresh: (): string | null => (isBrowser ? localStorage.getItem(REFRESH) : null),
  user: (): AuthUser | null => {
    if (!isBrowser) return null;
    const raw = localStorage.getItem(USER);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  },
  set(access: string, refresh: string, user: AuthUser) {
    if (!isBrowser) return;
    localStorage.setItem(ACCESS, access);
    localStorage.setItem(REFRESH, refresh);
    localStorage.setItem(USER, JSON.stringify(user));
  },
  setAccess(access: string) {
    if (isBrowser) localStorage.setItem(ACCESS, access);
  },
  clear() {
    if (!isBrowser) return;
    [ACCESS, REFRESH, USER].forEach((k) => localStorage.removeItem(k));
  },
};
