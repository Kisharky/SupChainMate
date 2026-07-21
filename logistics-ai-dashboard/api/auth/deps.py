"""
api/auth/deps.py — FastAPI dependencies + the central auth/RBAC gate.

Authentication and role checks are enforced once, in middleware, keyed on the
request path — so the existing domain routes are never touched. A per-request
principal is attached to ``request.state.user`` for handlers that want it.
"""

from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth import rbac, security

def auth_enabled() -> bool:
    """Read dynamically so deployment config (and tests) can toggle it."""
    return os.getenv("AUTH_ENABLED", "true").lower() == "true"


# Paths reachable without a token.
_PUBLIC = {"/api/health", "/api/auth/login", "/api/auth/refresh",
           "/docs", "/openapi.json", "/redoc", "/"}

_bearer = HTTPBearer(auto_error=False)


def _principal_from_token(token: str) -> dict | None:
    payload = security.decode_token(token)
    if not payload or payload.get("type") != "access":
        return None
    role = payload.get("role", "")
    return {"email": payload.get("sub"), "role": role,
            "permissions": rbac.permissions_for(role)}


def is_public(path: str) -> bool:
    return path in _PUBLIC or path.startswith("/docs") or path.startswith("/static")


def enforce(request: Request) -> tuple[int, str] | None:
    """Return (status, detail) to reject, or None to allow. Used by middleware."""
    if not auth_enabled():
        return None
    path = request.url.path
    if request.method == "OPTIONS" or is_public(path) or not path.startswith("/api/"):
        return None
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    principal = _principal_from_token(token) if token else None
    if principal is None:
        return (401, "Not authenticated")
    request.state.user = principal
    needed = rbac.required_permission(path)
    if needed and needed not in principal["permissions"]:
        return (403, f"Requires '{needed}' permission")
    return None


def current_user(request: Request,
                 creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    """Handler dependency: the authenticated principal (or 401)."""
    principal = getattr(request.state, "user", None)
    if principal is None and creds is not None:
        principal = _principal_from_token(creds.credentials)
    if principal is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return principal
