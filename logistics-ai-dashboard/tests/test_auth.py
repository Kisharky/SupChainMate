"""Enterprise auth + RBAC tests. Enables the auth gate explicitly."""

import os

import pytest
from fastapi.testclient import TestClient

from api.auth import security
from api.auth.rbac import Role, has_permission, permissions_for


@pytest.fixture()
def client():
    os.environ["AUTH_ENABLED"] = "true"
    os.environ["JWT_SECRET"] = "test-secret"
    import api.main as m
    with TestClient(m.app) as c:      # startup seeds demo users
        yield c
    os.environ["AUTH_ENABLED"] = "false"


def _login(c, email, pw="supchain123"):
    return c.post("/api/auth/login", json={"email": email, "password": pw})


# ---- password + token unit tests ----
def test_password_hash_roundtrip():
    h = security.hash_password("s3cret!")
    assert h.startswith("pbkdf2_sha256$")
    assert security.verify_password("s3cret!", h)
    assert not security.verify_password("wrong", h)


def test_jwt_roundtrip_and_tamper():
    tok = security.create_access_token("a@b.io", "Admin")
    claims = security.decode_token(tok)
    assert claims["sub"] == "a@b.io" and claims["role"] == "Admin" and claims["type"] == "access"
    assert security.decode_token(tok + "x") is None       # tampered signature rejected


def test_rbac_matrix():
    assert has_permission(Role.ADMIN.value, "administration")
    assert not has_permission(Role.WAREHOUSE_MANAGER.value, "commercial")
    assert not has_permission(Role.WAREHOUSE_MANAGER.value, "intelligence")
    assert "approve" in permissions_for(Role.EXECUTIVE.value)


# ---- integration ----
def test_unauthenticated_is_rejected(client):
    assert client.get("/api/kpis").status_code == 401
    assert client.get("/api/health").status_code == 200      # public


def test_login_and_access(client):
    r = _login(client, "exec@supchainmate.io")
    assert r.status_code == 200
    body = r.json()
    assert body["user"]["role"] == "Executive"
    h = {"Authorization": f"Bearer {body['access_token']}"}
    assert client.get("/api/kpis", headers=h).status_code == 200
    assert client.get("/api/auth/me", headers=h).json()["email"] == "exec@supchainmate.io"


def test_rbac_forbids_wrong_role(client):
    r = _login(client, "warehouse@supchainmate.io")
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert client.get("/api/warehouse", headers=h).status_code == 200
    assert client.get("/api/commercial/brief", headers=h).status_code == 403
    assert client.get("/api/workspace/brief", headers=h).status_code == 403
    assert client.get("/api/workers", headers=h).status_code == 403  # intelligence-gated
    assert client.get("/api/fraud", headers=h).status_code == 403     # operations-gated
    assert client.get("/api/documents", headers=h).status_code == 403  # operations-gated
    assert client.get("/api/freight", headers=h).status_code == 403     # operations-gated


def test_connectors_require_admin(client):
    # Read-only cannot reach the admin-only connectors surface.
    viewer = _login(client, "viewer@supchainmate.io")
    hv = {"Authorization": f"Bearer {viewer.json()['access_token']}"}
    assert client.get("/api/connectors", headers=hv).status_code == 403
    # Admin can.
    admin = _login(client, "admin@supchainmate.io")
    ha = {"Authorization": f"Bearer {admin.json()['access_token']}"}
    assert client.get("/api/connectors", headers=ha).status_code == 200


def test_refresh_rotation_is_single_use(client):
    r = _login(client, "admin@supchainmate.io")
    refresh = r.json()["refresh_token"]
    assert client.post("/api/auth/refresh", json={"refresh_token": refresh}).status_code == 200
    # old refresh token is now revoked (rotated)
    assert client.post("/api/auth/refresh", json={"refresh_token": refresh}).status_code == 401


def test_bad_password_rejected(client):
    assert _login(client, "admin@supchainmate.io", "nope").status_code == 401
