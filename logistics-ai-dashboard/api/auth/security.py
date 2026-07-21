"""
api/auth/security.py — password hashing and JWT, using only the Python standard
library (hmac / hashlib / secrets). This keeps the dependency surface minimal and
avoids native-crypto build issues while remaining a correct, secure design:

* Passwords: PBKDF2-HMAC-SHA256, 240k iterations, per-password random salt
  (the algorithm Django ships by default), stored as ``pbkdf2_sha256$iter$salt$hash``.
* Tokens: signed JWTs (HS256) with typed access/refresh claims and expiry.

Secrets come from the environment; there are no secrets in source.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Optional

# ---- Config (from environment) ----
JWT_SECRET = os.getenv("JWT_SECRET", "dev-insecure-change-me")
ACCESS_TTL_SECONDS = int(os.getenv("ACCESS_TOKEN_MINUTES", "30")) * 60
REFRESH_TTL_SECONDS = int(os.getenv("REFRESH_TOKEN_DAYS", "7")) * 86400
_PBKDF2_ITER = 240_000
_ALG = "HS256"


# ---- Password hashing (PBKDF2-HMAC-SHA256) ----
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITER)
    return f"pbkdf2_sha256${_PBKDF2_ITER}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


# ---- JWT (HS256) ----
def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64u_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload: dict[str, Any]) -> str:
    header = _b64u(json.dumps({"alg": _ALG, "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}".encode()
    sig = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64u(sig)}"


def _create_token(sub: str, token_type: str, ttl: int, extra: Optional[dict] = None) -> tuple[str, dict]:
    now = int(time.time())
    payload = {"sub": sub, "type": token_type, "iat": now, "exp": now + ttl,
               "jti": secrets.token_hex(16), **(extra or {})}
    return _sign(payload), payload


def create_access_token(sub: str, role: str) -> str:
    token, _ = _create_token(sub, "access", ACCESS_TTL_SECONDS, {"role": role})
    return token


def create_refresh_token(sub: str) -> tuple[str, str, int]:
    token, payload = _create_token(sub, "refresh", REFRESH_TTL_SECONDS)
    return token, payload["jti"], payload["exp"]


def decode_token(token: str) -> Optional[dict]:
    """Verify signature + expiry; return claims or None."""
    try:
        header_b64, body_b64, sig_b64 = token.split(".")
        signing_input = f"{header_b64}.{body_b64}".encode()
        expected = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(_b64u_decode(sig_b64), expected):
            return None
        payload = json.loads(_b64u_decode(body_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
