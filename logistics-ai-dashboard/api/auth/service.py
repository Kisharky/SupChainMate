"""
api/auth/service.py — identity use-cases (authenticate, issue/rotate tokens, seed
default users). Pure application logic over the ORM; no HTTP concerns here.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.auth import security
from api.auth.models import RefreshToken, User
from api.auth.rbac import Role

# Demo accounts seeded on first boot (offline/portfolio mode). Passwords come from
# env with a sane default; override DEMO_PASSWORD in real deployments.
_DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "supchain123")
_SEED_USERS = [
    ("admin@supchainmate.io", "Admin User", Role.ADMIN),
    ("exec@supchainmate.io", "Executive User", Role.EXECUTIVE),
    ("scm@supchainmate.io", "Supply Chain Manager", Role.SUPPLY_CHAIN_MANAGER),
    ("planner@supchainmate.io", "Planner User", Role.PLANNER),
    ("warehouse@supchainmate.io", "Warehouse Manager", Role.WAREHOUSE_MANAGER),
    ("viewer@supchainmate.io", "Read Only User", Role.READ_ONLY),
]


def get_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.lower()))


def authenticate(db: Session, email: str, password: str) -> User | None:
    user = get_by_email(db, email)
    if user and user.is_active and security.verify_password(password, user.password_hash):
        return user
    return None


def issue_tokens(db: Session, user: User) -> tuple[str, str, int]:
    access = security.create_access_token(user.email, user.role)
    refresh, jti, exp = security.create_refresh_token(user.email)
    db.add(RefreshToken(jti=jti, user_id=user.id,
                        expires_at=datetime.fromtimestamp(exp, tz=timezone.utc)))
    db.commit()
    return access, refresh, security.ACCESS_TTL_SECONDS


def rotate_refresh(db: Session, refresh_token: str) -> tuple[User, str, str, int] | None:
    payload = security.decode_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        return None
    rec = db.scalar(select(RefreshToken).where(RefreshToken.jti == payload["jti"]))
    if not rec or rec.revoked:
        return None
    user = get_by_email(db, payload["sub"])
    if not user:
        return None
    rec.revoked = True                      # rotate: single-use refresh tokens
    access, new_refresh, ttl = issue_tokens(db, user)
    db.commit()
    return user, access, new_refresh, ttl


def revoke_all(db: Session, user: User) -> None:
    for rec in db.scalars(select(RefreshToken).where(
            RefreshToken.user_id == user.id, RefreshToken.revoked == False)):  # noqa: E712
        rec.revoked = True
    db.commit()


def seed_default_users(db: Session) -> int:
    created = 0
    for email, name, role in _SEED_USERS:
        if get_by_email(db, email) is None:
            db.add(User(email=email.lower(), name=name, role=role.value,
                        password_hash=security.hash_password(_DEMO_PASSWORD)))
            created += 1
    if created:
        db.commit()
    return created
