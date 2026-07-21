"""api/auth/router.py — the authentication HTTP API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.auth import service
from api.auth.deps import current_user
from api.auth.rbac import permissions_for
from api.auth.schemas import (LoginRequest, RefreshRequest, TokenResponse, UserOut)
from api.db import get_session

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _user_out(user) -> UserOut:
    return UserOut(id=user.id, email=user.email, name=user.name, role=user.role,
                   permissions=permissions_for(user.role))


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_session)) -> TokenResponse:
    user = service.authenticate(db, req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    access, refresh, ttl = service.issue_tokens(db, user)
    return TokenResponse(access_token=access, refresh_token=refresh,
                         expires_in=ttl, user=_user_out(user))


@router.post("/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_session)) -> TokenResponse:
    result = service.rotate_refresh(db, req.refresh_token)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    user, access, new_refresh, ttl = result
    return TokenResponse(access_token=access, refresh_token=new_refresh,
                         expires_in=ttl, user=_user_out(user))


@router.post("/logout")
def logout(principal: dict = Depends(current_user), db: Session = Depends(get_session)) -> dict:
    user = service.get_by_email(db, principal["email"])
    if user:
        service.revoke_all(db, user)        # revoke all refresh tokens
    return {"ok": True}


@router.get("/me", response_model=UserOut)
def me(principal: dict = Depends(current_user), db: Session = Depends(get_session)) -> UserOut:
    user = service.get_by_email(db, principal["email"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_out(user)
