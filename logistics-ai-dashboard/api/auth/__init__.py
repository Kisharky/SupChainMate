"""
api/auth/ — enterprise authentication & RBAC, isolated from all business logic.

JWT (HS256) access + rotating refresh tokens, PBKDF2 password hashing, six RBAC
roles, and a central path-based auth/permission gate enforced in middleware so
the existing domain routes are never modified.
"""
