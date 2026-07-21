"""
api/auth/rbac.py — role-based access control. Six enterprise roles map to a set
of permissions; endpoints and navigation are gated on permissions, never on the
role name directly (open/closed principle — add a role by editing this matrix
only).
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    ADMIN = "Admin"
    EXECUTIVE = "Executive"
    SUPPLY_CHAIN_MANAGER = "Supply Chain Manager"
    PLANNER = "Planner"
    WAREHOUSE_MANAGER = "Warehouse Manager"
    READ_ONLY = "Read Only"


class Permission(str, Enum):
    DASHBOARD = "dashboard"
    INTELLIGENCE = "intelligence"       # Decision Intelligence workspace (executive)
    OPERATIONS = "operations"
    FORECASTING = "forecasting"
    INVENTORY = "inventory"
    PROCUREMENT = "procurement"
    COMMERCIAL = "commercial"           # Commercial Intelligence (executive)
    WAREHOUSE = "warehouse"
    LOGISTICS = "logistics"
    DECISIONS = "decisions"
    KNOWLEDGE = "knowledge"
    REPORTS = "reports"                 # executive reports
    ADMINISTRATION = "administration"
    PLANNER = "planner"
    APPROVE = "approve"                 # act on decisions/recommendations


_ALL = set(Permission)

# Role → permissions. Warehouse Manager deliberately lacks intelligence /
# commercial / reports, so executive-only dashboards are invisible to them.
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: set(_ALL),
    Role.EXECUTIVE: {
        Permission.DASHBOARD, Permission.INTELLIGENCE, Permission.COMMERCIAL,
        Permission.DECISIONS, Permission.REPORTS, Permission.KNOWLEDGE,
        Permission.FORECASTING, Permission.INVENTORY, Permission.LOGISTICS,
        Permission.PROCUREMENT, Permission.OPERATIONS, Permission.WAREHOUSE,
        Permission.PLANNER, Permission.APPROVE,
    },
    Role.SUPPLY_CHAIN_MANAGER: {
        Permission.DASHBOARD, Permission.INTELLIGENCE, Permission.OPERATIONS,
        Permission.FORECASTING, Permission.INVENTORY, Permission.PROCUREMENT,
        Permission.WAREHOUSE, Permission.LOGISTICS, Permission.DECISIONS,
        Permission.KNOWLEDGE, Permission.REPORTS, Permission.PLANNER, Permission.APPROVE,
    },
    Role.PLANNER: {
        Permission.DASHBOARD, Permission.FORECASTING, Permission.INVENTORY,
        Permission.PLANNER, Permission.KNOWLEDGE, Permission.DECISIONS,
        Permission.INTELLIGENCE,
    },
    Role.WAREHOUSE_MANAGER: {
        Permission.DASHBOARD, Permission.WAREHOUSE, Permission.INVENTORY,
        Permission.LOGISTICS, Permission.KNOWLEDGE,
    },
    Role.READ_ONLY: {
        Permission.DASHBOARD, Permission.FORECASTING, Permission.INVENTORY,
        Permission.LOGISTICS, Permission.REPORTS, Permission.KNOWLEDGE,
        Permission.COMMERCIAL,
    },
}


def permissions_for(role: str) -> list[str]:
    try:
        return sorted(p.value for p in ROLE_PERMISSIONS[Role(role)])
    except (KeyError, ValueError):
        return []


def has_permission(role: str, permission: str) -> bool:
    try:
        return Permission(permission) in ROLE_PERMISSIONS.get(Role(role), set())
    except ValueError:
        return False


# Path-prefix → required permission (RBAC enforced centrally in the auth
# middleware, so existing route handlers stay untouched). Exact paths take
# precedence over prefixes.
EXACT_PERMISSIONS: dict[str, Permission] = {
    "/api/decisions/decide": Permission.APPROVE,
    "/api/commercial/decide": Permission.APPROVE,
    "/api/commercial/invoice": Permission.APPROVE,
}
PREFIX_PERMISSIONS: list[tuple[str, Permission]] = [
    ("/api/admin", Permission.ADMINISTRATION),
    ("/api/connectors", Permission.ADMINISTRATION),
    ("/api/commercial", Permission.COMMERCIAL),
    ("/api/workspace", Permission.INTELLIGENCE),
    ("/api/workers", Permission.INTELLIGENCE),
    ("/api/fraud", Permission.OPERATIONS),
    ("/api/documents", Permission.OPERATIONS),
    ("/api/freight", Permission.OPERATIONS),
    ("/api/planner", Permission.PLANNER),
    ("/api/reports", Permission.REPORTS),
]


def required_permission(path: str) -> str | None:
    if path in EXACT_PERMISSIONS:
        return EXACT_PERMISSIONS[path].value
    for prefix, perm in PREFIX_PERMISSIONS:
        if path.startswith(prefix):
            return perm.value
    return None
