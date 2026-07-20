"""
ai/coding.py
Coding service — SQL / Python / workflow generation for internal developer
tools. Routes to the coding capability (deepseek-v4-flash, high reasoning
effort). Read-only by design: it generates code, it never executes it.
"""

from __future__ import annotations

from ai.router import AI
from ai.types import AIResponse, Capability

_SQL_SYSTEM = (
    "You generate a single, safe, read-only SQL SELECT query for the user's "
    "request against the described schema. No DDL/DML. Return only the SQL.")

_PY_SYSTEM = (
    "You generate concise, correct Python for the described data task using "
    "pandas. Return only the code, no prose.")


def generate_sql(request: str, schema: str) -> AIResponse:
    return AI.ask(Capability.CODING, "sql_generation", context=None,
                  system=_SQL_SYSTEM,
                  user=f"Schema:\n{schema}\n\nRequest: {request}")


def generate_python(request: str, context: str = "") -> AIResponse:
    return AI.ask(Capability.CODING, "python_generation", context=None,
                  system=_PY_SYSTEM,
                  user=f"Context:\n{context}\n\nTask: {request}")
