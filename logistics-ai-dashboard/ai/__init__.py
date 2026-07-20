"""
ai — provider-agnostic AI layer for SupChainMate.

Architecture (dependencies point inward):

    Business Logic (deterministic)  →  Agents  →  ai.AI (Router)
                                                     ↓
                                            Capability Registry
                                                     ↓
                                              Model Providers  →  NVIDIA NIM

Agents call `AI.ask(capability=..., task=..., context=...)` and never learn
which model answered. Only the router resolves capability → model.
"""

from ai.router import AI, AIRouter, build_default_router  # noqa: F401
from ai.registry import CapabilityRegistry  # noqa: F401
from ai.types import (AIResponse, Capability, EmbeddingResponse,  # noqa: F401
                      Message, ModelSpec)

__all__ = ["AI", "AIRouter", "build_default_router", "CapabilityRegistry",
           "Capability", "ModelSpec", "Message", "AIResponse", "EmbeddingResponse"]
