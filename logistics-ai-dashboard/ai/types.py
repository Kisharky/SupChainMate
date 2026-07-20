"""
ai/types.py
Shared, framework-free data contracts for the AI layer.

These dataclasses are the vocabulary every layer speaks. They never import
Streamlit, a provider SDK, or a business module — the dependency arrows all
point inward toward this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Capability(str, Enum):
    """Abstract capabilities agents request. Never a concrete model name."""
    REASONING_EXECUTIVE = "reasoning.executive"
    REASONING_OPERATIONS = "reasoning.operations"
    CODING = "coding"
    EMBEDDING = "embedding"
    VISION = "vision"
    OCR = "ocr"
    SAFETY = "safety"


@dataclass(frozen=True)
class ModelSpec:
    """How a capability is fulfilled by a concrete provider+model."""
    capability: Capability
    provider: str                       # e.g. "nvidia"
    model: str                          # e.g. "z-ai/glm-5.2"
    api_key_env: str                    # env var holding this model's key
    kind: str = "chat"                  # "chat" | "embedding"
    temperature: float = 0.4
    top_p: float = 0.95
    max_tokens: int = 4096
    extra_body: dict[str, Any] = field(default_factory=dict)
    fallback: Optional[Capability] = None   # capability to try if this fails


@dataclass
class Message:
    role: str
    content: str


@dataclass
class TokenUsage:
    """Token accounting from a provider, when reported."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class AIResponse:
    """The uniform result of every AI.ask() call."""
    text: str
    capability: str
    ok: bool = True
    model_used: Optional[str] = None
    provider: Optional[str] = None
    reasoning: Optional[str] = None         # chain-of-thought when the model emits it
    latency_ms: int = 0
    fell_back: bool = False                 # a fallback path served this
    cached: bool = False                    # served from the response cache
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: Optional[str] = None

    @classmethod
    def failure(cls, capability: str, error: str) -> "AIResponse":
        return cls(text="", capability=capability, ok=False, error=error)


@dataclass
class EmbeddingResponse:
    vectors: list[list[float]]
    model_used: Optional[str] = None
    ok: bool = True
    error: Optional[str] = None


class ProviderError(RuntimeError):
    """Raised by a provider when a call cannot be completed."""
