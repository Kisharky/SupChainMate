"""
ai/providers/base.py
The Provider interface (interface segregation + dependency inversion).

The router depends on this abstraction, never on a concrete SDK. A new
provider (Bedrock, Vertex, OpenAI, a local vLLM) is added by implementing
this Protocol and registering it — no change to the router or the agents.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai.types import AIResponse, EmbeddingResponse, Message, ModelSpec


@runtime_checkable
class Provider(Protocol):
    """A model-serving backend."""

    name: str

    def is_configured(self, spec: ModelSpec) -> bool:
        """True when this spec's credentials/model are available to call."""
        ...

    def chat(self, messages: list[Message], spec: ModelSpec) -> AIResponse:
        """Run a chat/reasoning completion. Must not raise — return
        AIResponse.failure(...) on error so the router can fall back."""
        ...

    def embed(self, texts: list[str], spec: ModelSpec,
              input_type: str = "passage") -> EmbeddingResponse:
        """Embed texts. Must not raise — return EmbeddingResponse(ok=False)."""
        ...
