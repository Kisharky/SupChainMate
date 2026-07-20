"""
ai/registry.py
Capability Registry — the single source of truth mapping abstract
capabilities to concrete provider+model specs.

Agents ask for a Capability; only this registry knows which model answers
it. Swapping a model, re-pointing a capability, or adding a provider is a
one-line change here and nowhere else. Assignments follow the platform
model plan (NVIDIA NIM):

    embedding             → nemotron-3-embed-1b
    reasoning.executive   → nemotron-3-ultra-550b-a55b   (deep thinking)
    reasoning.operations  → z-ai/glm-5.2
    coding                → deepseek-v4-flash            (high reasoning effort)
    vision / ocr / safety → abstractions (no model wired yet)
"""

from __future__ import annotations

from typing import Optional

import config
from ai.types import Capability, ModelSpec

_log = config.get_logger(__name__)


# ── Default model plan ─────────────────────────────────────────────────────────
# extra_body carries the exact per-model tuning from the NIM API examples.
_DEFAULT_SPECS: dict[Capability, ModelSpec] = {
    Capability.EMBEDDING: ModelSpec(
        capability=Capability.EMBEDDING, provider="nvidia",
        model="nvidia/nemotron-3-embed-1b", api_key_env="NVIDIA_EMBED_API_KEY",
        kind="embedding"),

    Capability.REASONING_EXECUTIVE: ModelSpec(
        capability=Capability.REASONING_EXECUTIVE, provider="nvidia",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        api_key_env="NVIDIA_REASONING_EXEC_API_KEY",
        temperature=1.0, top_p=0.95, max_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 16384},
        fallback=Capability.REASONING_OPERATIONS),

    Capability.REASONING_OPERATIONS: ModelSpec(
        capability=Capability.REASONING_OPERATIONS, provider="nvidia",
        model="z-ai/glm-5.2", api_key_env="NVIDIA_REASONING_OPS_API_KEY",
        temperature=1.0, top_p=1.0, max_tokens=16384,
        extra_body={"seed": 42}),

    Capability.CODING: ModelSpec(
        capability=Capability.CODING, provider="nvidia",
        model="deepseek-ai/deepseek-v4-flash", api_key_env="NVIDIA_CODING_API_KEY",
        temperature=1.0, top_p=0.95, max_tokens=16384,
        extra_body={"chat_template_kwargs": {"thinking": True,
                                             "reasoning_effort": "high"}},
        fallback=Capability.REASONING_OPERATIONS),

    # Abstractions — declared so agents can request them; resolve() returns the
    # spec, but no key is set until a model is wired, so calls degrade cleanly.
    Capability.VISION: ModelSpec(
        capability=Capability.VISION, provider="nvidia",
        model="", api_key_env="NVIDIA_VISION_API_KEY"),
    Capability.OCR: ModelSpec(
        capability=Capability.OCR, provider="nvidia",
        model="", api_key_env="NVIDIA_OCR_API_KEY"),
    Capability.SAFETY: ModelSpec(
        capability=Capability.SAFETY, provider="nvidia",
        model="", api_key_env="NVIDIA_SAFETY_API_KEY",
        fallback=Capability.REASONING_OPERATIONS),
}


class CapabilityRegistry:
    """Resolves capabilities to model specs. Injectable and overridable."""

    def __init__(self, specs: Optional[dict[Capability, ModelSpec]] = None) -> None:
        self._specs: dict[Capability, ModelSpec] = dict(specs or _DEFAULT_SPECS)

    def resolve(self, capability: Capability | str) -> Optional[ModelSpec]:
        cap = Capability(capability) if isinstance(capability, str) else capability
        return self._specs.get(cap)

    def register(self, spec: ModelSpec) -> "CapabilityRegistry":
        """Override or add a capability mapping (e.g. wire a vision model)."""
        self._specs[spec.capability] = spec
        _log.info("Registered %s -> %s (%s)", spec.capability.value,
                  spec.model or "(unwired)", spec.provider)
        return self

    def configured_capabilities(self) -> dict[str, bool]:
        """Which capabilities have a key set — for status display."""
        return {c.value: bool(config.get_env(s.api_key_env) and s.model)
                for c, s in self._specs.items()}
