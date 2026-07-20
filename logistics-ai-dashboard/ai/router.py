"""
ai/router.py
The AI Router — the ONLY component that turns a capability into a concrete
model call. Business logic and agents call AI.ask(capability=...) and never
learn which model answered.

Responsibilities (single):
  1. Resolve capability → ModelSpec via the registry
  2. Dispatch to the right provider
  3. Walk the fallback chain (spec.fallback → … → offline) on failure
  4. Record every call to memory for audit/observability

Dependency injection: the router is constructed with a registry, a
provider map, and an optional offline handler, so tests inject fakes and
future providers slot in without touching agents.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import config
from ai.providers.base import Provider
from ai.providers.nvidia import NvidiaProvider
from ai.registry import CapabilityRegistry
from ai.types import (AIResponse, Capability, EmbeddingResponse, Message,
                      ModelSpec)

_log = config.get_logger(__name__)

# An offline handler answers when no provider is configured/reachable. It
# receives (capability, messages) and returns text or None.
OfflineHandler = Callable[[str, list[Message]], Optional[str]]
MAX_FALLBACK_DEPTH = 4


class AIRouter:
    def __init__(self, registry: Optional[CapabilityRegistry] = None,
                 providers: Optional[dict[str, Provider]] = None,
                 offline_handler: Optional[OfflineHandler] = None) -> None:
        self.registry = registry or CapabilityRegistry()
        # NB: `providers is None` — an explicit empty dict means "no providers"
        # (used by tests) and must NOT be replaced by the default.
        self.providers: dict[str, Provider] = (
            providers if providers is not None else {"nvidia": NvidiaProvider()})
        self._offline = offline_handler
        self._memory_sink: Optional[Callable[[dict], None]] = None

    def set_memory_sink(self, sink: Callable[[dict], None]) -> None:
        """Inject where call records go (defaults to none)."""
        self._memory_sink = sink

    # ── Public facade ─────────────────────────────────────────────────────────
    def ask(self, capability: Capability | str, task: str, context: Any = None,
            system: Optional[str] = None, user: Optional[str] = None) -> AIResponse:
        """
        Fulfil a reasoning/coding request by capability.

        `task` names the operation (for logging/memory); `context` is the
        structured business-logic output the model reasons over; `system`
        and `user` override the default prompt construction.
        """
        messages = self._build_messages(task, context, system, user)
        cap = Capability(capability) if isinstance(capability, str) else capability
        resp = self._dispatch_chat(cap, messages, depth=0)
        self._record(task, cap.value, resp)
        return resp

    def embed(self, texts: list[str], input_type: str = "passage") -> EmbeddingResponse:
        spec = self.registry.resolve(Capability.EMBEDDING)
        if spec is None:
            return EmbeddingResponse(vectors=[], ok=False, error="no embedding spec")
        provider = self.providers.get(spec.provider)
        if provider is None or not provider.is_configured(spec):
            return EmbeddingResponse(vectors=[], ok=False,
                                     error="embedding provider not configured")
        return provider.embed(texts, spec, input_type=input_type)

    def status(self) -> dict[str, bool]:
        return self.registry.configured_capabilities()

    # ── Internal dispatch with fallback chain ─────────────────────────────────
    def _dispatch_chat(self, cap: Capability, messages: list[Message],
                       depth: int) -> AIResponse:
        spec = self.registry.resolve(cap)
        if spec is None:
            return self._offline_response(cap.value, messages, "no spec")
        provider = self.providers.get(spec.provider)

        if provider is not None and spec.model and provider.is_configured(spec):
            resp = provider.chat(messages, spec)
            if resp.ok:
                return resp
            _log.warning("Capability %s failed via %s: %s",
                         cap.value, spec.provider, resp.error)

        # Fallback: declared capability chain, then offline
        if spec.fallback is not None and depth < MAX_FALLBACK_DEPTH:
            _log.info("Falling back %s -> %s", cap.value, spec.fallback.value)
            fb = self._dispatch_chat(spec.fallback, messages, depth + 1)
            fb.fell_back = True
            return fb
        return self._offline_response(cap.value, messages, "provider unavailable")

    def _offline_response(self, capability: str, messages: list[Message],
                          reason: str) -> AIResponse:
        if self._offline is not None:
            text = self._offline(capability, messages)
            if text:
                return AIResponse(text=text, capability=capability, ok=True,
                                  provider="offline", model_used="offline",
                                  fell_back=True)
        return AIResponse.failure(capability, f"AI unavailable ({reason})")

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _build_messages(task: str, context: Any, system: Optional[str],
                        user: Optional[str]) -> list[Message]:
        import json
        sys_txt = system or (
            "You are a supply chain reasoning assistant. You are given the "
            "structured output of deterministic engines. Never invent or "
            "recompute numbers — reason over the values provided and be concise.")
        if user is not None:
            usr_txt = user
        else:
            ctx = ""
            if context is not None:
                try:
                    ctx = json.dumps(context, default=str, indent=2)[:6000]
                except (TypeError, ValueError):
                    ctx = str(context)[:6000]
            usr_txt = f"Task: {task}\n\nStructured context:\n{ctx}"
        return [Message("system", sys_txt), Message("user", usr_txt)]

    def _record(self, task: str, capability: str, resp: AIResponse) -> None:
        if self._memory_sink is None:
            return
        try:
            self._memory_sink({
                "task": task, "capability": capability,
                "model": resp.model_used, "provider": resp.provider,
                "ok": resp.ok, "fell_back": resp.fell_back,
                "latency_ms": resp.latency_ms, "error": resp.error})
        except Exception as e:  # observability must never break a request
            _log.warning("memory sink failed: %s", e)


# ── Singleton facade: agents import `AI` and call AI.ask(...) ──────────────────
class _AIFacade:
    """Lazily-built process-wide router so agents never construct one."""

    def __init__(self) -> None:
        self._router: Optional[AIRouter] = None

    def configure(self, router: AIRouter) -> None:
        self._router = router

    @property
    def router(self) -> AIRouter:
        if self._router is None:
            self._router = build_default_router()
        return self._router

    def ask(self, capability: Capability | str, task: str, context: Any = None,
            **kw) -> AIResponse:
        return self.router.ask(capability, task, context, **kw)

    def embed(self, texts: list[str], input_type: str = "passage") -> EmbeddingResponse:
        return self.router.embed(texts, input_type=input_type)

    def status(self) -> dict[str, bool]:
        return self.router.status()


def build_default_router() -> AIRouter:
    """Wire the default registry + NVIDIA provider + Groq/offline fallback."""
    def _offline(capability: str, messages: list[Message]) -> Optional[str]:
        # Ultimate fallback: reuse the existing Groq path so the platform
        # keeps working with only a Groq key (or degrades to None).
        try:
            from modules import groq_ai
            if groq_ai.is_available():
                sys_m = next((m.content for m in messages if m.role == "system"), "")
                usr_m = next((m.content for m in messages if m.role == "user"), "")
                out = groq_ai._call(
                    messages=[{"role": "system", "content": sys_m},
                              {"role": "user", "content": usr_m}],
                    max_tokens=600, temperature=0.3)
                return out if out and not out.startswith("[") else None
        except Exception:
            return None
        return None

    router = AIRouter(offline_handler=_offline)
    try:
        from modules import store
        router.set_memory_sink(
            lambda rec: store.log_event(
                "ai_router", "ai_call",
                details=f"{rec['task']} · {rec['capability']} · "
                        f"{rec.get('model') or 'offline'} · "
                        f"{'ok' if rec['ok'] else 'FAIL'}"
                        f"{' · fallback' if rec['fell_back'] else ''}"))
    except Exception:
        pass
    return router


AI = _AIFacade()
