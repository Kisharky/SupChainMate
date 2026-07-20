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
        self._observer: Optional[Callable[[str, AIResponse], None]] = None
        from ai.cache import ResponseCache
        self.cache = ResponseCache()

    def set_observer(self, observer: Callable[[str, AIResponse], None]) -> None:
        """Inject the observability sink (task, response) -> None."""
        self._observer = observer

    def _fingerprint(self, cap: Capability) -> str:
        spec = self.registry.resolve(cap)
        return f"{spec.model}:{spec.temperature}:{spec.top_p}" if spec else cap.value

    # ── Public facade ─────────────────────────────────────────────────────────
    def ask(self, capability: Capability | str, task: str, context: Any = None,
            system: Optional[str] = None, user: Optional[str] = None,
            use_cache: bool = True) -> AIResponse:
        """
        Fulfil a reasoning/coding request by capability.

        `task` names the operation (observability); `context` is the
        structured business-logic output the model reasons over; `system`
        and `user` override the default prompt construction. Identical
        requests are served from the response cache when `use_cache`.
        """
        messages = self._build_messages(task, context, system, user)
        cap = Capability(capability) if isinstance(capability, str) else capability
        ckey = self.cache.key(cap.value, messages, self._fingerprint(cap))
        if use_cache:
            hit = self.cache.get(ckey)
            if hit is not None:
                self._observe(task, hit)
                return hit
        resp = self._dispatch_chat(cap, messages, depth=0)
        if use_cache:
            self.cache.put(ckey, resp)
        self._observe(task, resp)
        return resp

    async def aask(self, capability: Capability | str, task: str, context: Any = None,
                   system: Optional[str] = None, user: Optional[str] = None,
                   use_cache: bool = True) -> AIResponse:
        """Async variant for non-Streamlit (service/API) deployment. Shares
        message building, caching, and observability with `ask`."""
        messages = self._build_messages(task, context, system, user)
        cap = Capability(capability) if isinstance(capability, str) else capability
        ckey = self.cache.key(cap.value, messages, self._fingerprint(cap))
        if use_cache:
            hit = self.cache.get(ckey)
            if hit is not None:
                self._observe(task, hit)
                return hit
        resp = await self._adispatch_chat(cap, messages, depth=0)
        if use_cache:
            self.cache.put(ckey, resp)
        self._observe(task, resp)
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

    async def _adispatch_chat(self, cap: Capability, messages: list[Message],
                              depth: int) -> AIResponse:
        spec = self.registry.resolve(cap)
        if spec is None:
            return self._offline_response(cap.value, messages, "no spec")
        provider = self.providers.get(spec.provider)
        if (provider is not None and spec.model and provider.is_configured(spec)
                and hasattr(provider, "achat")):
            resp = await provider.achat(messages, spec)
            if resp.ok:
                return resp
            _log.warning("Async capability %s failed: %s", cap.value, resp.error)
        if spec.fallback is not None and depth < MAX_FALLBACK_DEPTH:
            fb = await self._adispatch_chat(spec.fallback, messages, depth + 1)
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

    def _observe(self, task: str, resp: AIResponse) -> None:
        if self._observer is None:
            return
        try:
            self._observer(task, resp)
        except Exception as e:  # observability must never break a request
            _log.warning("observability sink failed: %s", e)


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

    async def aask(self, capability: Capability | str, task: str,
                   context: Any = None, **kw) -> AIResponse:
        return await self.router.aask(capability, task, context, **kw)

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
        from ai import observability
        router.set_observer(observability.record)
    except Exception:  # observability wiring must never block the platform
        pass
    return router


AI = _AIFacade()
