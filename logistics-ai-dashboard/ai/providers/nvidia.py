"""
ai/providers/nvidia.py
NVIDIA NIM provider — one centralized, cached OpenAI-compatible client per
API key, with retries, logging, and non-raising error handling.

Each capability's model can carry its own key (the four NIM keys map to
four capabilities), so clients are cached by key. All per-model tuning
(thinking budget, reasoning effort, embedding input_type) rides on the
ModelSpec.extra_body — the provider stays generic.
"""

from __future__ import annotations

import time
from typing import Optional

import config
from ai.types import (AIResponse, EmbeddingResponse, Message, ModelSpec,
                      ProviderError, TokenUsage)

_log = config.get_logger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5
REQUEST_TIMEOUT_S = 30.0   # bound each call; our wrapper owns retries


class NvidiaProvider:
    """OpenAI-compatible NIM backend."""

    name = "nvidia"

    def __init__(self, base_url: str = NVIDIA_BASE_URL) -> None:
        self._base_url = base_url
        # Connection pooling / provider reuse: one client per key, reused for
        # the process lifetime (the OpenAI SDK holds a pooled httpx transport).
        self._clients: dict[str, object] = {}
        self._aclients: dict[str, object] = {}

    # ── Client management (centralized creation, pooled) ──────────────────────
    def _client(self, api_key: str):
        if api_key not in self._clients:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ProviderError("openai package not installed") from e
            # max_retries=0: our _with_retries owns retry policy (no nested storms)
            self._clients[api_key] = OpenAI(
                base_url=self._base_url, api_key=api_key,
                timeout=REQUEST_TIMEOUT_S, max_retries=0)
        return self._clients[api_key]

    def _aclient(self, api_key: str):
        if api_key not in self._aclients:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ProviderError("openai package not installed") from e
            self._aclients[api_key] = AsyncOpenAI(
                base_url=self._base_url, api_key=api_key,
                timeout=REQUEST_TIMEOUT_S, max_retries=0)
        return self._aclients[api_key]

    def is_configured(self, spec: ModelSpec) -> bool:
        return bool(config.get_env(spec.api_key_env))

    # ── Shared request/response shaping (no duplication across sync/async) ────
    @staticmethod
    def _chat_payload(messages: list[Message], spec: ModelSpec) -> dict:
        payload = dict(
            model=spec.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=spec.temperature, top_p=spec.top_p,
            max_tokens=spec.max_tokens, stream=False)
        if spec.extra_body:
            payload["extra_body"] = spec.extra_body
        return payload

    @staticmethod
    def _usage(resp) -> TokenUsage:
        u = getattr(resp, "usage", None)
        if u is None:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(u, "total_tokens", 0) or 0))

    def _shape_chat(self, resp, spec: ModelSpec, t0: float) -> AIResponse:
        choice = resp.choices[0].message
        text = (getattr(choice, "content", None) or "").strip()
        reasoning = (getattr(choice, "reasoning_content", None)
                     or getattr(choice, "reasoning", None))
        return AIResponse(
            text=text, capability=spec.capability.value, ok=bool(text),
            model_used=spec.model, provider=self.name,
            reasoning=reasoning.strip() if isinstance(reasoning, str) else None,
            latency_ms=round((time.perf_counter() - t0) * 1000),
            usage=self._usage(resp),
            error=None if text else "empty completion")

    # ── Retry wrapper ─────────────────────────────────────────────────────────
    def _with_retries(self, fn, what: str):
        last: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn()
            except Exception as e:  # provider/network errors → retry then give up
                last = e
                if attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE_S * (2 ** (attempt - 1))
                    _log.warning("NVIDIA %s attempt %d/%d failed (%s) — retry in %.1fs",
                                 what, attempt, MAX_RETRIES, e, wait)
                    time.sleep(wait)
        raise ProviderError(f"{what} failed after {MAX_RETRIES} attempts: {last}")

    # ── Chat / reasoning (sync) ───────────────────────────────────────────────
    def chat(self, messages: list[Message], spec: ModelSpec) -> AIResponse:
        key = config.get_env(spec.api_key_env)
        if not key:
            return AIResponse.failure(spec.capability.value,
                                      f"{spec.api_key_env} not set")
        t0 = time.perf_counter()
        payload = self._chat_payload(messages, spec)
        try:
            client = self._client(key)
            resp = self._with_retries(
                lambda: client.chat.completions.create(**payload), "chat")
        except ProviderError as e:
            return AIResponse.failure(spec.capability.value, str(e))
        return self._shape_chat(resp, spec, t0)

    # ── Chat / reasoning (async) — same shaping, non-blocking transport ───────
    async def achat(self, messages: list[Message], spec: ModelSpec) -> AIResponse:
        import asyncio
        key = config.get_env(spec.api_key_env)
        if not key:
            return AIResponse.failure(spec.capability.value,
                                      f"{spec.api_key_env} not set")
        t0 = time.perf_counter()
        payload = self._chat_payload(messages, spec)
        last: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = self._aclient(key)
                resp = await client.chat.completions.create(**payload)
                return self._shape_chat(resp, spec, t0)
            except Exception as e:  # noqa: BLE001 — non-raising contract
                last = e
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))
        return AIResponse.failure(spec.capability.value,
                                  f"achat failed after {MAX_RETRIES} attempts: {last}")

    # ── Embeddings ────────────────────────────────────────────────────────────
    def embed(self, texts: list[str], spec: ModelSpec,
              input_type: str = "passage") -> EmbeddingResponse:
        key = config.get_env(spec.api_key_env)
        if not key:
            return EmbeddingResponse(vectors=[], ok=False,
                                     error=f"{spec.api_key_env} not set")
        extra = dict(spec.extra_body)
        extra.setdefault("input_type", input_type)
        extra.setdefault("truncate", "NONE")
        try:
            client = self._client(key)
            resp = self._with_retries(
                lambda: client.embeddings.create(
                    model=spec.model, input=texts, extra_body=extra), "embed")
        except ProviderError as e:
            return EmbeddingResponse(vectors=[], ok=False, error=str(e))
        return EmbeddingResponse(
            vectors=[d.embedding for d in resp.data],
            model_used=spec.model, ok=True)
