"""
ai/cache.py
Response + embedding caching for the AI layer.

A bounded, TTL'd in-memory cache keyed by a stable hash of (capability,
messages, model params). Cutting a repeated identical call to a cache hit
is the cheapest latency and cost win available. The cache is process-local
by design (fast, no external store); embedding vectors additionally get a
persistent index in ai/rag.py.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import config
from ai.types import AIResponse, Message

_log = config.get_logger(__name__)

DEFAULT_MAX_ENTRIES = 512
DEFAULT_TTL_S = 3600.0


@dataclass
class _Entry:
    response: AIResponse
    expires_at: float


class ResponseCache:
    """Thread-unsafe LRU+TTL cache (Streamlit is single-threaded per session)."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES,
                 ttl_s: float = DEFAULT_TTL_S) -> None:
        self._store: "OrderedDict[str, _Entry]" = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_s
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(capability: str, messages: list[Message], spec_fingerprint: str) -> str:
        payload = json.dumps(
            {"cap": capability, "spec": spec_fingerprint,
             "msgs": [(m.role, m.content) for m in messages]},
            sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[AIResponse]:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.expires_at < time.monotonic():
            self._store.pop(key, None)
            self.misses += 1
            return None
        self._store.move_to_end(key)      # LRU touch
        self.hits += 1
        # Return a copy flagged as cached (never mutate the stored response)
        from dataclasses import replace
        return replace(entry.response, cached=True, latency_ms=0)

    def put(self, key: str, response: AIResponse) -> None:
        if not response.ok:
            return                        # never cache failures
        self._store[key] = _Entry(response, time.monotonic() + self._ttl)
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)   # evict LRU

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._store), "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total * 100, 1) if total else None}
