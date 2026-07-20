"""
AI-layer tests: registry resolution, capability→model mapping, the router's
fallback chain, provider client caching + retries + graceful failure, the
AI facade, and the capability services — all with mocked providers (no
network; the NVIDIA endpoint is never actually called).
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ai.providers.nvidia import NvidiaProvider
from ai.registry import CapabilityRegistry
from ai.router import AIRouter
from ai.types import (AIResponse, Capability, EmbeddingResponse, Message,
                      ModelSpec)


# ── Fake provider for router tests ────────────────────────────────────────────

class FakeProvider:
    name = "nvidia"

    def __init__(self, configured=True, ok=True, text="answer"):
        self._configured, self._ok, self._text = configured, ok, text
        self.calls: list[str] = []

    def is_configured(self, spec):
        return self._configured

    def chat(self, messages, spec):
        self.calls.append(spec.model)
        if self._ok:
            return AIResponse(text=self._text, capability=spec.capability.value,
                              ok=True, model_used=spec.model, provider=self.name)
        return AIResponse.failure(spec.capability.value, "boom")

    def embed(self, texts, spec, input_type="passage"):
        if self._ok:
            return EmbeddingResponse(vectors=[[0.1, 0.2]] * len(texts),
                                     model_used=spec.model, ok=True)
        return EmbeddingResponse(vectors=[], ok=False, error="boom")


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_maps_capabilities_to_models():
    reg = CapabilityRegistry()
    assert reg.resolve(Capability.EMBEDDING).model == "nvidia/nemotron-3-embed-1b"
    assert reg.resolve("reasoning.executive").model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert reg.resolve(Capability.REASONING_OPERATIONS).model == "z-ai/glm-5.2"
    assert reg.resolve(Capability.CODING).model == "deepseek-ai/deepseek-v4-flash"
    # executive declares an operations fallback
    assert reg.resolve(Capability.REASONING_EXECUTIVE).fallback == Capability.REASONING_OPERATIONS


def test_registry_override():
    reg = CapabilityRegistry()
    custom = ModelSpec(Capability.CODING, "nvidia", "some/other-model",
                       "NVIDIA_CODING_API_KEY")
    reg.register(custom)
    assert reg.resolve(Capability.CODING).model == "some/other-model"


def test_agents_never_hardcode_models():
    """The domain agents must not reference concrete model names."""
    import pathlib
    domain = pathlib.Path(__file__).resolve().parents[1] / "modules" / "agents" / "domain.py"
    text = domain.read_text()
    for forbidden in ("glm-5.2", "nemotron", "deepseek", "gpt-", "claude-"):
        assert forbidden not in text, f"agent hardcodes model '{forbidden}'"


# ── Router dispatch + fallback ────────────────────────────────────────────────

def test_router_happy_path():
    prov = FakeProvider(text="ops answer")
    router = AIRouter(providers={"nvidia": prov})
    resp = router.ask(Capability.REASONING_OPERATIONS, "inventory_review",
                      {"eoq": 420, "rop": 95})
    assert resp.ok and resp.text == "ops answer"
    assert resp.model_used == "z-ai/glm-5.2" and not resp.fell_back
    # the structured context was serialised into the user message
    assert prov.calls == ["z-ai/glm-5.2"]


def test_router_falls_back_on_provider_failure():
    prov = FakeProvider(ok=False)
    offline = MagicMock(return_value="offline text")
    router = AIRouter(providers={"nvidia": prov}, offline_handler=offline)
    # executive fails → operations fails → offline handler
    resp = router.ask(Capability.REASONING_EXECUTIVE, "synthesis", {"x": 1})
    assert resp.ok and resp.fell_back and resp.text == "offline text"
    assert prov.calls == ["nvidia/nemotron-3-ultra-550b-a55b", "z-ai/glm-5.2"]
    offline.assert_called_once()


def test_router_unconfigured_capability_uses_offline():
    prov = FakeProvider(configured=False)
    offline = MagicMock(return_value="fallback")
    router = AIRouter(providers={"nvidia": prov}, offline_handler=offline)
    resp = router.ask(Capability.REASONING_OPERATIONS, "t", {})
    assert resp.fell_back and resp.text == "fallback"
    assert prov.calls == []  # never called an unconfigured provider


def test_router_total_failure_returns_failure_response():
    prov = FakeProvider(configured=False)
    router = AIRouter(providers={"nvidia": prov}, offline_handler=lambda c, m: None)
    resp = router.ask(Capability.CODING, "sql", {})
    assert not resp.ok and "unavailable" in resp.error


def test_router_embed():
    router = AIRouter(providers={"nvidia": FakeProvider()})
    resp = router.embed(["a", "b"])
    assert resp.ok and len(resp.vectors) == 2


def test_memory_sink_records_calls():
    records = []
    router = AIRouter(providers={"nvidia": FakeProvider()})
    router.set_memory_sink(records.append)
    router.ask(Capability.REASONING_OPERATIONS, "review", {"a": 1})
    assert len(records) == 1 and records[0]["capability"] == "reasoning.operations"
    assert records[0]["ok"] is True


# ── NVIDIA provider (mocked SDK) ──────────────────────────────────────────────

def _spec():
    return ModelSpec(Capability.REASONING_OPERATIONS, "nvidia", "z-ai/glm-5.2",
                     "NVIDIA_REASONING_OPS_API_KEY", max_tokens=100)


def test_nvidia_client_cached_per_key(monkeypatch):
    monkeypatch.setattr(config, "get_env", lambda n: "key-123")
    prov = NvidiaProvider()
    made = []

    class _Client:
        def __init__(self, **kw):
            made.append(kw["api_key"])
    with patch("openai.OpenAI", _Client):
        prov._client("key-123")
        prov._client("key-123")   # cached
        prov._client("key-999")   # new key → new client
    assert made == ["key-123", "key-999"]


def test_nvidia_chat_parses_content_and_reasoning(monkeypatch):
    monkeypatch.setattr(config, "get_env", lambda n: "k")
    prov = NvidiaProvider()
    msg = MagicMock()
    msg.content = "the answer"
    msg.reasoning_content = "step by step"
    fake_resp = MagicMock(choices=[MagicMock(message=msg)])
    client = MagicMock()
    client.chat.completions.create.return_value = fake_resp
    prov._clients["k"] = client
    out = prov.chat([Message("user", "hi")], _spec())
    assert out.ok and out.text == "the answer" and out.reasoning == "step by step"
    assert out.model_used == "z-ai/glm-5.2"
    # extra_body params were passed through
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["max_tokens"] == 100 and kwargs["stream"] is False


def test_nvidia_chat_retries_then_fails(monkeypatch):
    monkeypatch.setattr(config, "get_env", lambda n: "k")
    monkeypatch.setattr("ai.providers.nvidia.BACKOFF_BASE_S", 0.0)  # no real sleep
    prov = NvidiaProvider()
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("503")
    prov._clients["k"] = client
    out = prov.chat([Message("user", "hi")], _spec())
    assert not out.ok and "failed after" in out.error
    assert client.chat.completions.create.call_count == 3  # MAX_RETRIES


def test_nvidia_missing_key_is_graceful(monkeypatch):
    monkeypatch.setattr(config, "get_env", lambda n: None)
    out = NvidiaProvider().chat([Message("user", "hi")], _spec())
    assert not out.ok and "not set" in out.error


# ── Capability services route correctly ───────────────────────────────────────

def test_reasoning_services_use_right_capability():
    from ai import reasoning
    prov = FakeProvider(text="ok")
    from ai.router import AI
    AI.configure(AIRouter(providers={"nvidia": prov}))
    try:
        reasoning.operations("inventory_review", {"eoq": 1})
        reasoning.executive("board_report", {"kpi": 2})
        assert prov.calls == ["z-ai/glm-5.2", "nvidia/nemotron-3-ultra-550b-a55b"]
    finally:
        AI._router = None  # reset the singleton for other tests


def test_agent_ai_narrative_via_router():
    """Agents get an AI narrative through the router when ai_enabled — and the
    router, not the agent, chose the model."""
    from modules.agents.domain import InventoryAgent, ExecutiveAgent
    from ai.router import AI
    import modules.decisions as decisions

    prov = FakeProvider(text="Inventory looks healthy; hold the line.")
    AI.configure(AIRouter(providers={"nvidia": prov}))
    try:
        profile = decisions.DemandProfile(100.0, 20.0, 7.0, 2.0, 36500.0, 700.0, 7)
        outputs = decisions.run_decision_engine(profile, service_level=0.95)
        shared = {"demand_profile": profile, "decision_outputs": outputs,
                  "sku_plan": None, "service_level": 0.95, "history_days": 300,
                  "avg_lead_time": 7.0}
        # disabled → no narrative, no model call
        r_off = InventoryAgent().execute(shared, {}, ai_enabled=False)
        assert r_off.ai_narrative is None and prov.calls == []
        # enabled → narrative via reasoning.operations
        r_on = InventoryAgent().execute(shared, {}, ai_enabled=True)
        assert r_on.ai_narrative == "Inventory looks healthy; hold the line."
        assert prov.calls == ["z-ai/glm-5.2"]
        # Executive routes to the executive model
        prov.calls.clear()
        ex = ExecutiveAgent().execute({"health": {"score": 80, "grade": "B"},
                                       "memory": {}}, {}, ai_enabled=True)
        assert prov.calls == ["nvidia/nemotron-3-ultra-550b-a55b"]
    finally:
        AI._router = None


def test_rag_uses_embedding_retriever_when_available():
    """When embeddings are configured, RAG retrieves semantically; otherwise
    it falls back to lexical — both return cited passages."""
    from modules import store, knowledge
    from ai.router import AI

    class EmbedProvider(FakeProvider):
        def is_configured(self, spec):
            return True
        def embed(self, texts, spec, input_type="passage"):
            # toy embedding: [len, count of 'lead'] so the query about lead
            # times scores the lead-time chunk highest
            return EmbeddingResponse(
                vectors=[[float(len(t)), float(t.lower().count("lead"))] for t in texts],
                model_used=spec.model, ok=True)

    store.add_document("policy.txt",
                       "Supplier lead times must not exceed 21 days.\n\n"
                       "Forklift operators need annual safety training.")
    # embeddings ON
    AI.configure(AIRouter(providers={"nvidia": EmbedProvider()}))
    try:
        hits = knowledge.retrieve("what is the supplier lead time policy?")
        assert hits and hits[0]["retriever"] == "embedding"
    finally:
        AI._router = None
    # embeddings OFF (no provider) → lexical
    AI.configure(AIRouter(providers={}, offline_handler=lambda c, m: None))
    try:
        hits = knowledge.retrieve("what is the supplier lead time policy?")
        assert hits and hits[0]["retriever"] == "tfidf"
    finally:
        AI._router = None


def test_status_reflects_configuration(monkeypatch):
    monkeypatch.setattr(config, "get_env",
                        lambda n: "k" if "OPS" in n or "EMBED" in n else None)
    reg = CapabilityRegistry()
    status = reg.configured_capabilities()
    assert status["reasoning.operations"] is True
    assert status["embedding"] is True
    assert status["vision"] is False  # no model wired
