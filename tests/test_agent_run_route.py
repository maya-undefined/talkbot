"""Tests for the Agent Run ABI endpoint."""

from __future__ import annotations

import asyncio

import httpx
from fastapi.testclient import TestClient

from app.api.routes_agent_run import _SESSION_LOCKS, _retrieve_context, _run_llm, agent_run
from app.main import app
from app.models.agent_run import AgentRunRequest


client = TestClient(app)


def test_agent_run_returns_abi_shape(monkeypatch) -> None:
    """The endpoint should accept the contract and return required fields."""

    async def fake_context(_req, top_k: int = 8) -> list[dict]:
        _ = top_k
        return [{"doc_id": "d1", "meta": {"title": "Doc", "page": 1}, "text": "stub"}]

    async def fake_run_llm(_req, context: list[dict]) -> str:
        assert context
        return "stubbed assistant message"

    monkeypatch.setattr("app.api.routes_agent_run._retrieve_context", fake_context)
    monkeypatch.setattr("app.api.routes_agent_run._run_llm", fake_run_llm)

    payload = {
        "session_id": "s1",
        "tenant_id": "t1",
        "persona_id": "budget_coach",
        "messages": [{"role": "user", "content": "hello"}],
        "capabilities_manifest": {"tools": ["doc_search", "calc"]},
        "policy": {"max_tool_calls": 2},
    }

    resp = client.post("/agent/run", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert body["assistant_message"] == "stubbed assistant message"
    assert body["actions"] == []
    assert body["memory_writes"] == []
    assert isinstance(body["trace"]["trace_id"], str)


def test_agent_run_validates_request_shape() -> None:
    """The endpoint should reject invalid payload values."""

    payload = {
        "session_id": "s1",
        "tenant_id": "t1",
        "persona_id": "budget_coach",
        "messages": [{"role": "bad-role", "content": "hello"}],
    }

    resp = client.post("/agent/run", json=payload)
    assert resp.status_code == 422


def test_agent_run_session_lock_serializes_runs(monkeypatch) -> None:
    """Runs for one session ID should be serialized in-process."""

    state = {"active": 0, "max_active": 0}

    async def fake_context(_req, top_k: int = 8) -> list[dict]:
        _ = top_k
        return []

    async def fake_run_llm(_req, context: list[dict]) -> str:
        _ = context
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return "ok"

    monkeypatch.setattr("app.api.routes_agent_run._retrieve_context", fake_context)
    monkeypatch.setattr("app.api.routes_agent_run._run_llm", fake_run_llm)

    req = AgentRunRequest(
        session_id="shared",
        tenant_id="t1",
        persona_id="budget_coach",
        messages=[{"role": "user", "content": "hello"}],
    )

    async def run_parallel_calls() -> None:
        _SESSION_LOCKS.clear()
        await asyncio.gather(agent_run(req), agent_run(req))

    asyncio.run(run_parallel_calls())
    assert state["max_active"] == 1


def test_agent_run_falls_back_on_openai_429(monkeypatch) -> None:
    """The run path should return local fallback text when OpenAI is rate-limited."""

    req = AgentRunRequest(
        session_id="s1",
        tenant_id="t1",
        persona_id="budget_coach",
        messages=[{"role": "user", "content": "hello"}],
    )

    async def rate_limited(*_args, **_kwargs) -> str:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr("app.api.routes_agent_run.OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.api.routes_agent_run.answer_with_openai", rate_limited)

    result = asyncio.run(_run_llm(req, []))
    assert "Answer (demo):" in result


def test_agent_run_retrieve_context_handles_embedding_429(monkeypatch) -> None:
    """Embedding rate limits should degrade retrieval to empty context."""

    req = AgentRunRequest(
        session_id="s1",
        tenant_id="t1",
        persona_id="budget_coach",
        messages=[{"role": "user", "content": "hello"}],
    )

    async def rate_limited_embed(_texts):
        request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr("app.api.routes_agent_run.embed_texts", rate_limited_embed)

    result = asyncio.run(_retrieve_context(req))
    assert result == []


def test_agent_run_retries_openai_429_then_succeeds(monkeypatch) -> None:
    """LLM call should retry on transient 429 and return provider output if retry succeeds."""

    req = AgentRunRequest(
        session_id="s1",
        tenant_id="t1",
        persona_id="budget_coach",
        messages=[{"role": "user", "content": "hello"}],
    )
    state = {"calls": 0}

    async def no_sleep(_seconds: float) -> None:
        return None

    async def flaky_answer(*_args, **_kwargs) -> str:
        state["calls"] += 1
        if state["calls"] < 3:
            request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return "recovered"

    monkeypatch.setattr("app.api.routes_agent_run.OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.api.routes_agent_run.answer_with_openai", flaky_answer)
    monkeypatch.setattr("app.api.routes_agent_run.asyncio.sleep", no_sleep)

    result = asyncio.run(_run_llm(req, []))
    assert result == "recovered"
    assert state["calls"] == 3


def test_agent_run_retries_embedding_429_then_succeeds(monkeypatch) -> None:
    """Retrieval should retry embedding calls on transient 429 responses."""

    req = AgentRunRequest(
        session_id="s1",
        tenant_id="t1",
        persona_id="budget_coach",
        messages=[{"role": "user", "content": "hello"}],
    )
    state = {"calls": 0}

    async def no_sleep(_seconds: float) -> None:
        return None

    async def flaky_embed(_texts):
        state["calls"] += 1
        if state["calls"] < 3:
            request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return [[0.1, 0.2, 0.3]]

    def fake_search(_tenant_id, _query_vec, top_k: int = 8) -> list[dict]:
        _ = top_k
        return [{"doc_id": "d1", "meta": {}, "text": "ok"}]

    monkeypatch.setattr("app.api.routes_agent_run.embed_texts", flaky_embed)
    monkeypatch.setattr("app.api.routes_agent_run.PGV.search", fake_search)
    monkeypatch.setattr("app.api.routes_agent_run.asyncio.sleep", no_sleep)

    result = asyncio.run(_retrieve_context(req))
    assert result == [{"doc_id": "d1", "meta": {}, "text": "ok"}]
    assert state["calls"] == 3


def test_existing_endpoints_still_present_in_openapi() -> None:
    """Ensure legacy endpoints remain mounted during rollout."""

    paths = client.get("/openapi.json").json()["paths"]
    assert "/ingest_pg" in paths
    assert "/chat_pg" in paths
    assert "/agent/run" in paths
