"""Tests for /chat_pg route behavior."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_chat_pg_retries_embedding_429_then_succeeds(monkeypatch) -> None:
    """Embedding requests should retry on transient 429 errors."""

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

    async def fake_answer(_system, _msgs, _ctx, _tools) -> str:
        return "ok"

    def fake_search(_tenant, _qv, top_k: int = 8) -> list[dict]:
        _ = top_k
        return [{"doc_id": "d1", "meta": {}, "text": "ctx"}]

    monkeypatch.setattr("app.api.routes_chat_pg.embed_texts", flaky_embed)
    monkeypatch.setattr("app.api.routes_chat_pg.PGV.search", fake_search)
    monkeypatch.setattr("app.api.routes_chat_pg.answer_with_openai", fake_answer)
    monkeypatch.setattr("app.api.routes_chat_pg.asyncio.sleep", no_sleep)

    payload = {
        "tenant_id": "t1",
        "persona_id": "budget_coach",
        "messages": [{"role": "user", "content": "hello"}],
    }
    resp = client.post("/chat_pg", json=payload)

    assert resp.status_code == 200
    assert resp.json()["answer"] == "ok"
    assert state["calls"] == 3


def test_chat_pg_falls_back_when_chat_429_persists(monkeypatch) -> None:
    """Persistent chat 429 responses should return local fallback output."""

    async def no_sleep(_seconds: float) -> None:
        return None

    async def fake_embed(_texts):
        return [[0.1, 0.2, 0.3]]

    def fake_search(_tenant, _qv, top_k: int = 8) -> list[dict]:
        _ = top_k
        return []

    async def always_429(_system, _msgs, _ctx, _tools) -> str:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr("app.api.routes_chat_pg.embed_texts", fake_embed)
    monkeypatch.setattr("app.api.routes_chat_pg.PGV.search", fake_search)
    monkeypatch.setattr("app.api.routes_chat_pg.answer_with_openai", always_429)
    monkeypatch.setattr("app.api.routes_chat_pg.asyncio.sleep", no_sleep)

    payload = {
        "tenant_id": "t1",
        "persona_id": "budget_coach",
        "messages": [{"role": "user", "content": "hello"}],
    }
    resp = client.post("/chat_pg", json=payload)

    assert resp.status_code == 200
    answer = resp.json()["answer"]
    assert "Disclosure:" in answer
    assert "SYSTEM:" not in answer
    assert "Q:" not in answer


def test_chat_pg_openapi_path_still_mounted() -> None:
    """Ensure /chat_pg remains mounted."""

    paths = client.get("/openapi.json").json()["paths"]
    assert "/chat_pg" in paths
