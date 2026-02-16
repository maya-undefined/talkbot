"""Tests for /ingest_pg route behavior."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_ingest_pg_retries_embedding_429_then_succeeds(monkeypatch) -> None:
    """Ingest should retry embedding generation on transient 429 errors."""

    state = {"calls": 0}

    async def no_sleep(_seconds: float) -> None:
        return None

    async def flaky_embed(_chunks):
        state["calls"] += 1
        if state["calls"] < 3:
            request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return [[0.1, 0.2, 0.3]]

    def fake_upsert_document(_tenant, _filename, _meta) -> str:
        return "doc-1"

    calls = {"stored": False}

    def fake_upsert_chunks(_tenant, _doc_id, _pairs, _vecs) -> None:
        calls["stored"] = True

    monkeypatch.setattr("app.api.routes_ingest_pg.embed_texts", flaky_embed)
    monkeypatch.setattr("app.api.routes_ingest_pg.PGV.upsert_document", fake_upsert_document)
    monkeypatch.setattr("app.api.routes_ingest_pg.PGV.upsert_chunks_embeddings", fake_upsert_chunks)
    monkeypatch.setattr("app.api.routes_ingest_pg.asyncio.sleep", no_sleep)

    files = {"file": ("sample.csv", b"a,b\n1,2", "text/csv")}
    data = {"tenant_id": "t1"}
    resp = client.post("/ingest_pg", files=files, data=data)

    assert resp.status_code == 200
    assert resp.json()["doc_id"] == "doc-1"
    assert state["calls"] == 3
    assert calls["stored"] is True


def test_ingest_pg_returns_503_when_embedding_429_persists(monkeypatch) -> None:
    """Persistent embedding 429 errors should return a retryable server response."""

    async def no_sleep(_seconds: float) -> None:
        return None

    async def always_429(_chunks):
        request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    def fake_upsert_document(_tenant, _filename, _meta) -> str:
        return "doc-1"

    monkeypatch.setattr("app.api.routes_ingest_pg.embed_texts", always_429)
    monkeypatch.setattr("app.api.routes_ingest_pg.PGV.upsert_document", fake_upsert_document)
    monkeypatch.setattr("app.api.routes_ingest_pg.asyncio.sleep", no_sleep)

    files = {"file": ("sample.csv", b"a,b\n1,2", "text/csv")}
    data = {"tenant_id": "t1"}
    resp = client.post("/ingest_pg", files=files, data=data)

    assert resp.status_code == 503
    assert "rate-limited" in resp.json()["detail"]


def test_ingest_pg_openapi_path_still_mounted() -> None:
    """Ensure /ingest_pg remains mounted."""

    paths = client.get("/openapi.json").json()["paths"]
    assert "/ingest_pg" in paths
