"""Route handlers for ingesting documents into pgvector storage."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..core.auth import require_tenant
from ..services.embeddings_openai import embed_texts
from ..store.pgvector_repo import PGV
from ..utils.chunking import split_into_chunks
from ..utils.parsing import extract_text

router = APIRouter()
T = TypeVar("T")
_MAX_429_RETRIES = 3
_BACKOFF_BASE_SECONDS = 0.2


class IngestResponse(BaseModel):
    """Response model for successful ingest requests."""

    doc_id: str
    chunks: int


def _is_rate_limited(exc: Exception) -> bool:
    """Return whether an exception corresponds to provider rate limiting."""

    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


async def _with_429_backoff(operation: Callable[[], Awaitable[T]]) -> T:
    """Run an async operation with bounded exponential backoff for HTTP 429 errors."""

    attempt = 0
    while True:
        try:
            return await operation()
        except httpx.HTTPStatusError as exc:
            if not _is_rate_limited(exc) or attempt >= _MAX_429_RETRIES:
                raise
            await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
            attempt += 1


@router.post("/ingest_pg", response_model=IngestResponse)
async def ingest_pg(file: UploadFile = File(...), tenant_id: str = Form("t1")) -> IngestResponse:
    """Ingest an uploaded file by extracting text, embedding chunks, and persisting them."""

    tenant = await require_tenant(tenant_id)
    raw = await file.read()
    text = extract_text(file.filename, raw)
    if not text.strip():
        raise HTTPException(400, detail="No text extracted from file.")

    doc_id = PGV.upsert_document(tenant, file.filename, {"title": file.filename})
    chunks = split_into_chunks(text, target_tokens=360, overlap=40)
    metas = [{"title": file.filename, "page": i + 1} for i in range(len(chunks))]

    async def _embed_chunks() -> list:
        return await embed_texts(chunks)

    try:
        vecs = await _with_429_backoff(_embed_chunks)
    except httpx.HTTPStatusError as exc:
        if _is_rate_limited(exc):
            raise HTTPException(
                status_code=503,
                detail="Embedding provider is rate-limited. Please retry shortly.",
            ) from exc
        raise

    PGV.upsert_chunks_embeddings(tenant, doc_id, list(zip(chunks, metas)), vecs)
    return IngestResponse(doc_id=doc_id, chunks=len(chunks))
