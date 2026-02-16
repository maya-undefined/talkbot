"""Route handlers for retrieval-augmented chat with pgvector context."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, Literal, TypeVar

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.auth import require_tenant
from ..core.personas import PERSONAS
from ..services.embeddings_openai import embed_texts
from ..services.llm import LLM_ENGINE
from ..services.llm_openai import answer_with_openai
from ..store.pgvector_repo import PGV

router = APIRouter()
T = TypeVar("T")
_MAX_429_RETRIES = 3
_BACKOFF_BASE_SECONDS = 0.2


class Message(BaseModel):
    """Chat message supplied in chat history."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Request body for pgvector-backed chat."""

    tenant_id: str = Field(...)
    persona_id: str = Field(...)
    messages: List[Message] = Field(...)
    top_k: int = 8


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


@router.post("/chat_pg")
async def chat_pg(req: ChatRequest):
    """Answer user chat using retrieval context and an LLM completion."""

    tenant = await require_tenant(req.tenant_id)

    persona = PERSONAS.get(req.persona_id)
    if not persona:
        raise HTTPException(404, detail="persona not found")

    user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")

    async def _embed_query() -> list:
        return await embed_texts([user_last])

    try:
        qv = (await _with_429_backoff(_embed_query))[0]
        ctx = PGV.search(tenant, qv, top_k=req.top_k)
    except httpx.HTTPStatusError as exc:
        if _is_rate_limited(exc):
            ctx = []
        else:
            raise

    oai_msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    async def _chat_completion() -> str:
        return await answer_with_openai(persona["system"], oai_msgs, ctx, [])

    try:
        content = await _with_429_backoff(_chat_completion)
    except httpx.HTTPStatusError as exc:
        if _is_rate_limited(exc):
            content = LLM_ENGINE.complete(persona["system"], oai_msgs, ctx, [])
        else:
            raise

    return {"answer": content, "citations": ctx}
