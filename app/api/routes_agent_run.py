"""API route for the Agent Run ABI."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from typing import Awaitable, Callable, TypeVar

from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException

from ..core.auth import require_tenant
from ..core.personas import PERSONAS
from ..models.agent_run import AgentRunRequest, AgentRunResponse, Trace
from ..services.embeddings_openai import embed_texts
from ..services.llm import LLM_ENGINE
from ..services.llm_openai import answer_with_openai
from ..services.openai_client import OPENAI_API_KEY
from ..store.pgvector_repo import PGV

router = APIRouter()
_SESSION_LOCKS: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
T = TypeVar("T")
_MAX_429_RETRIES = 3
_BACKOFF_BASE_SECONDS = 0.2


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



def _fallback_assistant_message(req: AgentRunRequest, context: list[dict]) -> str:
    """Generate a local fallback response when remote LLM calls are unavailable."""

    persona = PERSONAS.get(req.persona_id)
    if not persona:
        raise HTTPException(404, detail="persona not found")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    return LLM_ENGINE.complete(persona["system"], messages, context, [])


async def _retrieve_context(req: AgentRunRequest, top_k: int = 8) -> list[dict]:
    """Retrieve minimal vector-search context for the request."""

    user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if not user_last:
        return []

    async def _embed_query() -> list:
        return await embed_texts([user_last])

    try:
        query_vector = (await _with_429_backoff(_embed_query))[0]

    except httpx.HTTPStatusError as exc:
        if _is_rate_limited(exc):
            return []
        raise

    return PGV.search(req.tenant_id, query_vector, top_k=top_k)


async def _run_llm(req: AgentRunRequest, context: list[dict]) -> str:
    """Execute the current LLM provider call for Agent Run output."""

    persona = PERSONAS.get(req.persona_id)
    if not persona:
        raise HTTPException(404, detail="persona not found")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    if not OPENAI_API_KEY:
        return _fallback_assistant_message(req, context)

    async def _chat_completion() -> str:
        return await answer_with_openai(persona["system"], messages, context, [])

    try:
        return await _with_429_backoff(_chat_completion)

    except httpx.HTTPStatusError as exc:
        if _is_rate_limited(exc):
            return _fallback_assistant_message(req, context)
        raise


@router.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(req: AgentRunRequest) -> AgentRunResponse:
    """Execute a minimal agent run using existing retrieval and LLM wiring."""

    await require_tenant(req.tenant_id)
    lock = _SESSION_LOCKS[req.session_id]
    async with lock:
        context = await _retrieve_context(req)
        assistant_message = await _run_llm(req, context)
        return AgentRunResponse(
            assistant_message=assistant_message,
            actions=[],
            memory_writes=[],
            trace=Trace(trace_id=str(uuid4())),
        )
