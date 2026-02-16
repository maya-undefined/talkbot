"""Routes for the provider-agnostic Agent Run ABI."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from ..core.auth import require_tenant
from ..core.personas import PERSONAS
from ..models.agent import AgentRunRequest, AgentRunResponse

router = APIRouter()


@router.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(req: AgentRunRequest) -> AgentRunResponse:
    """Execute a minimal agent run and return ABI-compatible output.

    This first iteration validates tenant/persona and returns a stub runtime result
    while preserving existing `/ingest_pg` and `/chat_pg` behavior.
    """

    await require_tenant(req.tenant_id)

    persona = PERSONAS.get(req.persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="persona not found")

    last_user_message = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        "",
    )

    assistant_message = (
        f"Agent runtime scaffold active for persona '{req.persona_id}'. "
        f"Received: {last_user_message}".strip()
    )

    return AgentRunResponse(
        assistant_message=assistant_message,
        actions=[],
        memory_writes=[],
        trace_id=str(uuid.uuid4()),
    )
