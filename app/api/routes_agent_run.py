"""API route for the Agent Run ABI.

This is a minimal scaffold that introduces the stable request/response
contract while preserving existing endpoints.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter

from ..models.agent_run import AgentRunRequest, AgentRunResponse

router = APIRouter()


@router.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(req: AgentRunRequest) -> AgentRunResponse:
    """Execute a minimal agent run and return ABI-conformant output.

    Current behavior is intentionally minimal: it returns a deterministic
    placeholder assistant message, no actions, no memory writes, and a trace ID.
    """

    _ = req
    return AgentRunResponse(
        assistant_message="Agent runtime scaffold is active.",
        actions=[],
        memory_writes=[],
        trace_id=str(uuid4()),
    )
