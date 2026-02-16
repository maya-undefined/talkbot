"""Schemas for the provider-agnostic Agent Run ABI endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """A chat message exchanged during an agent run."""

    role: Literal["user", "assistant", "system"]
    content: str


class AgentAction(BaseModel):
    """Represents a tool/action event emitted by the agent runtime."""

    name: str = Field(..., description="Tool/action name.")
    status: Literal["planned", "executed", "failed"] = Field(
        "planned", description="Lifecycle status for the action."
    )
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)


class AgentMemoryWrite(BaseModel):
    """Represents a memory write emitted by the agent runtime."""

    memory_type: Literal["episodic", "semantic", "procedural"]
    key: str
    value: Dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    """Request payload for the Agent Run ABI."""

    session_id: str
    tenant_id: str
    persona_id: str
    messages: List[AgentMessage]
    capabilities_manifest: Dict[str, Any] = Field(default_factory=dict)
    policy: Dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    """Response payload for the Agent Run ABI."""

    assistant_message: str
    actions: List[AgentAction] = Field(default_factory=list)
    memory_writes: List[AgentMemoryWrite] = Field(default_factory=list)
    trace_id: str
