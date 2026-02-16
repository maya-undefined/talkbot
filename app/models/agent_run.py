"""Pydantic models for the Agent Run ABI endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """A chat-style message used as agent runtime input context."""

    role: Literal["user", "assistant", "system"]
    content: str


class AgentAction(BaseModel):
    """A tool/action invocation emitted by the runtime."""

    name: str = Field(..., description="Registered tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["planned", "succeeded", "failed"] = "planned"
    result: Optional[Dict[str, Any]] = None


class MemoryWrite(BaseModel):
    """A memory write operation produced during an agent run."""

    memory_type: Literal["episodic", "semantic", "procedural"]
    key: Optional[str] = None
    value: Dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    """Input contract for provider-agnostic agent runs."""

    session_id: str
    tenant_id: str
    persona_id: str
    messages: List[AgentMessage] = Field(default_factory=list)
    capabilities_manifest: Dict[str, Any] = Field(default_factory=dict)
    policy: Dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    """Output contract for provider-agnostic agent runs."""

    assistant_message: str
    actions: List[AgentAction] = Field(default_factory=list)
    memory_writes: List[MemoryWrite] = Field(default_factory=list)
    trace_id: str
