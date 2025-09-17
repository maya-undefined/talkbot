from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
import numpy as np

from ..core.auth import require_tenant
from ..core.personas import PERSONAS
from ..services.embeddings_openai import embed_texts
from ..store.pgvector_repo import PGV
from ..services.llm_openai import answer_with_openai

router = APIRouter()

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    tenant_id: str = Field(...)
    persona_id: str = Field(...)
    messages: List[Message] = Field(...)
    top_k: int = 8

@router.post("/chat_pg")
async def chat_pg(req: ChatRequest):

    tenant = await require_tenant(req.tenant_id)

    persona = PERSONAS.get(req.persona_id)
    # print(f'[DEBUG] {req} -- {persona}')

    if not persona:
        raise HTTPException(404, detail="persona not found")


    user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    # print(f'[DEBUG] {user_last}')
    qv = (await embed_texts([user_last]))[0]
    ctx = PGV.search(tenant, qv, top_k=req.top_k)
    # print(f"[DEBUG] ctx={ctx}")

    # Convert messages to OpenAI format
    oai_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    content = await answer_with_openai(persona["system"], oai_msgs, ctx, [])
    return {"answer": content, "citations": ctx}