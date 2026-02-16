"""Tenant metadata and memory subsystem manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select

from ..models.sql import MemoryItem, MemoryType, SessionRecord
from .db import SessionLocal

TENANTS = {
    "t1": {"name": "Demo Tenant", "policy_pack_id": "default"},
    "t2": {"name": "Demo Tenant 2", "policy_pack_id": "default"},
}


@dataclass
class MemoryWriteRequest:
    """Input payload describing one memory write candidate."""

    tenant_id: str
    content: str
    memory_type: MemoryType
    session_id: str | None = None
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    source: str | None = None
    embedding: list[float] | None = None


class MemoryManager:
    """Persistence-backed manager for retrieving and writing memory items."""

    def __init__(self) -> None:
        """Initialize fallback in-memory stores for non-DB test/dev usage."""

        self._fallback_items: list[MemoryItem] = []
        self._fallback_sessions: dict[tuple[str, str], SessionRecord] = {}

    def retrieve(
        self,
        session_id: str,
        tenant_id: str,
        query_text: str,
        top_k: int,
    ) -> list[MemoryItem]:
        """Retrieve top memory items for a query and session context."""

        query_tokens = {token.strip(".,!?\"").lower() for token in query_text.split() if token.strip()}
        query_tokens.discard("")

        if SessionLocal is None:
            candidates = [
                item
                for item in self._fallback_items
                if item.tenant_id == tenant_id
                and (item.session_id == session_id or item.session_id is None)
            ]
            return _rank_memories(candidates, query_tokens, top_k)

        with SessionLocal() as db:
            stmt = (
                select(MemoryItem)
                .where(MemoryItem.tenant_id == tenant_id)
                .where(or_(MemoryItem.session_id == session_id, MemoryItem.session_id.is_(None)))
            )
            candidates = list(db.execute(stmt).scalars().all())

        return _rank_memories(candidates, query_tokens, top_k)

    def write(self, memory_writes: list[MemoryWriteRequest]) -> list[MemoryItem]:
        """Write memory items with basic dedupe by exact content and tag overlap."""

        if not memory_writes:
            return []

        if SessionLocal is None:
            return self._write_fallback(memory_writes)

        upserted: list[MemoryItem] = []
        with SessionLocal() as db:
            for write in memory_writes:
                existing = db.execute(
                    select(MemoryItem)
                    .where(MemoryItem.tenant_id == write.tenant_id)
                    .where(MemoryItem.type == write.memory_type)
                    .where(MemoryItem.content == write.content.strip())
                    .where(MemoryItem.session_id == write.session_id)
                ).scalar_one_or_none()
                if existing and _has_tag_overlap(existing.tags or [], write.tags):
                    _merge_memory(existing, write)
                    db.add(existing)
                    upserted.append(existing)
                    continue

                item = _build_memory_item(write)
                db.add(item)
                upserted.append(item)

            db.commit()
            for item in upserted:
                db.refresh(item)

        return upserted

    def summarize_session(self, session_id: str, tenant_id: str) -> str:
        """Generate a concise deterministic summary from recent session memory."""

        if SessionLocal is None:
            memories = [
                item
                for item in self._fallback_items
                if item.tenant_id == tenant_id and item.session_id == session_id
            ]
            memories.sort(key=lambda item: item.updated_at or datetime.min, reverse=True)
            summary = " | ".join(memory.content for memory in memories[:5])
            key = (tenant_id, session_id)
            session = self._fallback_sessions.get(key) or SessionRecord(
                tenant_id=tenant_id,
                session_id=session_id,
            )
            session.summary = summary
            session.updated_at = datetime.now(timezone.utc)
            self._fallback_sessions[key] = session
            return summary

        with SessionLocal() as db:
            memories = (
                db.execute(
                    select(MemoryItem)
                    .where(MemoryItem.tenant_id == tenant_id)
                    .where(MemoryItem.session_id == session_id)
                    .order_by(MemoryItem.updated_at.desc())
                    .limit(5)
                )
                .scalars()
                .all()
            )
            summary = " | ".join(memory.content for memory in memories)

            session = db.execute(
                select(SessionRecord)
                .where(SessionRecord.tenant_id == tenant_id)
                .where(SessionRecord.session_id == session_id)
            ).scalar_one_or_none()

            if not session:
                session = SessionRecord(tenant_id=tenant_id, session_id=session_id)

            session.summary = summary
            session.updated_at = datetime.now(timezone.utc)
            db.add(session)
            db.commit()

        return summary

    def reset_fallback(self) -> None:
        """Clear fallback state used in unit tests."""

        self._fallback_items.clear()
        self._fallback_sessions.clear()

    def _write_fallback(self, memory_writes: list[MemoryWriteRequest]) -> list[MemoryItem]:
        """Persist writes in in-memory structures when database is unavailable."""

        upserted: list[MemoryItem] = []
        for write in memory_writes:
            existing = next(
                (
                    item
                    for item in self._fallback_items
                    if item.tenant_id == write.tenant_id
                    and item.session_id == write.session_id
                    and item.type == write.memory_type
                    and item.content == write.content.strip()
                    and _has_tag_overlap(item.tags or [], write.tags)
                ),
                None,
            )
            if existing:
                _merge_memory(existing, write)
                upserted.append(existing)
                continue

            item = _build_memory_item(write)
            self._fallback_items.append(item)
            upserted.append(item)

        return upserted


def _build_memory_item(write: MemoryWriteRequest) -> MemoryItem:
    """Build a new memory ORM entity from write input."""

    now = datetime.now(timezone.utc)
    return MemoryItem(
        id=str(uuid4()),
        tenant_id=write.tenant_id,
        session_id=write.session_id,
        type=write.memory_type,
        content=write.content.strip(),
        tags=sorted(set(write.tags)),
        importance=write.importance,
        source=write.source,
        embedding=write.embedding,
        created_at=now,
        updated_at=now,
    )


def _merge_memory(existing: MemoryItem, write: MemoryWriteRequest) -> None:
    """Merge a duplicate write into an existing memory item."""

    existing.tags = sorted(set((existing.tags or []) + write.tags))
    existing.importance = max(existing.importance or 0.0, write.importance)
    existing.source = write.source or existing.source
    existing.embedding = write.embedding or existing.embedding
    existing.updated_at = datetime.now(timezone.utc)


def _has_tag_overlap(existing_tags: list[Any], new_tags: list[str]) -> bool:
    """Return true when write tags are absent or overlap with existing tags."""

    if not new_tags:
        return True
    return bool(set(str(tag) for tag in existing_tags) & set(new_tags))


def _rank_memories(candidates: list[MemoryItem], query_tokens: set[str], top_k: int) -> list[MemoryItem]:
    """Rank memory candidates with a simple overlap and importance score."""

    def _score(item: MemoryItem) -> float:
        content_tokens = {
            token.strip(".,!?\"").lower() for token in item.content.split() if token.strip()
        }
        token_overlap = len(query_tokens & content_tokens)
        return (item.importance or 0.0) * 2 + token_overlap

    ranked = sorted(candidates, key=_score, reverse=True)
    return ranked[:top_k]


MEMORY_MANAGER = MemoryManager()
