"""SQLAlchemy ORM models for persisted application data."""

from __future__ import annotations

from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB

from ..store.db import Base


class Document(Base):
    """A source document uploaded for retrieval."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    tenant_id = Column(String, index=True)
    filename = Column(String)
    meta = Column(JSONB)


class Chunk(Base):
    """A chunked span of document text."""

    __tablename__ = "chunks"

    id = Column(String, primary_key=True)
    tenant_id = Column(String, index=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"))
    text = Column(Text)
    meta = Column(JSONB)


class Vector(Base):
    """Vector index metadata keyed by chunk id."""

    __tablename__ = "vectors"

    id = Column(String, primary_key=True)
    tenant_id = Column(String, index=True)
    dim = Column(Integer)


class MemoryType(str, PyEnum):
    """Supported memory item categories."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class SessionRecord(Base):
    """Conversation session metadata and running summary."""

    __tablename__ = "sessions"

    tenant_id = Column(String, primary_key=True)
    session_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    summary = Column(Text)


class MemoryItem(Base):
    """Persisted memory item for one tenant/session."""

    __tablename__ = "memory_items"

    id = Column(String, primary_key=True)
    tenant_id = Column(String, index=True, nullable=False)
    session_id = Column(String, nullable=True)
    type = Column(Enum(MemoryType, name="memory_type"), nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(JSONB, nullable=False, default=list)
    importance = Column(Float, nullable=False, default=0.5)
    source = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    embedding = Column(JSONB, nullable=True)
