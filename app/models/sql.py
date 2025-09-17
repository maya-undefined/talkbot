from sqlalchemy import Column, String, Integer, ForeignKey, Text, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from ..store.db import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    tenant_id = Column(String, index=True)
    filename = Column(String)
    meta = Column(JSONB)

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(String, primary_key=True)
    tenant_id = Column(String, index=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"))
    text = Column(Text)
    meta = Column(JSONB)
    # Vector column stored in pgvector table; linkage by id in vectors table

class Vector(Base):
    __tablename__ = "vectors"
    id = Column(String, primary_key=True)           # chunk_id
    tenant_id = Column(String, index=True)
    dim = Column(Integer)
    # Stored in pgvector extension table; we’ll manage via raw SQL in repo layer