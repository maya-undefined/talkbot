-- Enable pgvector and create minimal tables (dev-friendly; replace with Alembic later)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  filename TEXT,
  meta JSONB
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  text TEXT,
  meta JSONB
);

-- Adjust dimension to match your embedding model (text-embedding-3-large is 3072)
CREATE TABLE IF NOT EXISTS vectors (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  dim INT NOT NULL,
  embedding vector(1536)
);

CREATE INDEX IF NOT EXISTS vectors_tenant_idx ON vectors (tenant_id);
CREATE INDEX IF NOT EXISTS vectors_embed_idx ON vectors USING ivfflat (embedding vector_cosine_ops);
