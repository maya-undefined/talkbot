-- Memory subsystem tables for session summaries and reusable memories.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'memory_type') THEN
        CREATE TYPE memory_type AS ENUM ('episodic', 'semantic', 'procedural');
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS sessions (
  tenant_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  summary TEXT,
  PRIMARY KEY (tenant_id, session_id)
);

CREATE TABLE IF NOT EXISTS memory_items (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  session_id TEXT,
  type memory_type NOT NULL,
  content TEXT NOT NULL,
  tags JSONB NOT NULL DEFAULT '[]'::jsonb,
  importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
  source TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  embedding JSONB
);

CREATE INDEX IF NOT EXISTS sessions_tenant_session_idx ON sessions (tenant_id, session_id);
CREATE INDEX IF NOT EXISTS memory_items_tenant_session_idx ON memory_items (tenant_id, session_id);
CREATE INDEX IF NOT EXISTS memory_items_type_idx ON memory_items (type);
