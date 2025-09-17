CREATE TABLE IF NOT EXISTS documents (
  id        TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  filename  TEXT,
  meta      JSONB
);

CREATE TABLE IF NOT EXISTS chunks (
  id          TEXT PRIMARY KEY,
  tenant_id   TEXT NOT NULL,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  text        TEXT,
  meta        JSONB
);

-- pick your embedding dim here and keep it consistent with your code:
CREATE TABLE IF NOT EXISTS vectors (
  id        TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  dim       INT NOT NULL,
  embedding vector(1536)  -- or vector(3072)
);
