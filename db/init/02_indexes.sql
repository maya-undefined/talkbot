CREATE INDEX IF NOT EXISTS vectors_tenant_idx ON vectors (tenant_id);

-- If vector(1536)  -> IVFFLAT is fine:
CREATE INDEX IF NOT EXISTS vectors_embed_idx
ON vectors USING ivfflat (embedding vector_cosine_ops);

-- If vector(3072)  -> comment the IVFFLAT line above and use HNSW:
-- CREATE INDEX IF NOT EXISTS vectors_embed_idx
-- ON vectors USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);
