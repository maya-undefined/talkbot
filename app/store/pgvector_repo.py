from typing import List, Tuple, Dict, Any
import uuid
import numpy as np
from sqlalchemy import text
from .db import SessionLocal

from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB

# Schema we expect:
#   CREATE EXTENSION IF NOT EXISTS vector;
#   CREATE TABLE IF NOT EXISTS documents (...);
#   CREATE TABLE IF NOT EXISTS chunks (...);
#   CREATE TABLE IF NOT EXISTS vectors (
#       id TEXT PRIMARY KEY, tenant_id TEXT, embedding vector(3072)  -- adjust dim
#   );
#   CREATE INDEX IF NOT EXISTS vectors_tenant_idx ON vectors (tenant_id);
#   CREATE INDEX IF NOT EXISTS vectors_embed_idx ON vectors USING ivfflat (embedding vector_cosine_ops);

class PgVectorRepo:
    def __init__(self, dim: int = 3072):
        self.dim = dim

    def upsert_document(self, tenant_id: str, filename: str, meta: Dict[str, Any]) -> str:
        doc_id = str(uuid.uuid4())
        stmt = text(
            "INSERT INTO documents (id, tenant_id, filename, meta) "
            "VALUES (:id, :t, :f, :m)"
        ).bindparams(bindparam("m", type_=JSONB))

        with SessionLocal() as s:
            s.execute(stmt, {"id": doc_id, "t": tenant_id, "f": filename, "m": meta})
            s.commit()

        return doc_id

    def upsert_chunks_embeddings(self, tenant_id: str, doc_id: str, chunks: List[Tuple[str, Dict[str, Any]]], embeddings: List[np.ndarray]):
        # chunks: list of (text, meta)
        with SessionLocal() as s:
            for (text_chunk, meta), emb in zip(chunks, embeddings):
                cid = str(uuid.uuid4())
                stmt_chunks = text(
                    "INSERT INTO chunks (id, tenant_id, document_id, text, meta) "
                    "VALUES (:id, :t, :d, :tx, :m)"
                ).bindparams(bindparam("m", type_=JSONB))
                s.execute(
                    stmt_chunks,
                    {"id": cid, "t": tenant_id, "d": doc_id, "tx": text_chunk, "m": meta},
                )

                # store vector
                vec_list = ",".join(str(float(x)) for x in emb.tolist())
                s.execute(text(f"INSERT INTO vectors (id, tenant_id, dim, embedding) VALUES (:id,:t,:dim, '[{vec_list}]') ON CONFLICT (id) DO UPDATE SET embedding=EXCLUDED.embedding"),
                          {"id": cid, "t": tenant_id, "dim": len(emb)})
            s.commit()

    def search(self, tenant_id: str, query_vec: np.ndarray, top_k: int = 8) -> List[Dict[str, Any]]:
        q = ",".join(str(float(x)) for x in query_vec.tolist())
        sql = text(
            """
            SELECT c.id as chunk_id, c.document_id as doc_id, c.meta, c.text,
                   1 - (v.embedding <=> '[""" + q + """]') as score
            FROM vectors v
            JOIN chunks c ON c.id = v.id
            WHERE v.tenant_id = :t
            ORDER BY v.embedding <-> '[""" + q + """]'
            LIMIT :k
            """
        )
        with SessionLocal() as s:
            rows = s.execute(sql, {"t": tenant_id, "k": top_k}).mappings().all()
        out = []
        for r in rows:
            out.append({
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "score": float(r["score"]),
                "text": r["text"][:5000], # limits to 5000, probably should unlimit
                "meta": r["meta"],
            })
        return out

PGV = PgVectorRepo()
