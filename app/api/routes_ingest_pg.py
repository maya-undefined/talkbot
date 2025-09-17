from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uuid
from ..core.auth import require_tenant
from ..utils.parsing import extract_text
from ..utils.chunking import split_into_chunks
from ..services.embeddings_openai import embed_texts
from ..store.pgvector_repo import PGV

router = APIRouter()

class IngestResponse(BaseModel):
    doc_id: str
    chunks: int

@router.post("/ingest_pg", response_model=IngestResponse)
async def ingest_pg(file: UploadFile = File(...), tenant_id: str = Form("t1")):
    tenant = await require_tenant(tenant_id)
    raw = await file.read()
    text = extract_text(file.filename, raw)
    if not text.strip():
        raise HTTPException(400, detail="No text extracted from file.")

    doc_id = PGV.upsert_document(tenant, file.filename, {"title": file.filename})
    chunks = split_into_chunks(text, target_tokens=360, overlap=40)
    metas = [{"title": file.filename, "page": i+1} for i in range(len(chunks))]
    vecs = await embed_texts(chunks)
    PGV.upsert_chunks_embeddings(tenant, doc_id, list(zip(chunks, metas)), vecs)

    return IngestResponse(doc_id=doc_id, chunks=len(chunks))
