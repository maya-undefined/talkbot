from fastapi import FastAPI
from .api.routes_ingest_pg import router as ingest_pg_router
from .api.routes_chat_pg import router as chat_pg_router

app = FastAPI(title="Financial Advisor Bot — Modular", version="0.2.0")

@app.get("/")
async def root():
    return {"ok": True, "service": app.title, "version": app.version}

app.include_router(ingest_pg_router, prefix="")
app.include_router(chat_pg_router, prefix="")