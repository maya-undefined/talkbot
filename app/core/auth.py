from fastapi import HTTPException
from ..store.memory import TENANTS

async def require_tenant(tenant_id: str | None = None) -> str:
    if tenant_id and tenant_id not in TENANTS:
        raise HTTPException(404, detail="tenant not found")
    return tenant_id or "t1"
