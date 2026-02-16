"""Tests for the Agent Run ABI scaffold endpoint."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_agent_run_returns_abi_shape() -> None:
    """The endpoint should accept the new contract and return required fields."""

    payload = {
        "session_id": "s1",
        "tenant_id": "t1",
        "persona_id": "budget_coach",
        "messages": [{"role": "user", "content": "hello"}],
        "capabilities_manifest": {"tools": ["doc_search", "calc"]},
        "policy": {"max_tool_calls": 2},
    }

    resp = client.post("/agent/run", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert body["assistant_message"]
    assert isinstance(body["actions"], list)
    assert isinstance(body["memory_writes"], list)
    assert isinstance(body["trace_id"], str)


def test_existing_endpoints_still_present_in_openapi() -> None:
    """Ensure legacy endpoints remain mounted during scaffold rollout."""

    paths = client.get("/openapi.json").json()["paths"]
    assert "/ingest_pg" in paths
    assert "/chat_pg" in paths
    assert "/agent/run" in paths
