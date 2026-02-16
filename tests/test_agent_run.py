"""Tests for the /agent/run ABI scaffold endpoint."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_agent_run_returns_abi_shape() -> None:
    payload = {
        "session_id": "s1",
        "tenant_id": "t1",
        "persona_id": "budget_coach",
        "messages": [{"role": "user", "content": "Hello"}],
        "capabilities_manifest": {"tools": ["doc_search", "calc"]},
        "policy": {"max_steps": 3},
    }

    response = client.post("/agent/run", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"assistant_message", "actions", "memory_writes", "trace_id"}
    assert isinstance(body["assistant_message"], str)
    assert isinstance(body["actions"], list)
    assert isinstance(body["memory_writes"], list)
    assert isinstance(body["trace_id"], str)


def test_agent_run_rejects_unknown_persona() -> None:
    payload = {
        "session_id": "s1",
        "tenant_id": "t1",
        "persona_id": "unknown_persona",
        "messages": [{"role": "user", "content": "Hello"}],
        "capabilities_manifest": {},
        "policy": {},
    }

    response = client.post("/agent/run", json=payload)

    assert response.status_code == 404
    assert response.json()["detail"] == "persona not found"
