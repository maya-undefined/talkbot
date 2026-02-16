"""Tests for OpenAI message augmentation behavior."""

from __future__ import annotations

import asyncio

from app.services.llm_openai import answer_with_openai


def test_answer_with_openai_augments_only_latest_user_message(monkeypatch) -> None:
    """Retrieval context should be prepended only to the newest user turn."""

    captured: dict[str, object] = {}

    async def fake_responses(model: str, messages: list[dict], tools=None) -> str:
        captured["model"] = model
        captured["messages"] = messages
        captured["tools"] = tools
        return "ok"

    monkeypatch.setattr("app.services.llm_openai.OPENAI.responses", fake_responses)

    messages = [
        {"role": "user", "content": "older question"},
        {"role": "assistant", "content": "older answer"},
        {"role": "user", "content": "latest question"},
    ]
    context = [{"doc_id": "d1", "meta": {"title": "Doc", "page": 2}, "text": "source text"}]

    result = asyncio.run(answer_with_openai("sys", messages, context, []))

    assert result == "ok"
    payload_messages = captured["messages"]
    assert isinstance(payload_messages, list)
    assert payload_messages[0] == {"role": "system", "content": "sys"}
    assert payload_messages[1]["content"] == "older question"
    assert payload_messages[2]["content"] == "older answer"
    assert "Use the following sources" in payload_messages[3]["content"]
    assert payload_messages[3]["content"].endswith("latest question")


def test_answer_with_openai_no_context_keeps_messages_unchanged(monkeypatch) -> None:
    """No retrieval context should leave message content unchanged."""

    captured: dict[str, object] = {}

    async def fake_responses(_model: str, messages: list[dict], tools=None) -> str:
        captured["messages"] = messages
        captured["tools"] = tools
        return "ok"

    monkeypatch.setattr("app.services.llm_openai.OPENAI.responses", fake_responses)

    messages = [{"role": "user", "content": "plain question"}]

    result = asyncio.run(answer_with_openai("sys", messages, [], []))

    assert result == "ok"
    assert captured["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "plain question"},
    ]
