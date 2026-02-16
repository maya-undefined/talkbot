"""OpenAI chat completion wrapper for retrieval-grounded answers."""

from __future__ import annotations

from typing import Any, Dict, List

from .openai_client import OPENAI

CHAT_MODEL = "gpt-4o-mini"  # fast, good for tool use; upgrade per budget


def _build_context_prefix(context_blocks: List[Dict[str, Any]]) -> str:
    """Build a compact retrieval prefix for grounding and citation instructions."""

    if not context_blocks:
        return ""

    ctx_text: list[str] = []
    for i, block in enumerate(context_blocks[:6], 1):
        title = block["meta"].get("title", block["doc_id"])
        page = block["meta"].get("page", "?")
        snippet = block["text"][:50000]
        ctx_text.append(f"[{i}] {title} p.{page}: {snippet}\n")

    return (
        "Use the following sources to answer. "
        "Always cite with [n] referring to this source list.\n"
        + "".join(ctx_text)
        + "\n"
    )


def _augment_messages_with_context(
    messages: List[Dict[str, str]], context_prefix: str
) -> List[Dict[str, str]]:
    """Attach retrieval context to only the latest user message."""

    if not context_prefix:
        return messages

    last_user_index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), None)
    if last_user_index is None:
        return messages

    augmented = list(messages)
    last_user = dict(augmented[last_user_index])
    last_user["content"] = f"{context_prefix}{last_user['content']}"
    augmented[last_user_index] = last_user
    return augmented


async def answer_with_openai(
    system: str,
    messages: List[Dict[str, str]],
    context_blocks: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
) -> str:
    """Call OpenAI chat completions with retrieval context and optional tool results."""

    _ = tool_results
    sys = {"role": "system", "content": system}
    context_prefix = _build_context_prefix(context_blocks)
    user_augmented_messages = _augment_messages_with_context(messages, context_prefix)
    return await OPENAI.responses(CHAT_MODEL, [sys] + user_augmented_messages)
