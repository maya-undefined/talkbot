"""OpenAI chat completion wrapper for retrieval-grounded answers."""

from __future__ import annotations

from typing import Any, Dict, List

from .openai_client import OPENAI

CHAT_MODEL = "gpt-4o-mini"  # fast, good for tool use; upgrade per budget


def _context_title_and_page(block: Dict[str, Any], index: int) -> tuple[str, str]:
    """Return a display title/page pair for document or memory context blocks."""

    if block.get("type") == "memory":
        memory_type = str(block.get("memory_type") or "memory")
        return f"Memory ({memory_type})", "-"

    meta = block.get("meta") if isinstance(block.get("meta"), dict) else {}
    doc_id = str(block.get("doc_id") or f"source_{index}")
    title = str(meta.get("title") or doc_id)
    page = str(meta.get("page") or "?")
    return title, page


def _context_snippet(block: Dict[str, Any]) -> str:
    """Return a bounded context snippet for mixed retrieval block shapes."""

    if block.get("type") == "memory":
        return str(block.get("content") or "")[:50000]
    return str(block.get("text") or "")[:50000]


def _build_context_prefix(context_blocks: List[Dict[str, Any]]) -> str:
    """Build a compact retrieval prefix for grounding and citation instructions."""

    if not context_blocks:
        return ""

    ctx_text: list[str] = []
    for i, block in enumerate(context_blocks[:6], 1):
        title, page = _context_title_and_page(block, i)
        snippet = _context_snippet(block)
        if not snippet:
            continue
        ctx_text.append(f"[{i}] {title} p.{page}: {snippet}\n")

    if not ctx_text:
        return ""

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
