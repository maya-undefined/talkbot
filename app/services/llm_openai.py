from typing import List, Dict, Any
from .openai_client import OPENAI

CHAT_MODEL = "gpt-4o-mini"  # fast, good for tool use; upgrade per budget

async def answer_with_openai(system: str, messages: List[Dict[str, str]], context_blocks: List[Dict[str, Any]], tool_results: List[Dict[str, Any]]):
    sys = {"role": "system", "content": system}
    ctx_text = []

    # print(f"[DEBUG] {context_blocks}")

    for i, c in enumerate(context_blocks[:6], 1):
        title = c["meta"].get("title", c["doc_id"]) ; page = c["meta"].get("page", "?")
        snippet = c["text"][:50000]
        ctx_text.append(f"[{i}] {title} p.{page}: {snippet}")
    
    ctx_blob = "".join(ctx_text)
    content_prefix = (
        "Use the following sources listed to answer. Always cite with [n] referring to the list below." 
        + ctx_blob + ""
    )
    
    # print(f"[DEBUG] {ctx_blob}")
    
    user_aug = []
    for m in messages:
        if m["role"] == "user":
            user_aug.append({"role": "user", "content": content_prefix + m["content"]})
        else:
            user_aug.append(m)
    return await OPENAI.responses(CHAT_MODEL, [sys] + user_aug)