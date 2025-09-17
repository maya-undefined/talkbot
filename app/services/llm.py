from typing import List, Dict, Any
from ..core.policies import POLICY_PACKS
from ..store.memory import TENANTS
from ..models.schemas import Message


def sanitize_retrieved_text(tenant_id: str, text: str) -> str:
    rules = POLICY_PACKS[TENANTS[tenant_id]["policy_pack_id"]]["prompt_injection"]
    out = text
    for marker in rules["strip_markers"]:
        out = out.replace(marker, "")
    return out

class LLM:
    def complete(self, system: str, messages: List[Dict[str, str]], context: List[dict], tool_results: List[dict]) -> str:
        lines = []
        lines.append(f"SYSTEM: {system[:140]}…\n")
        user_last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        lines.append(f"Q: {user_last}\n")
        if context:
            lines.append("\nRetrieved context (top):\n")
            for i, c in enumerate(context[:3], 1):
                title = c["meta"].get("title", c["doc_id"]); page = c["meta"].get("page", "?")
                preview = (c["text"] or "")[:180].replace("\n", " ")
                lines.append(f"  [{i}] {title} p.{page}: {preview} …\n")
        if tool_results:
            lines.append("\nTool results:\n")
            for tr in tool_results:
                lines.append(f"  - {tr['name']}: {tr['output']}\n")
        lines.append("\nAnswer (demo): Based on the retrieved statements and calculations above, here’s a concise, grounded response with citations [1]-[3]. ")
        lines.append("\nDisclosure: This information is for educational purposes and is not financial advice.")
        return "".join(lines)

LLM_ENGINE = LLM()
