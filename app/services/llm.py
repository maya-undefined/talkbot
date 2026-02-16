from typing import Dict, List

from ..core.policies import POLICY_PACKS
from ..store.memory import TENANTS


def sanitize_retrieved_text(tenant_id: str, text: str) -> str:
    rules = POLICY_PACKS[TENANTS[tenant_id]["policy_pack_id"]]["prompt_injection"]
    out = text
    for marker in rules["strip_markers"]:
        out = out.replace(marker, "")
    return out


class LLM:
    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        context: List[dict],
        tool_results: List[dict],
    ) -> str:
        _ = system
        lines = []
        user_last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if user_last:
            lines.append(f"You asked: {user_last}\n")
        if context:
            lines.append("\nSupporting sources:\n")
            for i, c in enumerate(context[:3], 1):
                title = c["meta"].get("title", c["doc_id"])
                page = c["meta"].get("page", "?")
                preview = (c["text"] or "")[:180].replace("\n", " ")
                lines.append(f"  [{i}] {title} p.{page}: {preview} …\n")
        if tool_results:
            lines.append("\nTool results:\n")
            for tr in tool_results:
                lines.append(f"  - {tr['name']}: {tr['output']}\n")
        lines.append(
            "\nI can help with budgeting questions and next-step recommendations. "
            "Please share a bit more detail so I can give a concrete answer."
        )
        lines.append("\nDisclosure: This information is for educational purposes and is not financial advice.")
        return "".join(lines)


LLM_ENGINE = LLM()
