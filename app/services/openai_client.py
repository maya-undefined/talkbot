import os
from typing import Any, Dict, List
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY:
    # Allow local dev without a key; callers should handle fallback
    pass

class OpenAIClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")
        self._headers = {"Authorization": f"Bearer {self.api_key}"}

    async def embeddings(self, model: str, input_texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=self._headers, json={"model": model, "input": input_texts})
            r.raise_for_status()
            data = r.json()
            return [item["embedding"] for item in data["data"]]

    async def responses(self, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> str:
        # Minimal wrapper using Chat Completions for broad compatibility
        url = f"{self.base_url}/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": 0.2}
        if tools:
            payload["tools"] = tools
        async with httpx.AsyncClient(timeout=120) as client:
            # print(f"DEBUG  {payload}")
            r = await client.post(url, headers=self._headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

OPENAI = OpenAIClient()