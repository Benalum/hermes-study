from __future__ import annotations

from typing import Any

import httpx


class OllamaWebSearch:
    """Optional external research via Ollama's hosted web-search API.

    It is disabled unless an OLLAMA_API_KEY is configured. Course material remains the
    primary authority; web results should only supplement it.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post("https://ollama.com/api/web_search", headers=headers, json={"query": query, "max_results": max_results})
            r.raise_for_status()
            data = r.json()
        return list(data.get("results", []))
