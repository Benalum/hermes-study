from __future__ import annotations

from typing import Any

import httpx


class OllamaClient:
    def __init__(self, base_url: str, chat_model: str, embed_model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout = timeout

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                data = r.json()
            names = [m.get("name", "") for m in data.get("models", [])]
            return {"ok": True, "models": names, "chat_model": self.chat_model, "embed_model": self.embed_model}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "chat_model": self.chat_model, "embed_model": self.embed_model}

    async def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "think": False,
        }
        if json_mode:
            payload["format"] = "json"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        return str(data.get("message", {}).get("content", "")).strip()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embed_model, "input": texts, "truncate": True},
            )
            r.raise_for_status()
            data = r.json()
        return data["embeddings"]
