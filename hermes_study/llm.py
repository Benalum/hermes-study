from __future__ import annotations

from typing import Any

import httpx


class HermesClient:
    """Small server-to-server client for Hermes Agent's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "hermes-agent",
        session_key: str = "hermes-study",
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model
        self.session_key = session_key
        self.timeout = timeout

    @property
    def server_url(self) -> str:
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.session_key:
            headers["X-Hermes-Session-Key"] = self.session_key
        return headers

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                live = await client.get(f"{self.server_url}/health")
                live.raise_for_status()
                if not self.api_key:
                    return {
                        "ok": False,
                        "reachable": True,
                        "error": "Hermes API is reachable but HERMES_AGENT_API_KEY is not configured.",
                        "model": self.model,
                    }
                models = await client.get(f"{self.base_url}/models", headers=self._headers())
                models.raise_for_status()
                data = models.json()
            names = [m.get("id", "") for m in data.get("data", [])]
            return {"ok": True, "reachable": True, "models": names, "model": self.model}
        except Exception as exc:
            return {"ok": False, "reachable": False, "error": str(exc), "model": self.model}

    async def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Hermes API key is not configured. Run scripts/install_mac.sh so Hermes Study can use the local Hermes API."
            )
        outgoing = [dict(m) for m in messages]
        if json_mode:
            outgoing.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "The calling application requires machine-readable output. Return exactly one valid JSON object, "
                        "with no markdown fence or commentary outside the JSON."
                    ),
                },
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": outgoing,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Hermes returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    text_parts.append(str(item.get("text", "")))
            content = "\n".join(text_parts)
        return str(content).strip()

    async def research(self, query: str) -> str:
        """Ask Hermes to use its configured tools for supplemental research when useful."""
        return await self.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are supporting a study assistant. Research the user's question using your configured tools when "
                        "current or external information would help. Distinguish reliable external information from class-specific "
                        "rules, and never claim an external source overrides a professor, syllabus, or assigned material. Cite the "
                        "sources you actually used in plain text."
                    ),
                },
                {"role": "user", "content": query},
            ]
        )


# Backward-compatible import name for v0.1 Tutor/type hints.
OllamaClient = HermesClient
