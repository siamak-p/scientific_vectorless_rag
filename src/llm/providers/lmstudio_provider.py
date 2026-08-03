"""LM Studio local provider adapter.

LM Studio exposes an OpenAI compatible server, so the OpenAI chat client is
reused while discovery talks to the local ``/models`` endpoint.
"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel

from core.constants import DEFAULT_LMSTUDIO_URL, LLMProviderType
from core.exceptions import LocalProviderUnavailableError
from llm.base import ModelInfo, ProviderAdapter

# LM Studio ignores the credential but the OpenAI client requires a non-empty value.
_PLACEHOLDER_KEY = "lm-studio"


class LMStudioAdapter(ProviderAdapter):
    """Models served by a locally running LM Studio server."""

    provider_type = LLMProviderType.LMSTUDIO

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        streaming: bool,
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=self.config.api_key or _PLACEHOLDER_KEY,
            base_url=self.base_url() or DEFAULT_LMSTUDIO_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            streaming=streaming,
            **kwargs,
        )

    async def discover_models(self) -> list[ModelInfo]:
        url = (self.base_url() or DEFAULT_LMSTUDIO_URL).rstrip("/")
        try:
            payload = await self.http_json(f"{url}/models")
        except (httpx.HTTPError, OSError) as exc:
            raise LocalProviderUnavailableError(self.provider_type.label, url) from exc

        return [
            ModelInfo(
                id=item["id"],
                label=item["id"],
                provider=self.provider_type,
                is_local=True,
                details={"owned_by": item.get("owned_by", "")},
            )
            for item in payload.get("data", [])
            if item.get("id")
        ]
