"""Anthropic Claude provider adapter."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config.catalog import provider_entry
from core.constants import LLMProviderType
from llm.base import ModelInfo, ProviderAdapter

_ANTHROPIC_API_VERSION = "2023-06-01"


class ClaudeAdapter(ProviderAdapter):
    """Chat models served by the Anthropic Messages API."""

    provider_type = LLMProviderType.ANTHROPIC

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        streaming: bool,
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        params: dict[str, Any] = {
            "model": model,
            "api_key": self.require_api_key(),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "streaming": streaming,
            **kwargs,
        }
        if self.config.base_url:
            params["base_url"] = self.config.base_url
        return ChatAnthropic(**params)

    async def discover_models(self) -> list[ModelInfo]:
        endpoint = provider_entry(self.provider_type).get(
            "models_endpoint", "https://api.anthropic.com/v1/models"
        )
        payload = await self.http_json(
            endpoint,
            headers={
                "x-api-key": self.require_api_key(),
                "anthropic-version": _ANTHROPIC_API_VERSION,
            },
            params={"limit": "100"},
        )
        return [
            ModelInfo(
                id=item["id"],
                label=item.get("display_name", ""),
                provider=self.provider_type,
            )
            for item in payload.get("data", [])
            if item.get("id")
        ]
