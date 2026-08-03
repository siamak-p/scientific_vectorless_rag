"""OpenAI provider adapter."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config.catalog import provider_entry
from core.constants import LLMProviderType
from llm.base import ModelInfo, ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    """Chat models served by the OpenAI API."""

    provider_type = LLMProviderType.OPENAI

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
        return ChatOpenAI(**params)

    async def discover_models(self) -> list[ModelInfo]:
        endpoint = provider_entry(self.provider_type).get(
            "models_endpoint", "https://api.openai.com/v1/models"
        )
        payload = await self.http_json(
            endpoint, headers={"Authorization": f"Bearer {self.require_api_key()}"}
        )
        return [
            ModelInfo(id=item["id"], provider=self.provider_type)
            for item in payload.get("data", [])
            if item.get("id")
        ]
