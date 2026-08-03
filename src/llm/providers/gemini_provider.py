"""Google Gemini provider adapter."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config.catalog import provider_entry
from core.constants import LLMProviderType
from llm.base import ModelInfo, ProviderAdapter


class GeminiAdapter(ProviderAdapter):
    """Chat models served by the Google Generative Language API."""

    provider_type = LLMProviderType.GOOGLE

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        streaming: bool,
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self.require_api_key(),
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )

    async def discover_models(self) -> list[ModelInfo]:
        endpoint = provider_entry(self.provider_type).get(
            "models_endpoint", "https://generativelanguage.googleapis.com/v1beta/models"
        )
        payload = await self.http_json(
            endpoint, params={"key": self.require_api_key(), "pageSize": "200"}
        )

        models: list[ModelInfo] = []
        for item in payload.get("models", []):
            name = str(item.get("name", ""))
            if not name:
                continue
            if "generateContent" not in (item.get("supportedGenerationMethods") or []):
                continue
            models.append(
                ModelInfo(
                    id=name.removeprefix("models/"),
                    label=item.get("displayName", ""),
                    provider=self.provider_type,
                    context_window=item.get("inputTokenLimit"),
                )
            )
        return models
