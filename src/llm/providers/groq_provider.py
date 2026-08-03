"""Groq provider adapter."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config.catalog import provider_entry
from core.constants import LLMProviderType
from llm.base import ModelInfo, ProviderAdapter


class GroqAdapter(ProviderAdapter):
    """Chat models hosted on Groq's inference platform."""

    provider_type = LLMProviderType.GROQ

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        streaming: bool,
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=self.require_api_key(),
            temperature=temperature,
            max_tokens=max_tokens,
            # LLMClient owns the retry policy. ChatGroq otherwise retries each
            # request internally and the outer client retries that whole
            # sequence again, multiplying a transient failure into as many as
            # nine network attempts.
            max_retries=0,
            model_kwargs={"top_p": top_p},
            streaming=streaming,
            **kwargs,
        )

    async def discover_models(self) -> list[ModelInfo]:
        endpoint = provider_entry(self.provider_type).get(
            "models_endpoint", "https://api.groq.com/openai/v1/models"
        )
        payload = await self.http_json(
            endpoint, headers={"Authorization": f"Bearer {self.require_api_key()}"}
        )
        return [
            ModelInfo(
                id=item["id"],
                provider=self.provider_type,
                context_window=item.get("context_window"),
                details={"owned_by": item.get("owned_by", "")},
            )
            for item in payload.get("data", [])
            if item.get("id") and item.get("active", True)
        ]
