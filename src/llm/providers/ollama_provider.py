"""Ollama local provider adapter."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel

from core.constants import DEFAULT_OLLAMA_URL, LLMProviderType
from core.exceptions import LocalProviderUnavailableError
from llm.base import ModelInfo, ProviderAdapter


class OllamaAdapter(ProviderAdapter):
    """Models served by a locally running Ollama daemon."""

    provider_type = LLMProviderType.OLLAMA

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        streaming: bool,
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=self.base_url() or DEFAULT_OLLAMA_URL,
            temperature=temperature,
            num_predict=max_tokens,
            top_p=top_p,
            **kwargs,
        )

    async def discover_models(self) -> list[ModelInfo]:
        url = (self.base_url() or DEFAULT_OLLAMA_URL).rstrip("/")
        try:
            payload = await self.http_json(f"{url}/api/tags")
        except (httpx.HTTPError, OSError) as exc:
            raise LocalProviderUnavailableError(self.provider_type.label, url) from exc

        models: list[ModelInfo] = []
        for item in payload.get("models", []):
            name = item.get("name") or item.get("model")
            if not name:
                continue
            details = item.get("details") or {}
            models.append(
                ModelInfo(
                    id=name,
                    label=name,
                    provider=self.provider_type,
                    is_local=True,
                    details={
                        "parameter_size": details.get("parameter_size", ""),
                        "quantization": details.get("quantization_level", ""),
                        "size_bytes": item.get("size", 0),
                    },
                )
            )
        return models
