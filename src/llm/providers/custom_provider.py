"""Adapter for any third-party service that speaks the OpenAI API.

One endpoint, supplied by the user: an aggregator such as OpenRouter or
Together, a self-hosted vLLM/LiteLLM proxy, or a company gateway. Nothing
about the service is assumed beyond the OpenAI wire format, so the model list
comes from the endpoint itself and no model name is ever filtered out.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel

from core.constants import LLMProviderType
from core.exceptions import ProviderUnavailableError
from llm.base import ModelInfo, ProviderAdapter

# Servers that authenticate by other means still reject an empty credential.
_PLACEHOLDER_KEY = "not-required"


class CustomOpenAIAdapter(ProviderAdapter):
    """Chat models served by a user-configured OpenAI-compatible endpoint."""

    provider_type = LLMProviderType.CUSTOM

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
            base_url=self.endpoint(),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            streaming=streaming,
            # LLMClient owns the retry policy; nested retries would multiply it.
            max_retries=0,
            **kwargs,
        )

    async def discover_models(self) -> list[ModelInfo]:
        payload = await self.http_json(f"{self.endpoint()}/models", headers=self.headers())
        items = payload.get("data")
        if items is None and isinstance(payload.get("models"), list):
            items = payload["models"]

        return [
            ModelInfo(
                id=str(item["id"]),
                label=str(item.get("name") or ""),
                provider=self.provider_type,
                context_window=item.get("context_length") or item.get("context_window"),
            )
            for item in (items or [])
            if isinstance(item, dict) and item.get("id")
        ]

    # -- helpers ---------------------------------------------------------

    def endpoint(self) -> str:
        """Return the validated base URL of the configured service."""
        url = (self.config.base_url or "").strip().rstrip("/")
        if not url:
            raise ProviderUnavailableError(
                self.provider_type.label,
                "Set the server URL of your OpenAI-compatible service, for example "
                "https://openrouter.ai/api/v1.",
            )

        parsed = urlparse(url)
        # The API key is sent to whatever this points at, so refuse anything
        # that is not a plain HTTP(S) address.
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ProviderUnavailableError(
                self.provider_type.label,
                "The server URL must be a complete http:// or https:// address.",
            )
        return url

    def headers(self) -> dict[str, str]:
        """Return the authorisation header, when a key was configured."""
        key = (self.config.api_key or "").strip()
        return {"Authorization": f"Bearer {key}"} if key else {}
