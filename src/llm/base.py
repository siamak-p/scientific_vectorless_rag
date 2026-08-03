"""Provider-agnostic abstractions for language model backends.

Every concrete backend implements :class:`ProviderAdapter`. Nothing outside
``llm/`` may import a provider specific package, which keeps the workflow
completely provider agnostic and makes new backends a drop-in addition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from config.catalog import fallback_models, model_filters
from core.constants import HTTP_TIMEOUT_SECONDS, HTTP_USER_AGENT, LLMProviderType
from core.exceptions import APIKeyMissingError
from core.models import ProviderSettings
from observability.logger import get_logger, log_event

logger = get_logger("llm.base")


class ModelInfo(BaseModel):
    """A model offered by a provider."""

    id: str
    label: str = ""
    provider: LLMProviderType
    context_window: int | None = None
    is_local: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    def display(self) -> str:
        """Return the string shown in model selectors."""
        return self.label or self.id


class ProviderAdapter(ABC):
    """Base class for every LLM backend."""

    provider_type: LLMProviderType

    def __init__(self, config: ProviderSettings) -> None:
        self.config = config

    # -- capabilities ----------------------------------------------------

    @abstractmethod
    def create_chat_model(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        streaming: bool,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Return a configured LangChain chat model."""

    @abstractmethod
    async def discover_models(self) -> list[ModelInfo]:
        """Return the models currently available on this backend."""

    # -- shared helpers --------------------------------------------------

    def require_api_key(self) -> str:
        """Return the configured API key or raise a user-facing error."""
        key = (self.config.api_key or "").strip()
        if not key:
            raise APIKeyMissingError(self.provider_type.label)
        return key

    def base_url(self) -> str:
        """Return the effective endpoint for this provider."""
        return self.config.effective_base_url()

    async def list_models(self) -> list[ModelInfo]:
        """Discover models, degrading gracefully to the catalogue fallback."""
        try:
            models = await self.discover_models()
        except APIKeyMissingError:
            raise
        except Exception as exc:  # noqa: BLE001 - discovery must never break the UI
            log_event(
                logger,
                "llm.discover",
                f"Model discovery failed for {self.provider_type.value}",
                level=30,
                provider=self.provider_type.value,
                error=str(exc),
            )
            models = []

        models = self.filter_models(models)
        if models:
            return models

        return [
            ModelInfo(id=name, provider=self.provider_type, is_local=self.provider_type.is_local)
            for name in fallback_models(self.provider_type)
        ]

    def filter_models(self, models: list[ModelInfo]) -> list[ModelInfo]:
        """Drop models that cannot be used for chat completion."""
        include, exclude = model_filters(self.provider_type)
        result: list[ModelInfo] = []
        for model in models:
            name = model.id.lower()
            if exclude and any(token in name for token in exclude):
                continue
            if include and not any(name.startswith(prefix) for prefix in include):
                continue
            result.append(model)
        return sorted(result, key=lambda m: m.id)

    @staticmethod
    async def http_json(
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform a GET request returning JSON, with a shared timeout policy."""
        merged = {"User-Agent": HTTP_USER_AGENT, **(headers or {})}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=merged, params=params)
            response.raise_for_status()
            return response.json()
