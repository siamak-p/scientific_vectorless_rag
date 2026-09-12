"""Factory for provider adapters and LangChain chat models.

This is the only place in the application that maps a provider identifier to a
concrete implementation, so adding a backend means adding one adapter class and
one registry entry.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import get_settings
from core.constants import LLMProviderType
from core.exceptions import (
    LLMError,
    LLMInvalidCredentialsError,
    LLMRateLimitError,
    LLMTimeoutError,
    ProviderUnavailableError,
)
from core.models import AppSettings, ProviderSettings
from llm.base import ModelCatalog, ModelInfo, ProviderAdapter
from llm.providers.claude_provider import ClaudeAdapter
from llm.providers.custom_provider import CustomOpenAIAdapter
from llm.providers.gemini_provider import GeminiAdapter
from llm.providers.groq_provider import GroqAdapter
from llm.providers.lmstudio_provider import LMStudioAdapter
from llm.providers.ollama_provider import OllamaAdapter
from llm.providers.openai_provider import OpenAIAdapter
from observability.logger import get_logger, log_event

logger = get_logger("llm.factory")

_ADAPTERS: dict[LLMProviderType, type[ProviderAdapter]] = {
    LLMProviderType.OPENAI: OpenAIAdapter,
    LLMProviderType.ANTHROPIC: ClaudeAdapter,
    LLMProviderType.GOOGLE: GeminiAdapter,
    LLMProviderType.GROQ: GroqAdapter,
    LLMProviderType.OLLAMA: OllamaAdapter,
    LLMProviderType.LMSTUDIO: LMStudioAdapter,
    LLMProviderType.CUSTOM: CustomOpenAIAdapter,
}

# Chat models are cheap to reuse and expensive to keep rebuilding: every node
# in a single workflow turn (query understanding, ranking, navigation,
# generation, validation, ...) asks for one, and most ask for the identical
# configuration. Caching by the exact configuration means a genuine settings
# change (model, credentials, temperature, ...) still gets a fresh instance.
#
# A normal turn alternates between non-streaming structured calls and a
# streaming answer. Keeping only one entry made those two configurations
# evict and recreate each other repeatedly. A tiny LRU retains both while
# still bounding clients held after genuine settings/model changes.
_MODEL_CACHE_SIZE = 4
_model_cache: OrderedDict[tuple[Any, ...], BaseChatModel] = OrderedDict()


def register_adapter(
    provider_type: LLMProviderType, adapter_class: type[ProviderAdapter]
) -> None:
    """Register (or replace) the adapter used for a provider."""
    _ADAPTERS[provider_type] = adapter_class


def resolve_provider(provider: LLMProviderType | str | None = None) -> LLMProviderType:
    """Normalise a provider argument, falling back to the active provider."""
    if isinstance(provider, LLMProviderType):
        return provider

    value = (provider or get_settings().active_provider or "").strip()
    if not value:
        raise ProviderUnavailableError(
            "none", "No provider is selected. Choose one on the Settings page."
        )
    try:
        return LLMProviderType(value)
    except ValueError as exc:
        raise ProviderUnavailableError(value, "Unknown provider identifier.") from exc


def get_adapter(
    provider: LLMProviderType | str | None = None,
    settings: AppSettings | None = None,
    config: ProviderSettings | None = None,
) -> ProviderAdapter:
    """Return the adapter instance for a provider.

    ``config`` lets a caller supply credentials that are not persisted yet —
    the Settings page must be able to test a key the moment it is typed.
    """
    provider_type = resolve_provider(provider)
    if config is None:
        app_settings = settings or get_settings()
        config = app_settings.provider(provider_type)
    adapter_class = _ADAPTERS.get(provider_type)
    if adapter_class is None:
        raise ProviderUnavailableError(provider_type.value, "No adapter is registered.")
    return adapter_class(config)


async def fetch_models(
    provider: LLMProviderType | str | None = None,
    settings: AppSettings | None = None,
    config: ProviderSettings | None = None,
) -> ModelCatalog:
    """Return the provider's models together with the outcome of the lookup."""
    return await get_adapter(provider, settings, config).fetch_models()


async def list_models(
    provider: LLMProviderType | str | None = None,
    settings: AppSettings | None = None,
    config: ProviderSettings | None = None,
) -> list[ModelInfo]:
    """Return the models available on a provider."""
    return (await fetch_models(provider, settings, config)).models


def get_llm(
    provider: LLMProviderType | str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    streaming: bool | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a chat model from settings plus optional overrides.

    Identical configurations are cached and reused instead of re-instantiated,
    so a single workflow turn - which asks for a chat model once per node -
    only pays for construction the first time.

    Raises:
        ProviderUnavailableError: The provider is unknown or has no model selected.
        APIKeyMissingError: A required credential is not configured.
    """
    settings = get_settings()
    provider_type = resolve_provider(provider)
    adapter = get_adapter(provider_type, settings)

    model = (model_name or adapter.config.selected_model or "").strip()
    if not model:
        raise ProviderUnavailableError(
            provider_type.label, "No model is selected for this provider."
        )

    resolved_temperature = settings.llm.temperature if temperature is None else temperature
    resolved_max_tokens = settings.llm.max_tokens if max_tokens is None else max_tokens
    resolved_top_p = settings.llm.top_p if top_p is None else top_p
    resolved_streaming = settings.llm.streaming if streaming is None else streaming

    # Both parts matter: a custom endpoint can change either its address or
    # its key, and the cached client must not survive either change.
    credential = (adapter.config.api_key, adapter.config.effective_base_url())
    cache_key = (
        provider_type,
        model,
        resolved_temperature,
        resolved_max_tokens,
        resolved_top_p,
        resolved_streaming,
        credential,
        tuple(sorted(kwargs.items())),
    )
    cached = _model_cache.get(cache_key)
    if cached is not None:
        _model_cache.move_to_end(cache_key)
        return cached

    log_event(
        logger,
        "llm.create",
        "Instantiating chat model",
        provider=provider_type.value,
        model=model,
    )

    chat_model = adapter.create_chat_model(
        model=model,
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        top_p=resolved_top_p,
        streaming=resolved_streaming,
        **kwargs,
    )
    _model_cache[cache_key] = chat_model
    _model_cache.move_to_end(cache_key)
    while len(_model_cache) > _MODEL_CACHE_SIZE:
        _model_cache.popitem(last=False)
    return chat_model


def translate_error(provider: LLMProviderType, exc: Exception) -> LLMError:
    """Convert a provider SDK exception into a user-facing application error."""
    if isinstance(exc, LLMError):
        return exc

    text = str(exc).lower()
    label = provider.label

    if any(marker in text for marker in (
        "rate limit", "rate_limit", "429", "quota", "tokens per minute", "tpm"
    )):
        return LLMRateLimitError(label)
    if "timeout" in text or "timed out" in text:
        return LLMTimeoutError(label, 60.0)
    if any(token in text for token in ("api key", "unauthorized", "401", "invalid_api_key", "403")):
        return LLMInvalidCredentialsError(label)
    if any(token in text for token in ("connection", "connect", "refused", "unreachable", "dns")):
        return ProviderUnavailableError(label, "The provider endpoint could not be reached.")
    if "not found" in text and "model" in text:
        return ProviderUnavailableError(label, "The selected model is not available.")

    return ProviderUnavailableError(label, _readable_reason(provider, exc))


def _readable_reason(provider: LLMProviderType, exc: Exception) -> str:
    """Compact an SDK message into something safe to show to the user.

    Gemini puts the credential in the request URL, so the raw text of a
    transport error can contain the key itself.
    """
    reason = " ".join(str(exc).split())
    key = (get_settings().provider(provider).api_key or "").strip()
    if key:
        reason = reason.replace(key, "***")
    return reason[:300]
