"""The user-configured OpenAI-compatible endpoint."""

from __future__ import annotations

import pytest

from core.constants import LLMProviderType
from core.exceptions import ScientificRAGError
from core.models import ProviderSettings
from llm.factory import fetch_models, get_adapter
from llm.providers.custom_provider import CustomOpenAIAdapter


def _config(url: str = "https://gateway.example.com/v1", key: str = "secret") -> ProviderSettings:
    return ProviderSettings(provider_type=LLMProviderType.CUSTOM, base_url=url, api_key=key)


def test_the_endpoint_is_whatever_the_user_configured() -> None:
    adapter = get_adapter(LLMProviderType.CUSTOM, config=_config())

    assert isinstance(adapter, CustomOpenAIAdapter)
    assert adapter.endpoint() == "https://gateway.example.com/v1"


def test_a_url_that_is_not_http_is_refused() -> None:
    """The API key is sent to this address, so it must be a plain web address."""
    for url in ("file:///etc/passwd", "gateway.example.com", "ftp://host/v1"):
        with pytest.raises(ScientificRAGError):
            CustomOpenAIAdapter(_config(url)).endpoint()


def test_a_missing_url_is_explained_rather_than_guessed() -> None:
    with pytest.raises(ScientificRAGError):
        CustomOpenAIAdapter(_config(url="")).endpoint()


async def test_models_are_listed_from_that_endpoint(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def fake_http_json(url, headers=None, params=None):
        seen["url"] = url
        seen["headers"] = headers
        return {
            "data": [
                {"id": "deepseek-ai/DeepSeek-V3", "context_length": 64000},
                {"id": "meta-llama/Llama-3.3-70B-Instruct"},
            ]
        }

    monkeypatch.setattr("llm.base.ProviderAdapter.http_json", staticmethod(fake_http_json))

    catalog = await fetch_models(LLMProviderType.CUSTOM, config=_config())

    assert seen["url"] == "https://gateway.example.com/v1/models"
    assert seen["headers"] == {"Authorization": "Bearer secret"}
    assert catalog.live
    # No name is filtered out: a custom service may serve anything.
    assert [model.id for model in catalog.models] == [
        "deepseek-ai/DeepSeek-V3",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]


async def test_a_server_without_authentication_is_supported(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def fake_http_json(url, headers=None, params=None):
        seen["headers"] = headers
        return {"data": [{"id": "local-model"}]}

    monkeypatch.setattr("llm.base.ProviderAdapter.http_json", staticmethod(fake_http_json))

    catalog = await fetch_models(LLMProviderType.CUSTOM, config=_config(key=""))

    assert seen["headers"] == {}
    assert [model.id for model in catalog.models] == ["local-model"]


async def test_an_unreachable_endpoint_leaves_the_model_name_to_be_typed(monkeypatch) -> None:
    async def fake_http_json(url, headers=None, params=None):
        raise OSError("no route to host")

    monkeypatch.setattr("llm.base.ProviderAdapter.http_json", staticmethod(fake_http_json))

    catalog = await fetch_models(LLMProviderType.CUSTOM, config=_config())

    assert catalog.models == []
    assert catalog.error


def test_the_provider_is_neither_local_nor_forced_to_have_a_key() -> None:
    provider = LLMProviderType.CUSTOM

    assert provider.requires_base_url
    assert provider.accepts_api_key
    assert not provider.requires_api_key
    assert not provider.is_local
