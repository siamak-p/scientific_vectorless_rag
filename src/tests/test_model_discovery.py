"""Model discovery: unsaved credentials, complete listings and failure reporting."""

from __future__ import annotations

import httpx

from config.catalog import fallback_models
from config.settings import SettingsManager
from core.constants import LLMProviderType
from core.models import AppSettings, ProviderSettings
from llm.base import ModelInfo
from llm.factory import fetch_models, get_adapter
from llm.providers.claude_provider import ClaudeAdapter
from llm.providers.gemini_provider import GeminiAdapter


def _config(provider: LLMProviderType, key: str = "typed-key") -> ProviderSettings:
    return ProviderSettings(provider_type=provider, api_key=key)


async def test_discovery_uses_the_key_that_was_just_typed(monkeypatch) -> None:
    """The Settings page must not wait for a save before a key can be tested."""
    seen: list[str] = []

    async def fake_http_json(url, headers=None, params=None):
        seen.append((headers or {}).get("Authorization", ""))
        return {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}

    monkeypatch.setattr("llm.base.ProviderAdapter.http_json", staticmethod(fake_http_json))

    catalog = await fetch_models(
        LLMProviderType.OPENAI, config=_config(LLMProviderType.OPENAI, "unsaved-key")
    )

    assert seen == ["Bearer unsaved-key"]
    assert catalog.live
    assert [model.id for model in catalog.models] == ["gpt-4o", "gpt-4o-mini"]


def test_an_explicit_config_overrides_the_saved_settings() -> None:
    saved = AppSettings()
    saved.provider(LLMProviderType.GROQ).api_key = "stale"

    adapter = get_adapter(
        LLMProviderType.GROQ, saved, config=_config(LLMProviderType.GROQ, "fresh")
    )

    assert adapter.config.api_key == "fresh"


async def test_every_active_model_is_listed(monkeypatch) -> None:
    async def fake_http_json(url, headers=None, params=None):
        return {"data": [{"id": f"model-{index}"} for index in range(40)]}

    monkeypatch.setattr("llm.base.ProviderAdapter.http_json", staticmethod(fake_http_json))

    catalog = await fetch_models(LLMProviderType.GROQ, config=_config(LLMProviderType.GROQ))

    assert len(catalog.models) == 40


async def test_paginated_providers_return_every_page(monkeypatch) -> None:
    pages = [
        {"data": [{"id": "claude-a"}], "has_more": True, "last_id": "claude-a"},
        {"data": [{"id": "claude-b"}], "has_more": False},
    ]
    calls: list[dict] = []

    async def fake_http_json(url, headers=None, params=None):
        calls.append(params or {})
        return pages[len(calls) - 1]

    monkeypatch.setattr("llm.base.ProviderAdapter.http_json", staticmethod(fake_http_json))

    models = await ClaudeAdapter(_config(LLMProviderType.ANTHROPIC)).discover_models()

    assert [model.id for model in models] == ["claude-a", "claude-b"]
    assert calls[1]["after_id"] == "claude-a"


async def test_a_failed_lookup_reports_why_and_falls_back(monkeypatch) -> None:
    async def fake_http_json(url, headers=None, params=None):
        request = httpx.Request("GET", "https://api.openai.com/v1/models")
        raise httpx.HTTPStatusError(
            "Unauthorized", request=request, response=httpx.Response(401, request=request)
        )

    monkeypatch.setattr("llm.base.ProviderAdapter.http_json", staticmethod(fake_http_json))

    catalog = await fetch_models(LLMProviderType.OPENAI, config=_config(LLMProviderType.OPENAI))

    assert not catalog.live
    assert "rejected the API key" in catalog.error
    assert [model.id for model in catalog.models] == fallback_models(LLMProviderType.OPENAI)


async def test_a_missing_key_is_reported_instead_of_raising() -> None:
    catalog = await fetch_models(
        LLMProviderType.OPENAI, config=_config(LLMProviderType.OPENAI, "")
    )

    assert not catalog.live
    assert catalog.error


async def test_the_api_key_is_never_echoed_back_to_the_user(monkeypatch) -> None:
    """Gemini passes the key in the query string, so raw errors would leak it."""

    async def fake_http_json(url, headers=None, params=None):
        raise RuntimeError("failed for url 'https://x/models?key=super-secret'")

    monkeypatch.setattr("llm.base.ProviderAdapter.http_json", staticmethod(fake_http_json))

    catalog = await fetch_models(
        LLMProviderType.GOOGLE, config=_config(LLMProviderType.GOOGLE, "super-secret")
    )

    assert "super-secret" not in catalog.error


def test_gemini_keeps_every_chat_capable_family() -> None:
    adapter = GeminiAdapter(_config(LLMProviderType.GOOGLE))
    discovered = [
        ModelInfo(id=name, provider=LLMProviderType.GOOGLE)
        for name in ("gemini-2.5-pro", "gemma-3-27b-it", "text-embedding-004")
    ]

    kept = {model.id for model in adapter.filter_models(discovered)}

    assert kept == {"gemini-2.5-pro", "gemma-3-27b-it"}


def test_environment_supplied_keys_are_not_written_to_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    manager = SettingsManager(tmp_path / "settings.json")

    settings = manager.load(refresh=True)
    assert settings.provider(LLMProviderType.OPENAI).api_key == "from-environment"

    manager.save(settings)
    saved = (tmp_path / "settings.json").read_text(encoding="utf-8")

    assert "from-environment" not in saved
