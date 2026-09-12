"""Cost estimation: prices come from data, never from the source tree."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from config.pricing import PricingStore, price_key
from core.constants import LLMProviderType
from core.models import ModelPrice

_CATALOGUE = {
    "gpt-4o": {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1e-05,
    },
    "claude-sonnet-4-5": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
    },
    "gemini/gemini-2.5-flash": {
        "litellm_provider": "gemini",
        "mode": "chat",
        "input_cost_per_token": 3e-07,
        "output_cost_per_token": 2.5e-06,
    },
    "text-embedding-3-small": {
        "litellm_provider": "openai",
        "mode": "embedding",
        "input_cost_per_token": 2e-08,
    },
}


@pytest.fixture
def store(tmp_path, monkeypatch) -> PricingStore:
    instance = PricingStore(tmp_path / "pricing.json")

    async def fake_get(self, url, headers=None):
        request = httpx.Request("GET", url)
        return httpx.Response(200, json=_CATALOGUE, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    return instance


async def test_prices_are_downloaded_and_normalised_per_million_tokens(store) -> None:
    count = await store.refresh()

    price = store.resolve(LLMProviderType.OPENAI, "gpt-4o")

    assert count == 3  # the embedding model is not a chat model
    assert price is not None
    assert price.input == pytest.approx(2.5)
    assert price.output == pytest.approx(10.0)
    assert price.source == "catalogue"


async def test_the_cost_follows_the_tokens_the_provider_reported(store) -> None:
    await store.refresh()

    price = store.resolve(LLMProviderType.OPENAI, "gpt-4o")

    assert price.cost(1_000_000, 0) == pytest.approx(2.5)
    assert price.cost(0, 500_000) == pytest.approx(5.0)


async def test_a_dated_model_revision_inherits_its_family_price(store) -> None:
    await store.refresh()

    price = store.resolve(LLMProviderType.ANTHROPIC, "claude-sonnet-4-5-20250929")

    assert price is not None
    assert price.input == pytest.approx(3.0)


async def test_provider_qualified_catalogue_keys_are_matched(store) -> None:
    await store.refresh()

    assert price_key(LLMProviderType.GOOGLE, "gemini/gemini-2.5-flash") == (
        "google/gemini-2.5-flash"
    )
    assert store.resolve(LLMProviderType.GOOGLE, "gemini-2.5-flash") is not None


def test_a_manual_override_wins_over_every_downloaded_price(store) -> None:
    key = price_key(LLMProviderType.GROQ, "llama-3.3-70b-versatile")
    overrides = {key: ModelPrice(input=1.0, output=2.0)}

    price = store.resolve(LLMProviderType.GROQ, "llama-3.3-70b-versatile", overrides)

    assert price.source == "override"
    assert price.cost(1_000_000, 1_000_000) == pytest.approx(3.0)


def test_the_offline_table_covers_the_first_run(store) -> None:
    price = store.resolve(LLMProviderType.OPENAI, "gpt-4o-mini")

    assert price is not None
    assert price.source == "offline"


def test_local_models_are_free(store) -> None:
    price = store.resolve(LLMProviderType.OLLAMA, "llama3.2")

    assert price.cost(10_000_000, 10_000_000) == 0.0


def test_an_unknown_model_reports_no_price_instead_of_guessing(store) -> None:
    assert store.resolve(LLMProviderType.OPENAI, "some-unreleased-model") is None


async def test_a_download_failure_keeps_the_previous_prices(tmp_path, monkeypatch) -> None:
    from core.exceptions import ScientificRAGError

    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "prices": {"openai/gpt-4o": {"input": 2.5, "output": 10.0, "source": "catalogue"}},
            }
        ),
        encoding="utf-8",
    )

    async def failing_get(self, url, headers=None):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx.AsyncClient, "get", failing_get)
    store = PricingStore(path)

    with pytest.raises(ScientificRAGError):
        await store.refresh()

    assert store.resolve(LLMProviderType.OPENAI, "gpt-4o").input == pytest.approx(2.5)


def test_a_cached_catalogue_expires(tmp_path) -> None:
    path = tmp_path / "pricing.json"
    old = datetime.now(timezone.utc) - timedelta(days=365)
    path.write_text(
        json.dumps({"updated_at": old.isoformat(), "prices": {}}), encoding="utf-8"
    )

    assert PricingStore(path).is_stale()
