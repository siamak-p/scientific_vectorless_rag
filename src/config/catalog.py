"""Loader for the data-driven provider catalogue (``providers.yaml``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from core.constants import LLMProviderType
from observability.logger import get_logger, log_event

logger = get_logger("config.catalog")

CATALOG_PATH = Path(__file__).resolve().parent / "providers.yaml"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    """Read and cache the provider catalogue."""
    try:
        data = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log_event(logger, "catalog.load", "Provider catalogue unreadable",
                  level=40, error=str(exc))
        return {"providers": {}, "fallback_models": {}, "search_providers": {}}
    return data


def provider_entry(provider_type: LLMProviderType) -> dict[str, Any]:
    """Return the catalogue entry for one LLM provider."""
    return load_catalog().get("providers", {}).get(provider_type.value, {})


def fallback_models(provider_type: LLMProviderType) -> list[str]:
    """Return the offline fallback model list for a provider."""
    return list(load_catalog().get("fallback_models", {}).get(provider_type.value, []))


def model_filters(provider_type: LLMProviderType) -> tuple[list[str], list[str]]:
    """Return ``(include_prefixes, exclude_contains)`` filters for a provider."""
    filters = provider_entry(provider_type).get("model_filters", {}) or {}
    return (
        list(filters.get("include_prefixes", []) or []),
        list(filters.get("exclude_contains", []) or []),
    )


def price_table(provider_type: LLMProviderType) -> dict[str, dict[str, float]]:
    """Return the USD-per-million-token price table for a provider."""
    return provider_entry(provider_type).get("prices", {}) or {}


def search_endpoint(provider_key: str) -> str:
    """Return the configured HTTP endpoint of a scientific search provider."""
    entry = load_catalog().get("search_providers", {}).get(provider_key, {}) or {}
    return str(entry.get("endpoint", ""))
