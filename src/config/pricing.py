"""Model prices resolved from data instead of from the source tree.

No LLM provider publishes its rates through its API — the models endpoint
returns identifiers and context windows, never money — so a price has to come
from somewhere else. This module resolves one, in order:

1. a manual override the user typed on the Settings page,
2. the published price catalogue configured in ``providers.yaml``, downloaded
   once and cached in the application home,
3. the small offline table bundled in ``providers.yaml``, used only until that
   download has happened.

Token counts themselves are never guessed here: they come from the provider's
own usage metadata (see :mod:`llm.usage`).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from config.catalog import price_table, pricing_refresh_days, pricing_source
from core.constants import (
    APP_HOME,
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
    LLMProviderType,
)
from core.exceptions import ConfigurationError
from core.models import ModelPrice
from observability.logger import get_logger, log_event

logger = get_logger("config.pricing")

PRICING_FILE: Path = APP_HOME / "pricing.json"

# The catalogue names providers with its own vocabulary.
_PROVIDER_ALIASES: dict[str, LLMProviderType] = {
    "openai": LLMProviderType.OPENAI,
    "text-completion-openai": LLMProviderType.OPENAI,
    "anthropic": LLMProviderType.ANTHROPIC,
    "gemini": LLMProviderType.GOOGLE,
    "vertex_ai-language-models": LLMProviderType.GOOGLE,
    "groq": LLMProviderType.GROQ,
    "ollama": LLMProviderType.OLLAMA,
}

_CHAT_MODES = {"chat", "completion", "responses"}


def price_key(provider: LLMProviderType, model: str) -> str:
    """Return the catalogue key identifying one model of one provider."""
    name = model.strip().lower()
    if "/" in name and name.split("/", 1)[0] in _PROVIDER_ALIASES:
        name = name.split("/", 1)[1]
    return f"{provider.value}/{name}"


class PricingStore:
    """Downloads, caches and resolves per-model prices."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or PRICING_FILE
        self._lock = threading.Lock()
        self._prices: dict[str, ModelPrice] | None = None
        self._updated_at: datetime | None = None

    # -- state -----------------------------------------------------------

    @property
    def updated_at(self) -> datetime | None:
        """When the cached catalogue was downloaded, if it ever was."""
        self._ensure_loaded()
        return self._updated_at

    def count(self) -> int:
        """How many models the cached catalogue prices."""
        self._ensure_loaded()
        return len(self._prices or {})

    def is_stale(self) -> bool:
        """Whether the cached catalogue is missing or older than configured."""
        updated = self.updated_at
        if updated is None:
            return True
        return datetime.now(timezone.utc) - updated > timedelta(days=pricing_refresh_days())

    # -- resolution ------------------------------------------------------

    def resolve(
        self,
        provider: LLMProviderType,
        model: str,
        overrides: dict[str, ModelPrice] | None = None,
    ) -> ModelPrice | None:
        """Return the price of a model, or ``None`` when it is unknown."""
        if provider.is_local:
            return ModelPrice(source="local")

        key = price_key(provider, model)
        override = (overrides or {}).get(key)
        if override is not None:
            return override.model_copy(update={"source": "override"})

        self._ensure_loaded()
        downloaded = self._match(self._prices or {}, key)
        if downloaded is not None:
            return downloaded

        return self._offline_price(provider, model)

    @staticmethod
    def _match(prices: dict[str, ModelPrice], key: str) -> ModelPrice | None:
        """Look a key up exactly, then by the longest related model name.

        Providers version their models (``claude-3-5-sonnet-20241022``), so a
        dated or aliased revision inherits the price of the family it belongs
        to instead of silently costing nothing.
        """
        exact = prices.get(key)
        if exact is not None:
            return exact

        provider_prefix, _, name = key.partition("/")
        best: tuple[int, ModelPrice] | None = None
        for candidate_key, price in prices.items():
            candidate_provider, _, candidate = candidate_key.partition("/")
            if candidate_provider != provider_prefix:
                continue
            if not (name.startswith(candidate) or candidate.startswith(name)):
                continue
            length = len(candidate)
            if best is None or length > best[0]:
                best = (length, price)
        return best[1] if best else None

    @staticmethod
    def _offline_price(provider: LLMProviderType, model: str) -> ModelPrice | None:
        table = price_table(provider)
        name = model.lower()
        best_key = ""
        for key in table:
            lowered = key.lower()
            if lowered in name and len(lowered) > len(best_key):
                best_key = lowered
        if not best_key:
            return None

        entry = next(value for key, value in table.items() if key.lower() == best_key)
        return ModelPrice(
            input=float(entry.get("input", 0.0)),
            output=float(entry.get("output", 0.0)),
            source="offline",
        )

    # -- refreshing ------------------------------------------------------

    async def refresh(self) -> int:
        """Download the price catalogue and cache it. Returns the model count."""
        url = pricing_source()
        if not url:
            raise ConfigurationError("No price catalogue is configured in providers.yaml.")

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = await client.get(url, headers={"User-Agent": HTTP_USER_AGENT})
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, OSError, ValueError) as exc:
            raise ConfigurationError(
                "The price catalogue could not be downloaded. "
                "Prices from the offline table are used instead.",
                details=str(exc),
            ) from exc

        prices = self._parse(payload)
        if not prices:
            raise ConfigurationError("The downloaded price catalogue contained no usable prices.")

        self._write(prices)
        log_event(logger, "pricing.refresh", "Price catalogue updated",
                  source=url, models=len(prices))
        return len(prices)

    @staticmethod
    def _parse(payload: Any) -> dict[str, ModelPrice]:
        """Normalise the published catalogue into per-million-token prices."""
        if not isinstance(payload, dict):
            return {}

        prices: dict[str, ModelPrice] = {}
        for raw_key, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            provider = _PROVIDER_ALIASES.get(str(entry.get("litellm_provider", "")))
            if provider is None:
                continue
            if str(entry.get("mode", "chat")) not in _CHAT_MODES:
                continue

            input_cost = entry.get("input_cost_per_token")
            output_cost = entry.get("output_cost_per_token")
            if input_cost is None and output_cost is None:
                continue

            try:
                price = ModelPrice(
                    input=float(input_cost or 0.0) * 1_000_000,
                    output=float(output_cost or 0.0) * 1_000_000,
                    source="catalogue",
                )
            except (TypeError, ValueError):
                continue

            prices[price_key(provider, str(raw_key))] = price
        return prices

    # -- persistence -----------------------------------------------------

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._prices is not None:
                return
            self._prices, self._updated_at = self._read()

    def _read(self) -> tuple[dict[str, ModelPrice], datetime | None]:
        if not self._path.is_file():
            return {}, None
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            prices = {
                key: ModelPrice.model_validate(value)
                for key, value in (raw.get("prices") or {}).items()
            }
            updated = raw.get("updated_at")
            return prices, datetime.fromisoformat(updated) if updated else None
        except (OSError, ValueError) as exc:
            log_event(logger, "pricing.load", "Cached price catalogue is unreadable",
                      level=30, error=str(exc))
            return {}, None

    def _write(self, prices: dict[str, ModelPrice]) -> None:
        updated = datetime.now(timezone.utc)
        payload = {
            "updated_at": updated.isoformat(),
            "source": pricing_source(),
            "prices": {key: price.model_dump() for key, price in prices.items()},
        }
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(payload), encoding="utf-8")
            except OSError as exc:
                log_event(logger, "pricing.save", "Price catalogue could not be cached",
                          level=30, error=str(exc))
            self._prices = prices
            self._updated_at = updated


pricing_store = PricingStore()
