"""Token accounting and cost estimation."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler

from config.catalog import price_table
from core.constants import LLMProviderType
from core.models import TokenUsage


def estimate_cost(provider: LLMProviderType, model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a call using the catalogue price table.

    Prices are expressed per one million tokens and matched by the longest
    model-name prefix, so newly released model revisions inherit the price of
    their family. Local providers always cost zero.
    """
    if provider.is_local:
        return 0.0

    table = price_table(provider)
    if not table:
        return 0.0

    name = model.lower()
    best_key = ""
    for key in table:
        lowered = key.lower()
        if lowered in name and len(lowered) > len(best_key):
            best_key = lowered

    if not best_key:
        return 0.0

    prices = next(v for k, v in table.items() if k.lower() == best_key)
    cost = (input_tokens / 1_000_000) * float(prices.get("input", 0.0))
    cost += (output_tokens / 1_000_000) * float(prices.get("output", 0.0))
    return round(cost, 6)


class UsageTracker:
    """Accumulates token usage across every LLM call of a single request."""

    def __init__(self, provider: LLMProviderType) -> None:
        self._provider = provider
        self._handler = UsageMetadataCallbackHandler()
        self._manual = TokenUsage()

    @property
    def callbacks(self) -> list[Any]:
        """Callbacks to attach to a LangChain runnable invocation."""
        return [self._handler]

    def record_manual(self, input_tokens: int, output_tokens: int, model: str) -> None:
        """Record usage for a call that did not report usage metadata."""
        cost = estimate_cost(self._provider, model, input_tokens, output_tokens)
        self._manual = self._manual.add(
            TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                estimated_cost_usd=cost,
                calls=1,
            )
        )

    def snapshot(self) -> TokenUsage:
        """Return the aggregated usage collected so far."""
        total = TokenUsage()
        for model, usage in (self._handler.usage_metadata or {}).items():
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            total = total.add(
                TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=int(usage.get("total_tokens", input_tokens + output_tokens)),
                    estimated_cost_usd=estimate_cost(
                        self._provider, model, input_tokens, output_tokens
                    ),
                    calls=1,
                )
            )
        return total.add(self._manual)

    def reset(self) -> None:
        """Clear the accumulated usage so the next stage starts from zero."""
        self._handler = UsageMetadataCallbackHandler()
        self._manual = TokenUsage()


def approximate_tokens(text: str) -> int:
    """Approximate the token count of a string without a provider round trip."""
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001 - tiktoken may lack its data files offline
        return max(1, len(text) // 4)
