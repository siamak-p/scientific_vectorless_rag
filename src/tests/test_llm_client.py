"""Transient LLM failures should be recovered before reaching the UI."""

from __future__ import annotations

from types import SimpleNamespace

from core.constants import LLMProviderType
from core.exceptions import LLMRateLimitError
from llm.client import LLMClient, _retry_delay
from llm.factory import translate_error


def test_retry_delay_honors_provider_seconds_and_milliseconds() -> None:
    assert _retry_delay(RuntimeError("Please try again in 2.5s"), 1) == 2.75
    assert _retry_delay(RuntimeError("Please try again in 500ms"), 1) == 0.75


def test_tpm_request_limit_is_classified_as_rate_limit() -> None:
    error = RuntimeError(
        "HTTP 413: rate_limit_exceeded; TPM Limit 6000, Requested 9454"
    )

    translated = translate_error(LLMProviderType.GROQ, error)

    assert isinstance(translated, LLMRateLimitError)


class _TransientStreamingModel:
    def __init__(self) -> None:
        self.calls = 0

    async def astream(self, messages, config):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 rate limit; try again in 0ms")
        yield SimpleNamespace(content="grounded answer")


async def test_stream_retries_before_emitting_partial_text(monkeypatch) -> None:
    client = object.__new__(LLMClient)
    monkeypatch.setattr(client, "_config", lambda: {})
    monkeypatch.setattr(client, "_record_manual", lambda messages, output: None)

    async def no_wait(delay: float) -> None:
        return None

    monkeypatch.setattr("llm.client.asyncio.sleep", no_wait)
    model = _TransientStreamingModel()

    chunks = [
        chunk
        async for chunk in client._stream_with_retry(model, [SimpleNamespace(content="question")])
    ]

    assert chunks == ["grounded answer"]
    assert model.calls == 2