"""High level LLM client used by every node of the workflow.

The client binds together four concerns that would otherwise be duplicated
everywhere: prompt loading, structured output with a robust fallback, token
accounting, and translation of provider errors into user-facing messages.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Iterable, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from config.settings import get_settings
from core.constants import LLMProviderType
from core.models import Chat
from llm.factory import get_llm, resolve_provider, translate_error
from llm.usage import UsageTracker, approximate_tokens
from observability.logger import get_logger, log_event
from prompts.manager import get_prompt_manager

logger = get_logger("llm.client")

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RETRYABLE_MARKERS = ("rate limit", "429", "timeout", "timed out", "overloaded", "503", "502")
_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.5
_MAX_RETRY_DELAY = 60.0
_RETRY_AFTER_RE = re.compile(
    r"(?:try again in|retry[- ]?after[: ]+)\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds?)?",
    re.IGNORECASE,
)

# A single workflow turn can issue a dozen or more structured() calls, one per
# node. Some provider/model combinations never support LangChain's native
# with_structured_output() (e.g. many Groq and local models), so trying it
# first would pay for a doomed extra round trip on every single call. Once a
# (provider, model) pair is seen to fail the native path, remember it for the
# rest of the process so later calls go straight to the JSON-instructions
# fallback instead of repeating the same failure.
_native_structured_unsupported: set[tuple[str, str]] = set()
_native_structured_supported: set[tuple[str, str]] = set()
_native_structured_probe_locks: dict[tuple[str, str], asyncio.Lock] = {}


class LLMClient:
    """A configured, provider-agnostic entry point for prompt execution."""

    def __init__(
        self,
        provider: LLMProviderType | str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        usage: UsageTracker | None = None,
    ) -> None:
        settings = get_settings()
        self.provider = resolve_provider(provider)
        self.model = (model or settings.provider(self.provider).selected_model or "").strip()
        self.temperature = settings.llm.temperature if temperature is None else temperature
        self.max_tokens = settings.llm.max_tokens if max_tokens is None else max_tokens
        self.top_p = settings.llm.top_p if top_p is None else top_p
        self.usage = usage or UsageTracker(self.provider)
        self._prompts = get_prompt_manager()

    # -- construction helpers -------------------------------------------

    @classmethod
    def for_chat(cls, chat: Chat, usage: UsageTracker | None = None) -> "LLMClient":
        """Build a client using the model configuration stored on a chat.

        Provider and model always follow the current Settings selection -
        there is no UI to pin a chat to a different one - so they are never
        read from the chat here. Only the generation parameters captured at
        chat creation time (temperature, max tokens, top-p) are per-chat.
        """
        return cls(
            temperature=chat.temperature,
            max_tokens=chat.max_tokens,
            top_p=chat.top_p,
            usage=usage,
        )

    def with_temperature(self, temperature: float) -> "LLMClient":
        """Return a sibling client that shares the usage tracker."""
        return LLMClient(
            provider=self.provider,
            model=self.model,
            temperature=temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            usage=self.usage,
        )

    # -- public API ------------------------------------------------------

    async def structured(
        self,
        prompt_name: str,
        schema: type[T],
        temperature: float | None = None,
        **values: Any,
    ) -> T:
        """Run a prompt and parse the reply into a Pydantic model.

        Falls back to explicit JSON instructions when the backend does not
        support native structured output, which is common for local models.
        Once that happens for a given provider/model, later calls skip the
        native attempt entirely instead of paying for it every time.
        """
        messages = self._render(prompt_name, values)
        key = (self.provider.value, self.model)
        if key in _native_structured_unsupported:
            return await self._structured_fallback(messages, schema, temperature)
        if key in _native_structured_supported:
            try:
                return await self._structured_native(messages, schema, temperature)
            except Exception:  # noqa: BLE001 - preserve the resilient fallback
                return await self._structured_fallback(messages, schema, temperature)

        # Several documents are commonly indexed together. Without a
        # single-flight guard they all probe native structured output at the
        # same time, so an unsupported model pays for N doomed requests before
        # the first failure can populate the cache. Only the initial probe is
        # serialised; once support is known, normal calls remain concurrent.
        lock = _native_structured_probe_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in _native_structured_unsupported:
                return await self._structured_fallback(messages, schema, temperature)
            if key in _native_structured_supported:
                try:
                    return await self._structured_native(messages, schema, temperature)
                except Exception:  # noqa: BLE001 - preserve the resilient fallback
                    return await self._structured_fallback(messages, schema, temperature)
            try:
                result = await self._structured_native(messages, schema, temperature)
                _native_structured_supported.add(key)
                return result
            except Exception as exc:  # noqa: BLE001 - fallback path is intentional
                _native_structured_unsupported.add(key)
                log_event(
                    logger,
                    "llm.structured",
                    "Native structured output failed, retrying with JSON instructions "
                    "(will use the JSON fallback directly for this model from now on)",
                    level=30,
                    prompt=prompt_name,
                    provider=self.provider.value,
                    error=str(exc),
                )
                return await self._structured_fallback(messages, schema, temperature)

    async def complete(
        self,
        prompt_name: str,
        temperature: float | None = None,
        **values: Any,
    ) -> str:
        """Run a prompt and return the reply as plain text."""
        messages = self._render(prompt_name, values)
        model = self._chat_model(streaming=False, temperature=temperature)
        response = await self._invoke(model, messages)
        return _as_text(response.content)

    async def stream(
        self,
        prompt_name: str,
        temperature: float | None = None,
        **values: Any,
    ) -> AsyncIterator[str]:
        """Run a prompt and yield the reply progressively."""
        messages = self._render(prompt_name, values)
        model = self._chat_model(streaming=True, temperature=temperature)

        async for text in self._stream_with_retry(model, messages):
            yield text

    async def stream_messages(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        """Stream a reply for an already-rendered message list."""
        model = self._chat_model(streaming=True, temperature=None)
        async for text in self._stream_with_retry(model, messages):
            yield text

    # -- internals -------------------------------------------------------

    def _render(self, prompt_name: str, values: dict[str, Any]) -> list[BaseMessage]:
        return self._prompts.chat_prompt(prompt_name).format_messages(**values)

    def _chat_model(self, streaming: bool, temperature: float | None):
        return get_llm(
            provider=self.provider,
            model_name=self.model,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            streaming=streaming,
        )

    def _config(self) -> dict[str, Any]:
        return {"callbacks": self.usage.callbacks}

    async def _invoke(self, runnable: Any, payload: Any) -> Any:
        """Invoke a runnable with bounded retries on transient failures."""
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return await runnable.ainvoke(payload, config=self._config())
            except Exception as exc:  # noqa: BLE001 - inspected below
                last_error = exc
                if attempt == _MAX_ATTEMPTS or not _is_retryable(exc):
                    break
                await asyncio.sleep(_retry_delay(exc, attempt))

        assert last_error is not None
        raise translate_error(self.provider, last_error) from last_error

    async def _stream_with_retry(
        self, model: Any, messages: list[BaseMessage]
    ) -> AsyncIterator[str]:
        """Retry transient failures only while no partial answer was emitted."""
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            collected: list[str] = []
            emitted = False
            try:
                async for chunk in model.astream(messages, config=self._config()):
                    text = _as_text(chunk.content)
                    if text:
                        emitted = True
                        collected.append(text)
                        yield text
                self._record_manual(messages, "".join(collected))
                return
            except Exception as exc:  # noqa: BLE001 - normalised after retries
                last_error = exc
                if emitted or attempt == _MAX_ATTEMPTS or not _is_retryable(exc):
                    break
                await asyncio.sleep(_retry_delay(exc, attempt))

        assert last_error is not None
        raise translate_error(self.provider, last_error) from last_error

    async def _structured_native(
        self, messages: list[BaseMessage], schema: type[T], temperature: float | None
    ) -> T:
        model = self._chat_model(streaming=False, temperature=temperature)
        structured = model.with_structured_output(schema)
        result = await self._invoke(structured, messages)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
        raise TypeError(f"Unexpected structured output type: {type(result)!r}")

    async def _structured_fallback(
        self, messages: list[BaseMessage], schema: type[T], temperature: float | None
    ) -> T:
        parser = PydanticOutputParser(pydantic_object=schema)
        instructed = [
            *messages,
            HumanMessage(
                content=(
                    "Reply with a single JSON object and nothing else. "
                    "No prose, no explanation, no code fence.\n\n"
                    + parser.get_format_instructions()
                )
            ),
        ]
        model = self._chat_model(streaming=False, temperature=temperature)
        response = await self._invoke(model, instructed)
        raw = _as_text(response.content)
        self._record_manual(instructed, raw)
        return _parse_json_into(raw, schema)

    def _record_manual(self, messages: Iterable[BaseMessage], output: str) -> None:
        """Record an approximate usage entry when the backend reports none."""
        prompt_text = "\n".join(_as_text(m.content) for m in messages)
        self.usage.record_manual(
            input_tokens=approximate_tokens(prompt_text),
            output_tokens=approximate_tokens(output),
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _as_text(content: Any) -> str:
    """Flatten LangChain message content into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def _is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Honor a provider reset hint, otherwise use the bounded default backoff."""
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        delay = float(match.group(1))
        if (match.group(2) or "").lower() == "ms":
            delay /= 1_000
        return min(_MAX_RETRY_DELAY, max(0.0, delay + 0.25))
    return min(_MAX_RETRY_DELAY, _RETRY_BASE_DELAY * attempt)


def _parse_json_into(raw: str, schema: type[T]) -> T:
    """Extract the first JSON object from ``raw`` and validate it."""
    candidate = raw.strip()

    fenced = _JSON_BLOCK_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        candidate = candidate[start : end + 1]

    return schema.model_validate(json.loads(candidate))


async def gather_limited(coroutines: list[Any], limit: int) -> list[Any]:
    """Run coroutines with bounded concurrency, returning results or exceptions."""
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _run(coro: Any) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coroutines), return_exceptions=True)
