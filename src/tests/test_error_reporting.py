"""A failed turn must say why, and the reason must reach the logs."""

from __future__ import annotations

from types import SimpleNamespace

from config.settings import get_settings, save_settings
from core.constants import LLMProviderType
from core.exceptions import LLMInvalidCredentialsError, LLMRateLimitError
from core.models import AppSettings, QueryPlan, TokenUsage
from graph.checkpoints import _checkpointed_types, _serializer
from graph.nodes import WorkflowNodes
from llm.factory import translate_error


class _Context(SimpleNamespace):
    def emit_stage(self, label: str) -> None:
        self.last_stage = label

    def take_usage(self) -> TokenUsage:
        return TokenUsage()


def _nodes() -> WorkflowNodes:
    return WorkflowNodes(_Context(client=None, settings=AppSettings()))  # type: ignore[arg-type]


async def test_the_recorded_reason_reaches_the_user() -> None:
    failure = LLMInvalidCredentialsError("Groq")

    result = await _nodes().handle_error({"chat_id": "c1", "error": failure.user_message})

    assert failure.user_message in result["answer"]
    assert "Settings" in result["answer"]


async def test_a_rate_limit_keeps_its_dedicated_wording() -> None:
    result = await _nodes().handle_error(
        {"chat_id": "c1", "error": LLMRateLimitError("Groq").user_message}
    )

    assert "rate limit" in result["answer"].lower()


async def test_an_unrecorded_failure_still_produces_a_reply() -> None:
    result = await _nodes().handle_error({"chat_id": "c1"})

    assert result["answer"].strip()


def test_a_provider_message_never_carries_the_api_key() -> None:
    """Gemini puts the key in the URL, so raw SDK text can contain it."""
    settings = get_settings()
    original = settings.provider(LLMProviderType.GOOGLE).api_key
    settings.provider(LLMProviderType.GOOGLE).api_key = "super-secret-key"
    save_settings(settings)
    try:
        error = translate_error(
            LLMProviderType.GOOGLE,
            RuntimeError("failed for url https://x/models?key=super-secret-key"),
        )
        assert "super-secret-key" not in error.user_message
    finally:
        settings.provider(LLMProviderType.GOOGLE).api_key = original
        save_settings(settings)


def test_an_incomplete_provider_is_reported_before_a_question_is_asked() -> None:
    """A custom endpoint saved without its URL used to fail only at answer time."""
    settings = AppSettings()
    settings.active_provider = LLMProviderType.CUSTOM.value
    custom = settings.provider(LLMProviderType.CUSTOM)
    custom.api_key = "key"
    custom.selected_model = "some-model"

    assert custom.missing_requirement() == "server URL"

    custom.base_url = "https://gateway.example.com/v1"
    assert custom.missing_requirement() == ""

    custom.selected_model = ""
    assert custom.missing_requirement() == "model"


def test_every_state_type_is_registered_for_checkpointing() -> None:
    registered = {(cls.__module__, cls.__name__) for cls in _checkpointed_types()}

    for cls in (QueryPlan, TokenUsage):
        assert (cls.__module__, cls.__name__) in registered


def test_a_checkpointed_value_survives_a_round_trip() -> None:
    serde = _serializer()
    plan = QueryPlan(original_query="hello", intent="greeting", search_queries=["hello"])

    restored = serde.loads_typed(serde.dumps_typed({"query_plan": plan}))

    assert restored["query_plan"] == plan
