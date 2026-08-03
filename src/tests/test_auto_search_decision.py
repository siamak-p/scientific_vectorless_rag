"""Automatic search is gap-driven and the configured paper count is a cap."""

from __future__ import annotations

from types import SimpleNamespace

from core.models import (
    AppSettings,
    DocumentRecord,
    PaperRankingEntry,
    QueryPlan,
    SearchAssessment,
    TokenUsage,
)
from graph.deep_research import _GapAnalysis, _deep_search_limit
from graph.graph_builder import _route_after_evidence, _route_after_search_assessment
from graph.nodes import (
    WorkflowNodes,
    _no_evidence_message,
    _select_documents_to_read,
    _with_rate_limit_warning,
)


class _DecisionClient:
    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self.payload = payload or {}
        self.error = error
        self.calls = 0

    async def structured(self, prompt_name, schema, temperature=0.0, **values):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return schema.model_validate(self.payload)


class _Context(SimpleNamespace):
    def emit_stage(self, label: str) -> None:
        self.last_stage = label

    def take_usage(self) -> TokenUsage:
        return TokenUsage()


def _nodes(client: _DecisionClient, cap: int = 3) -> WorkflowNodes:
    settings = AppSettings()
    settings.retrieval.max_search_papers = cap
    return WorkflowNodes(_Context(client=client, settings=settings))  # type: ignore[arg-type]


def _state(
    documents: list[DocumentRecord], enabled: bool = True
) -> dict:
    return {
        "chat_id": "chat-1",
        "query": "How does attention improve translation?",
        "query_plan": QueryPlan(
            original_query="How does attention improve translation?",
            intent="Explain attention in machine translation",
            search_queries=["attention machine translation"],
        ),
        "documents": documents,
        "auto_search_enabled": enabled,
    }


async def test_disabled_search_never_calls_the_assessment_model(document) -> None:
    client = _DecisionClient({"search_needed": True, "suggested_papers": 3})

    result = await _nodes(client).assess_search_need(_state([document], enabled=False))

    assert client.calls == 0
    assert result["search_assessment"].search_needed is False


async def test_existing_corpus_can_skip_search(document) -> None:
    client = _DecisionClient(
        {
            "search_needed": False,
            "suggested_papers": 0,
            "reasoning": "The indexed paper directly covers the question.",
        }
    )

    result = await _nodes(client).assess_search_need(_state([document]))

    assessment = result["search_assessment"]
    assert assessment.search_needed is False
    assert assessment.suggested_papers == 0
    assert _route_after_search_assessment(
        {"auto_search_enabled": True, "search_assessment": assessment}
    ) == "skip"


async def test_model_suggestion_is_capped_by_download_setting(document) -> None:
    client = _DecisionClient({"search_needed": True, "suggested_papers": 12})

    result = await _nodes(client, cap=3).assess_search_need(_state([document]))

    assessment = result["search_assessment"]
    assert assessment.search_needed is True
    assert assessment.suggested_papers == 3


async def test_empty_corpus_requires_only_smallest_search_when_model_declines() -> None:
    client = _DecisionClient({"search_needed": False, "suggested_papers": 0})

    result = await _nodes(client, cap=5).assess_search_need(_state([]))

    assessment = result["search_assessment"]
    assert assessment.search_needed is True
    assert assessment.suggested_papers == 1


async def test_assessment_failure_does_not_trigger_network_with_existing_papers(
    document,
) -> None:
    client = _DecisionClient(error=RuntimeError("provider unavailable"))

    result = await _nodes(client).assess_search_need(_state([document]))

    assessment = result["search_assessment"]
    assert assessment.search_needed is False
    assert assessment.suggested_papers == 0
    assert "warnings" not in result


async def test_assessment_rate_limit_is_reported_while_fallback_continues(
    document,
) -> None:
    client = _DecisionClient(error=RuntimeError("429 rate limit exceeded"))

    result = await _nodes(client).assess_search_need({
        **_state([document]),
        "provider": "groq",
        "warnings": [],
    })

    assert result["search_assessment"].search_needed is False
    assert "Groq provider reached its rate limit" in result["warnings"][0]


async def test_auto_search_uses_suggested_count_not_configured_max(
    document, monkeypatch
) -> None:
    nodes = _nodes(_DecisionClient(), cap=5)
    captured: dict[str, int] = {}

    async def fake_search_and_index(chat_id, queries, limit, existing):
        captured["limit"] = limit
        return [], []

    monkeypatch.setattr(nodes, "search_and_index", fake_search_and_index)
    state = {
        **_state([document]),
        "search_assessment": SearchAssessment(
            search_needed=True, suggested_papers=2, reasoning="A comparison is missing."
        ),
    }

    result = await nodes.auto_search(state)

    assert captured["limit"] == 2
    assert result["search_attempted"] is True


async def test_auto_search_does_not_touch_network_when_corpus_is_sufficient(
    document, monkeypatch
) -> None:
    nodes = _nodes(_DecisionClient(), cap=5)
    called = False

    async def unexpected_search(*args, **kwargs):
        nonlocal called
        called = True
        return [], []

    monkeypatch.setattr(nodes, "search_and_index", unexpected_search)
    state = {
        **_state([document]),
        "search_assessment": SearchAssessment(
            search_needed=False,
            suggested_papers=0,
            reasoning="The current corpus is sufficient.",
        ),
    }

    await nodes.auto_search(state)

    assert called is False


def test_route_skips_search_when_permission_is_disabled() -> None:
    state = {
        "auto_search_enabled": False,
        "search_assessment": SearchAssessment(search_needed=True, suggested_papers=2),
    }

    assert _route_after_search_assessment(state) == "skip"


def test_zero_evidence_recovers_with_search_when_permission_is_enabled() -> None:
    state = {
        "auto_search_enabled": True,
        "search_attempted": False,
        "evidence": [],
    }

    assert _route_after_evidence(state) == "recover_search"


def test_zero_evidence_does_not_loop_after_search_was_attempted() -> None:
    state = {
        "auto_search_enabled": True,
        "search_attempted": True,
        "evidence": [],
    }

    assert _route_after_evidence(state) == "answer"


async def test_zero_evidence_forces_search_and_keeps_model_paper_count(document) -> None:
    client = _DecisionClient(
        {
            "search_needed": False,
            "suggested_papers": 2,
            "reasoning": "Two new sources should cover the evidence gap.",
        }
    )

    result = await _nodes(client, cap=5).prepare_evidence_gap_search(_state([document]))

    assessment = result["search_assessment"]
    assert assessment.search_needed is True
    assert assessment.suggested_papers == 2


def test_no_evidence_message_does_not_suggest_enabling_search_after_attempt() -> None:
    message = _no_evidence_message(
        {"search_attempted": True, "documents": [object()]}
    )

    assert "after automatic scientific search" in message
    assert "enabling automatic" not in message


def test_rate_limit_warning_names_provider_and_is_deduplicated() -> None:
    state = {"provider": "groq", "warnings": []}

    warnings = _with_rate_limit_warning(state)
    warnings = _with_rate_limit_warning({**state, "warnings": warnings})

    assert len(warnings) == 1
    assert "Groq provider reached its rate limit" in warnings[0]


async def test_terminal_rate_limit_is_explicit_but_other_internals_are_hidden() -> None:
    nodes = _nodes(_DecisionClient())

    limited = await nodes.handle_error({"error": "Rate limit exceeded for Groq"})
    internal = await nodes.handle_error({"error": "secret parser stack trace"})

    assert "rate limit" in limited["answer"]
    assert "secret parser stack trace" not in internal["answer"]


def test_newly_discovered_papers_are_guaranteed_a_reading_slot() -> None:
    rankings = [
        PaperRankingEntry(document_id="old-a", overall_score=0.99),
        PaperRankingEntry(document_id="old-b", overall_score=0.95),
        PaperRankingEntry(document_id="new-a", overall_score=0.90),
    ]

    selected = _select_documents_to_read(rankings, keep=2, preferred_ids={"new-a"})

    assert selected == ["new-a", "old-a"]


def test_deep_research_uses_model_count_below_configured_cap() -> None:
    analysis = _GapAnalysis(
        is_sufficient=False,
        next_search_queries=["missing comparison"],
        suggested_papers=2,
    )

    assert _deep_search_limit(analysis, configured_cap=5) == 2
    assert _deep_search_limit(
        analysis.model_copy(update={"suggested_papers": 12}), configured_cap=3
    ) == 3
