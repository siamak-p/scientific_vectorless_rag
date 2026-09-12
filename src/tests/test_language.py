"""The assistant replies in the user's language and asks when it cannot understand."""

from __future__ import annotations

from types import SimpleNamespace

from core.constants import AnswerMode
from core.models import AppSettings, EvidenceItem, QueryPlan, TokenUsage
from graph.graph_builder import _route_after_planning, to_answer
from graph.nodes import WorkflowNodes, _response_language


class _Context(SimpleNamespace):
    def emit_stage(self, label: str) -> None:
        self.last_stage = label

    def emit_token(self, text: str) -> None:
        pass

    def take_usage(self) -> TokenUsage:
        return TokenUsage()


class _Client:
    """Records prompt names and values; answers structured calls from a payload."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self.payload = payload or {}
        self.error = error
        self.structured_calls: list[tuple[str, dict]] = []
        self.stream_calls: list[tuple[str, dict]] = []

    async def structured(self, prompt_name, schema, temperature=0.0, **values):
        self.structured_calls.append((prompt_name, values))
        if self.error is not None:
            raise self.error
        return schema.model_validate(self.payload)

    async def stream(self, prompt_name, temperature=None, **values):
        self.stream_calls.append((prompt_name, values))
        yield "answer text"


def _nodes(client: _Client) -> WorkflowNodes:
    return WorkflowNodes(_Context(client=client, settings=AppSettings()))  # type: ignore[arg-type]


def _persian_plan(**overrides) -> QueryPlan:
    return QueryPlan(
        original_query="آتلکتازی چیست؟",
        intent="Define atelectasis",
        search_queries=["atelectasis definition"],
        language="Persian",
        **overrides,
    )


# ---------------------------------------------------------------------------
# Planning: language, correction, clarification
# ---------------------------------------------------------------------------

async def test_planner_records_language_correction_and_clarification() -> None:
    client = _Client({
        "language": "Persian",
        "intent": "Define intestinal atelectasis",
        "search_queries": ["intestinal atelectasis"],
        "needs_clarification": True,
        "clarification_question": "منظورتان آتلکتازی ریه است یا آترزی روده؟",
        "correction_note": "«آشلکتازی» را «آتلکتازی» (atelectasis) در نظر گرفتم.",
    })

    result = await _nodes(client).query_understanding(
        {"chat_id": "chat-1", "query": "آشلکتازی روده چیه؟", "history": ""}
    )

    plan = result["query_plan"]
    assert plan.language == "Persian"
    assert plan.needs_clarification is True
    assert result["needs_clarification"] is True
    assert "آتلکتازی ریه" in plan.clarification_question
    assert "atelectasis" in plan.correction_note
    assert _route_after_planning(result) == "clarify"


async def test_conversational_messages_are_never_routed_to_clarification() -> None:
    client = _Client({
        "language": "English",
        "intent": "Greet",
        "is_conversational": True,
        "needs_clarification": True,
        "clarification_question": "What do you mean?",
    })

    result = await _nodes(client).query_understanding(
        {"chat_id": "chat-1", "query": "hello there, how has your day been so far?", "history": ""}
    )

    assert result["needs_clarification"] is False
    assert _route_after_planning(result) == "conversational"


async def test_clarification_turn_asks_the_planner_question_without_research() -> None:
    client = _Client()
    plan = _persian_plan(
        needs_clarification=True,
        clarification_question="منظورتان آتلکتازی ریه است یا آترزی روده؟",
    )

    result = await _nodes(client).ask_clarification(
        {"chat_id": "chat-1", "query": "آشلکتازی روده چیه؟", "query_plan": plan}
    )

    assert result["answer"] == plan.clarification_question
    assert result["citations"] == []
    assert client.structured_calls == []


def test_planning_failure_still_routes_to_research() -> None:
    assert _route_after_planning({"error": "boom", "needs_clarification": True}) == "error"
    assert _route_after_planning({"needs_clarification": False}) == "research"


# ---------------------------------------------------------------------------
# Generation: the model is told which language to write in
# ---------------------------------------------------------------------------

def test_response_language_falls_back_to_the_message_language() -> None:
    assert _response_language({"query_plan": _persian_plan()}) == "Persian"
    assert "same language" in _response_language({"query_plan": QueryPlan()})
    assert "same language" in _response_language({})


async def test_answer_prompt_receives_the_user_language() -> None:
    client = _Client()
    evidence = [EvidenceItem(
        id="ev-1", document_id="doc-a", document_title="Atelectasis in the ICU",
        page_number=3, content="Atelectasis is alveolar collapse.", confidence_score=0.9,
    )]

    result = await _nodes(client).generate_answer({
        "chat_id": "chat-1",
        "query": "آتلکتازی چیست؟",
        "query_plan": _persian_plan(),
        "evidence": evidence,
        "mode": AnswerMode.SHORT_ANSWER,
    })

    prompt_name, values = client.stream_calls[0]
    assert prompt_name == "answer/short_answer"
    assert values["response_language"] == "Persian"
    assert result["answer"] == "answer text"
    assert "answer_is_notice" not in result


async def test_conversational_prompt_receives_the_user_language() -> None:
    client = _Client()

    await _nodes(client).conversational_answer({
        "chat_id": "chat-1",
        "query": "سلام، تا الان درباره چه چیزهایی حرف زدیم؟",
        "query_plan": _persian_plan(is_conversational=True),
    })

    _, values = client.stream_calls[0]
    assert values["response_language"] == "Persian"


# ---------------------------------------------------------------------------
# Code-generated notices are translated in one call, never lost
# ---------------------------------------------------------------------------

async def test_notice_warnings_and_labels_are_localized_together() -> None:
    client = _Client({"items": ["اعلان", "هشدار", "یادداشت‌ها"]})

    result = await _nodes(client).format_output({
        "chat_id": "chat-1",
        "query_plan": _persian_plan(),
        "answer": "**No documents are available in this chat yet.**",
        "answer_is_notice": True,
        "warnings": ["Semantic Scholar reached its rate limit."],
        "citations": [],
        "evidence": [],
    })

    assert len(client.structured_calls) == 1
    prompt_name, values = client.structured_calls[0]
    assert prompt_name == "system/localize_notice"
    assert values["language"] == "Persian"
    assert result["answer"] == "اعلان"
    assert result["warnings"] == ["هشدار"]
    assert result["formatted_answer"]["labels"]["notes"] == "یادداشت‌ها"

    rendered = to_answer({**result, "citations": [], "evidence": []}).answer
    assert "**یادداشت‌ها**" in rendered
    assert "- هشدار" in rendered


async def test_model_written_answers_are_not_retranslated() -> None:
    client = _Client({"items": ["should not be used"]})

    result = await _nodes(client).format_output({
        "chat_id": "chat-1",
        "query_plan": _persian_plan(),
        "answer": "پاسخ مدل [1, p. 2]",
        "warnings": [],
        "citations": [],
        "evidence": [],
    })

    assert client.structured_calls == []
    assert result["answer"] == "پاسخ مدل [1, p. 2]"


async def test_english_and_unknown_languages_skip_translation() -> None:
    for plan in (QueryPlan(language="English"), QueryPlan(), None):
        client = _Client({"items": ["unexpected"]})
        state = {
            "chat_id": "chat-1",
            "answer": "**Notice**",
            "answer_is_notice": True,
            "warnings": ["A warning."],
            "citations": [],
            "evidence": [],
        }
        if plan is not None:
            state["query_plan"] = plan

        result = await _nodes(client).format_output(state)

        assert client.structured_calls == []
        assert result["answer"] == "**Notice**"
        assert result["warnings"] == ["A warning."]


async def test_translation_failure_or_mismatch_keeps_the_english_notice() -> None:
    state = {
        "chat_id": "chat-1",
        "query_plan": _persian_plan(),
        "answer": "**Notice**",
        "answer_is_notice": True,
        "warnings": ["A warning."],
        "citations": [],
        "evidence": [],
    }

    failed = await _nodes(_Client(error=RuntimeError("provider down"))).format_output(state)
    assert failed["answer"] == "**Notice**"
    assert failed["warnings"] == ["A warning."]

    mismatched = await _nodes(_Client({"items": ["only one"]})).format_output(state)
    assert mismatched["answer"] == "**Notice**"
    assert mismatched["warnings"] == ["A warning."]


async def test_error_notice_is_localized_but_not_on_rate_limit() -> None:
    client = _Client({"items": ["خطا"]})
    nodes = _nodes(client)

    localized = await nodes.handle_error({
        "chat_id": "chat-1", "query_plan": _persian_plan(), "error": "Server URL missing.",
    })
    assert localized["answer"] == "خطا"
    assert localized["answer_is_notice"] is True

    limited = await nodes.handle_error({
        "chat_id": "chat-1", "query_plan": _persian_plan(), "error": "429 rate limit",
    })
    assert "rate limit" in limited["answer"]
    assert len(client.structured_calls) == 1


# ---------------------------------------------------------------------------
# Corrections are shown, not silently applied
# ---------------------------------------------------------------------------

async def test_correction_note_is_prepended_to_the_answer() -> None:
    note = "«آشلکتازی» را «آتلکتازی» (atelectasis) در نظر گرفتم."
    result = await _nodes(_Client({"items": []})).format_output({
        "chat_id": "chat-1",
        "query_plan": _persian_plan(correction_note=note),
        "answer": "پاسخ",
        "warnings": [],
        "citations": [],
        "evidence": [],
    })

    assert result["answer"].startswith(f"> {note}\n\n")
    assert result["answer"].endswith("پاسخ")


async def test_correction_note_is_not_repeated_on_a_clarification_turn() -> None:
    plan = _persian_plan(
        needs_clarification=True,
        clarification_question="منظورتان چیست؟",
        correction_note="یادداشت",
    )
    result = await _nodes(_Client({"items": []})).format_output({
        "chat_id": "chat-1",
        "query_plan": plan,
        "answer": plan.clarification_question,
        "warnings": [],
        "citations": [],
        "evidence": [],
    })

    assert result["answer"] == plan.clarification_question
