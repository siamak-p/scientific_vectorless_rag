"""Nothing is indexed or answered from a source that does not fit the question."""

from __future__ import annotations

from types import SimpleNamespace

from core.constants import MIN_SEARCH_RELEVANCE, SearchProviderType
from core.models import AppSettings, DocumentMetadata, EvidenceItem, TokenUsage
from graph.nodes import WorkflowNodes, _eligible_candidates, _ungrounded_message
from scientific_search.base import SearchResult


class _Context(SimpleNamespace):
    def emit_stage(self, label: str) -> None:
        self.last_stage = label

    def take_usage(self) -> TokenUsage:
        return TokenUsage()


def _nodes(strict: bool = False) -> WorkflowNodes:
    settings = AppSettings()
    settings.research.strict_grounding = strict
    return WorkflowNodes(_Context(client=None, settings=settings))  # type: ignore[arg-type]


def _result(title: str, relevance: float) -> SearchResult:
    return SearchResult(
        metadata=DocumentMetadata(title=title),
        provider=SearchProviderType.ARXIV,
        pdf_url="https://example.org/paper.pdf",
        is_open_access=True,
        query_relevance=relevance,
    )


def test_off_topic_results_are_not_downloaded() -> None:
    """A database answers with its best matches even when none is on topic."""
    results = [
        _result("Exploring medical students' use of structured reflection", 0.08),
        _result("A survey of clerkship teaching methods", 0.05),
    ]

    candidates, rejected = _eligible_candidates(results, 3, set(), download_pdfs=True)

    assert candidates == []
    assert rejected == 2


def test_a_matching_result_is_still_downloaded() -> None:
    results = [
        _result("Postoperative atelectasis: mechanisms and management", 0.62),
        _result("Unrelated paper", 0.02),
    ]

    candidates, rejected = _eligible_candidates(results, 3, set(), download_pdfs=True)

    assert [item.metadata.title for item in candidates] == [
        "Postoperative atelectasis: mechanisms and management"
    ]
    assert rejected == 1


def test_papers_already_in_the_chat_are_not_downloaded_again() -> None:
    results = [_result("Atelectasis in the ICU", 0.7)]

    candidates, _ = _eligible_candidates(
        results, 3, {"atelectasis in the icu"}, download_pdfs=True
    )

    assert candidates == []


async def test_an_answer_that_cites_nothing_is_replaced() -> None:
    """General knowledge must never be presented as a sourced answer."""
    evidence = [
        EvidenceItem(
            id="ev-1",
            document_id="doc-a",
            document_title="Structured reflection during clerkship",
            page_number=3,
            content="Students used structured reflection during clerkship.",
            confidence_score=0.8,
        )
    ]
    state = {
        "chat_id": "chat-1",
        "query": "What is atelectasis?",
        "answer": "Atelectasis is a collapse of lung tissue.",
        "evidence": evidence,
        "documents": [],
        "search_attempted": True,
    }

    result = await _nodes().validate_answer(state)

    assert "Atelectasis is a collapse" not in result["answer"]
    assert result["citations"] == []
    assert "do not answer this question" in result["answer"]


async def test_a_cited_answer_is_kept() -> None:
    evidence = [
        EvidenceItem(
            id="ev-1",
            document_id="doc-a",
            document_title="Atelectasis in the ICU",
            page_number=3,
            content="Atelectasis is alveolar collapse.",
            confidence_score=0.9,
        )
    ]
    state = {
        "chat_id": "chat-1",
        "query": "What is atelectasis?",
        "answer": "Atelectasis is alveolar collapse [E1].",
        "evidence": evidence,
        "documents": [],
    }

    result = await _nodes().validate_answer(state)

    assert "Atelectasis is alveolar collapse" in result["answer"]
    assert len(result["citations"]) == 1


def test_the_replacement_names_the_papers_that_were_read() -> None:
    message = _ungrounded_message(
        {
            "evidence": [
                EvidenceItem(
                    id="ev-1",
                    document_id="doc-a",
                    document_title="Structured reflection during clerkship",
                    page_number=1,
                    content="...",
                )
            ],
            "search_attempted": True,
        }
    )

    assert "Structured reflection during clerkship" in message
    # The assistant translates queries itself; it must never push the user
    # to switch languages.
    assert "English" not in message
    assert "standard scientific term" in message


# ---------------------------------------------------------------------------
# Title/abstract screening before download
# ---------------------------------------------------------------------------

class _ScreeningClient:
    def __init__(self, verdicts: list[dict] | None = None, error: Exception | None = None):
        self.verdicts = verdicts
        self.error = error
        self.calls = 0
        self.kwargs: dict = {}

    async def structured(self, prompt_name, schema, temperature=0.0, **values):
        self.calls += 1
        self.kwargs = values
        if self.error is not None:
            raise self.error
        return schema.model_validate({"verdicts": self.verdicts or []})


def _screening_nodes(client: _ScreeningClient) -> WorkflowNodes:
    return WorkflowNodes(_Context(client=client, settings=AppSettings()))  # type: ignore[arg-type]


def test_a_generic_shared_word_passes_the_lexical_floor() -> None:
    """Why screening exists: 'definition' alone lifts an astronomy report over the floor."""
    from scientific_search.aggregator import _query_relevance

    astronomy = SearchResult(
        metadata=DocumentMetadata(title="Roman Galactic Plane Survey Definition Committee Report"),
        provider=SearchProviderType.ARXIV,
    )

    assert _query_relevance(astronomy, "intestinal atelectasis definition", 1) >= MIN_SEARCH_RELEVANCE


async def test_screening_rejects_off_topic_results_before_download() -> None:
    pool = [
        _result("Roman Galactic Plane Survey Definition Committee Report", 0.32),
        _result("Postoperative atelectasis: mechanisms and management", 0.6),
    ]
    client = _ScreeningClient([
        {"index": 0, "relevant": False, "reason": "Astronomy survey."},
        {"index": 1, "relevant": True, "reason": "Directly about atelectasis."},
    ])

    accepted, rejected, rate_limited = await _screening_nodes(client)._screen_candidates(
        "What is atelectasis?", "Define atelectasis", pool, limit=3
    )

    assert [item.metadata.title for item in accepted] == [
        "Postoperative atelectasis: mechanisms and management"
    ]
    assert rejected == 1
    assert rate_limited is False
    assert "CANDIDATE 0" in client.kwargs["candidates"]
    assert client.kwargs["intent"] == "Define atelectasis"


async def test_an_unjudged_candidate_is_not_downloaded() -> None:
    pool = [_result("Paper A", 0.5), _result("Paper B", 0.5)]
    client = _ScreeningClient([{"index": 1, "relevant": True}])

    accepted, rejected, _ = await _screening_nodes(client)._screen_candidates(
        "question", "", pool, limit=3
    )

    assert [item.metadata.title for item in accepted] == ["Paper B"]
    assert rejected == 1


async def test_screening_failure_keeps_the_lexical_order() -> None:
    pool = [_result("Paper A", 0.5), _result("Paper B", 0.4), _result("Paper C", 0.3)]
    client = _ScreeningClient(error=RuntimeError("429 rate limit"))

    accepted, rejected, rate_limited = await _screening_nodes(client)._screen_candidates(
        "question", "", pool, limit=2
    )

    assert [item.metadata.title for item in accepted] == ["Paper A", "Paper B"]
    assert rejected == 0
    assert rate_limited is True


async def test_screening_respects_the_paper_limit() -> None:
    pool = [_result(f"Paper {index}", 0.5) for index in range(4)]
    client = _ScreeningClient([{"index": index, "relevant": True} for index in range(4)])

    accepted, _, _ = await _screening_nodes(client)._screen_candidates(
        "question", "", pool, limit=2
    )

    assert len(accepted) == 2
