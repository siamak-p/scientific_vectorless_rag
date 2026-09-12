"""Workflow state.

A single typed dictionary flows through the graph. Every node reads what it
needs and returns only the keys it changes, which is what lets LangGraph
checkpoint and resume a run at any point.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.types import Overwrite

from core.constants import AnswerMode
from core.models import (
    Citation,
    DocumentRecord,
    EvidenceItem,
    NavigationDecision,
    PaperRankingEntry,
    QueryPlan,
    ResearchTrace,
    SearchAssessment,
    TokenUsage,
    ValidationReport,
)


def merge_usage(current: TokenUsage | None, update: TokenUsage | None) -> TokenUsage:
    """Reducer accumulating token usage across every node."""
    if current is None:
        return update or TokenUsage()
    if update is None:
        return current
    return current.add(update)


class GraphState(TypedDict, total=False):
    """The state carried through the retrieval and answering workflow."""

    # Request
    chat_id: str
    query: str
    mode: AnswerMode
    provider: str
    model: str
    explainability_enabled: bool
    auto_search_enabled: bool
    deep_research_enabled: bool
    search_attempted: bool

    # Memory
    history: str
    global_topics: str

    # Planning
    query_plan: QueryPlan
    is_conversational: bool
    needs_clarification: bool
    search_assessment: SearchAssessment

    # Corpus
    documents: list[DocumentRecord]
    selected_document_ids: list[str]
    paper_rankings: list[PaperRankingEntry]
    discovered_document_ids: list[str]

    # Retrieval
    navigation_trace: list[NavigationDecision]
    pages_by_document: dict[str, list[int]]
    evidence: list[EvidenceItem]

    # Answering
    answer: str
    # True when `answer` is a code-generated notice (no evidence, ungrounded,
    # error) rather than model output, so it can be rendered in the user's language.
    answer_is_notice: bool
    citations: list[Citation]
    validation: ValidationReport
    research_trace: ResearchTrace
    formatted_answer: dict[str, Any]

    # Bookkeeping
    token_usage: Annotated[TokenUsage, merge_usage]
    warnings: list[str]
    error: str


def initial_state(
    chat_id: str,
    query: str,
    mode: AnswerMode,
    provider: str,
    model: str,
    explainability_enabled: bool = True,
    auto_search_enabled: bool = False,
    deep_research_enabled: bool = False,
) -> GraphState:
    """Build the starting state for one user turn."""
    return GraphState(
        chat_id=chat_id,
        query=query,
        mode=mode,
        provider=provider,
        model=model,
        explainability_enabled=explainability_enabled,
        auto_search_enabled=auto_search_enabled,
        deep_research_enabled=deep_research_enabled,
        search_attempted=False,
        needs_clarification=False,
        answer_is_notice=False,
        documents=[],
        selected_document_ids=[],
        paper_rankings=[],
        discovered_document_ids=[],
        search_assessment=SearchAssessment(),
        navigation_trace=[],
        pages_by_document={},
        evidence=[],
        citations=[],
        warnings=[],
        # The checkpointer keeps one thread per chat (not per turn), so a plain
        # value here would be combined with the previous turn's total through
        # the `merge_usage` reducer instead of starting this turn at zero.
        # `Overwrite` bypasses the reducer for this one write.
        token_usage=Overwrite(TokenUsage()),
        error="",
    )


def documents_by_id(state: GraphState) -> dict[str, DocumentRecord]:
    """Index the state's documents for citation lookup."""
    return {document.id: document for document in state.get("documents", [])}
