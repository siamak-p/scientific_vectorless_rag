"""Provider configuration, prompt library and search result merging."""

from __future__ import annotations

import pytest

from config.catalog import fallback_models, load_catalog, search_endpoint
from core.constants import LLMProviderType, SearchProviderType
from core.exceptions import PromptNotFoundError
from core.models import AppSettings, DocumentMetadata
from prompts.manager import get_prompt_manager
from scientific_search.aggregator import (
    _query_relevance,
    available_providers,
    build_provider,
    merge_results,
)
from scientific_search.base import SearchResult

_REQUIRED_PROMPTS = [
    "system/research_assistant",
    "system/conversational",
    "retrieval/query_understanding",
    "retrieval/tree_navigation",
    "retrieval/paper_relevance",
    "reasoning/evidence_extraction",
    "reasoning/gap_analysis",
    "answer/short_answer",
    "answer/research_report",
    "answer/deep_research_report",
    "citation/claim_validation",
    "documents/metadata_extraction",
    "documents/structure_extraction",
    "memory/chat_summary",
]


@pytest.mark.parametrize("name", _REQUIRED_PROMPTS)
def test_every_prompt_is_available_and_parses(name: str) -> None:
    sections = get_prompt_manager().sections(name)

    assert sections
    assert all(role in {"system", "human", "assistant"} for role, _ in sections)
    assert all(body.strip() for _, body in sections)


def test_unknown_prompts_fail_loudly() -> None:
    with pytest.raises(PromptNotFoundError):
        get_prompt_manager().sections("answer/does_not_exist")


def test_prompt_rendering_substitutes_placeholders() -> None:
    rendered = get_prompt_manager().render(
        "answer/short_answer",
        role="human",
        question="What is attention?",
    )

    assert "What is attention?" in rendered
    assert "{question}" not in rendered


@pytest.mark.parametrize("provider", list(LLMProviderType))
def test_every_provider_has_a_catalogue_entry(provider: LLMProviderType) -> None:
    entry = load_catalog()["providers"][provider.value]

    assert entry.get("models_endpoint")
    # Local runtimes expose whatever the user has pulled, so they ship no static list.
    is_local = provider in {LLMProviderType.OLLAMA, LLMProviderType.LMSTUDIO}
    assert bool(fallback_models(provider)) is not is_local


@pytest.mark.parametrize("provider", list(SearchProviderType))
def test_every_search_provider_has_an_endpoint(provider: SearchProviderType) -> None:
    assert search_endpoint(provider.value).startswith("http")


def test_providers_requiring_a_key_are_skipped_when_unconfigured() -> None:
    settings = AppSettings()
    settings.retrieval.search_providers = [SearchProviderType.ARXIV, SearchProviderType.TAVILY]

    assert SearchProviderType.TAVILY not in available_providers(settings)
    assert SearchProviderType.ARXIV in available_providers(settings)


def test_tavily_becomes_available_once_a_key_is_set() -> None:
    settings = AppSettings(tavily_api_key="test-key")
    settings.retrieval.search_providers = [SearchProviderType.TAVILY]

    assert build_provider(SearchProviderType.TAVILY, settings).is_available()


def _result(provider: SearchProviderType, **fields) -> SearchResult:
    metadata = DocumentMetadata(**fields.pop("metadata", {}))
    return SearchResult(metadata=metadata, provider=provider, **fields)


def test_the_same_doi_from_two_providers_is_merged() -> None:
    first = _result(
        SearchProviderType.CROSSREF,
        metadata={"title": "A Paper", "doi": "10.1000/abc", "publisher": "ACM"},
    )
    second = _result(
        SearchProviderType.SEMANTIC_SCHOLAR,
        metadata={"title": "A Paper", "doi": "10.1000/ABC", "citation_count": 42},
        pdf_url="https://example.org/paper.pdf",
        is_open_access=True,
    )

    merged = merge_results([first, second])

    assert len(merged) == 1
    assert merged[0].metadata.citation_count == 42
    assert merged[0].metadata.publisher == "ACM"
    assert merged[0].is_open_access


def test_papers_without_a_doi_are_merged_by_title() -> None:
    first = _result(SearchProviderType.ARXIV, metadata={"title": "Deep Learning!"})
    second = _result(SearchProviderType.OPENALEX, metadata={"title": "deep learning"})

    assert len(merge_results([first, second])) == 1


def test_distinct_papers_are_kept_apart() -> None:
    first = _result(SearchProviderType.ARXIV, metadata={"title": "Paper One"})
    second = _result(SearchProviderType.ARXIV, metadata={"title": "Paper Two"})

    assert len(merge_results([first, second])) == 2


def test_query_relevance_beats_irrelevant_citation_popularity() -> None:
    popular_but_unrelated = _result(
        SearchProviderType.SEMANTIC_SCHOLAR,
        metadata={"title": "Official statistics data sources", "citation_count": 50_000},
        is_open_access=True,
        query_relevance=0.08,
        best_provider_rank=5,
    )
    directly_relevant = _result(
        SearchProviderType.ARXIV,
        metadata={"title": "Machine Learning and Artificial Intelligence Explained"},
        is_open_access=True,
        query_relevance=0.82,
        best_provider_rank=1,
    )

    merged = merge_results([popular_but_unrelated, directly_relevant])

    assert merged[0].metadata.title == directly_relevant.metadata.title


def test_query_relevance_is_derived_from_title_and_abstract() -> None:
    relevant = _result(
        SearchProviderType.ARXIV,
        metadata={
            "title": "Machine Learning and Artificial Intelligence",
            "abstract": "An introduction to learning patterns from data.",
        },
    )
    unrelated = _result(
        SearchProviderType.ARXIV,
        metadata={
            "title": "Changing official statistics data sources",
            "abstract": "A study of national survey collection.",
        },
    )
    query = "machine learning artificial intelligence introduction"

    assert _query_relevance(relevant, query, 2) > _query_relevance(unrelated, query, 1)
