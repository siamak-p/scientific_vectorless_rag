"""Cross-provider search aggregation.

Providers are queried in parallel and their results merged: the same paper
found by several sources is fused into one record that keeps the richest
field from each, so a Crossref DOI, a Semantic Scholar citation count and an
arXiv PDF link end up on a single entry.
"""

from __future__ import annotations

import asyncio
import re

from core.constants import SEARCH_CONCURRENCY, SearchProviderType
from core.exceptions import SearchError
from core.models import AppSettings, DocumentMetadata
from observability.logger import get_logger, log_event
from scientific_search.arxiv_provider import ArxivProvider
from scientific_search.base import ScientificSearchProvider, SearchResult
from scientific_search.crossref import CrossrefProvider
from scientific_search.custom_source import CustomSearchProvider
from scientific_search.doaj import DOAJProvider
from scientific_search.europe_pmc import EuropePMCProvider
from scientific_search.openalex import OpenAlexProvider
from scientific_search.pubmed import PubMedProvider
from scientific_search.semantic_scholar import SemanticScholarProvider
from scientific_search.tavily import TavilyProvider

logger = get_logger("scientific_search.aggregator")

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SEARCH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "what", "when", "which", "why", "with",
}

_PROVIDERS: dict[SearchProviderType, type[ScientificSearchProvider]] = {
    SearchProviderType.ARXIV: ArxivProvider,
    SearchProviderType.SEMANTIC_SCHOLAR: SemanticScholarProvider,
    SearchProviderType.OPENALEX: OpenAlexProvider,
    SearchProviderType.CROSSREF: CrossrefProvider,
    SearchProviderType.PUBMED: PubMedProvider,
    SearchProviderType.EUROPE_PMC: EuropePMCProvider,
    SearchProviderType.DOAJ: DOAJProvider,
    SearchProviderType.TAVILY: TavilyProvider,
}


def build_provider(
    provider_type: SearchProviderType, settings: AppSettings
) -> ScientificSearchProvider:
    """Instantiate one built-in search provider (custom sources use ``build_providers``)."""
    return _PROVIDERS[provider_type](settings)


def build_providers(
    provider_types: list[SearchProviderType], settings: AppSettings
) -> list[ScientificSearchProvider]:
    """Instantiate every provider behind the given types.

    ``CUSTOM`` expands to one provider per user-defined source, which is why
    the aggregator works on instances rather than enum members.
    """
    providers: list[ScientificSearchProvider] = []
    for provider_type in provider_types:
        if provider_type is SearchProviderType.CUSTOM:
            providers.extend(
                CustomSearchProvider(settings, source)
                for source in settings.custom_search_sources
            )
        elif provider_type in _PROVIDERS:
            providers.append(build_provider(provider_type, settings))
    return providers


def _is_available(provider_type: SearchProviderType, settings: AppSettings) -> bool:
    return any(p.is_available() for p in build_providers([provider_type], settings))


def available_providers(settings: AppSettings) -> list[SearchProviderType]:
    """Return the configured providers that can actually run."""
    return [
        provider_type
        for provider_type in settings.retrieval.search_providers
        if provider_type.is_builtin_database and _is_available(provider_type, settings)
    ]


def effective_search_providers(settings: AppSettings) -> list[SearchProviderType]:
    """Return the providers one auto-search request should actually query.

    Tavily and the user's custom sources are not "scientific databases" one
    ticks in the list: Tavily joins automatically when a key is configured,
    custom sources whenever at least one is enabled and configured. Neither
    produces a warning when absent - they were never explicitly requested.
    """
    selected = available_providers(settings)
    for bonus in (SearchProviderType.TAVILY, SearchProviderType.CUSTOM):
        if bonus not in selected and _is_available(bonus, settings):
            selected = [*selected, bonus]
    return selected


def covering_providers(
    settings: AppSettings,
    domain: str,
    exclude: list[SearchProviderType] | None = None,
) -> list[SearchProviderType]:
    """Key-free databases that index ``domain`` and are not in ``exclude``.

    Used when the databases the user selected cannot hold the literature of a
    question (or have failed): a clinical question sent only to arXiv can
    never be answered, whatever the query says. The result is restricted to
    providers that need no credentials, so nothing is called on the user's
    behalf that they have not been able to configure.
    """
    excluded = set(exclude or [])
    return [
        provider_type
        for provider_type in _PROVIDERS
        if provider_type not in excluded
        and not provider_type.requires_api_key
        and any(
            p.is_available() and p.covers(domain)
            for p in build_providers([provider_type], settings)
        )
    ]


def uncovered_domain(
    settings: AppSettings,
    providers: list[SearchProviderType],
    domain: str,
    failed: set[str] | None = None,
) -> bool:
    """Whether no working provider in ``providers`` indexes ``domain``.

    ``failed`` holds the labels of providers whose calls errored; a database
    that covers the field but returned nothing usable does not count.
    """
    if not domain:
        return False
    failed_labels = failed or set()
    return not any(
        provider.covers(domain) and provider.label not in failed_labels
        for provider in build_providers(providers, settings)
    )


class SearchAggregator:
    """Fans a query out to every enabled provider and merges the results."""

    def __init__(self, settings: AppSettings, concurrency: int = SEARCH_CONCURRENCY) -> None:
        self._settings = settings
        self._concurrency = max(1, concurrency)
        self.provider_errors: dict[str, str] = {}

    async def search(
        self,
        queries: list[str],
        limit_per_query: int = 5,
        providers: list[SearchProviderType] | None = None,
    ) -> list[SearchResult]:
        """Run every query against every provider and merge the outcome."""
        selected = providers or available_providers(self._settings)
        instances = [p for p in build_providers(selected, self._settings) if p.is_available()]
        unique_queries = _unique([q.strip() for q in queries if q.strip()])

        if not instances or not unique_queries:
            return []

        self.provider_errors = {}
        semaphore = asyncio.Semaphore(self._concurrency)
        # A provider's own tolerance (e.g. one request per second without a
        # key) bounds its requests separately from the global fan-out.
        per_provider = {
            id(provider): asyncio.Semaphore(max(1, provider.max_concurrency()))
            for provider in instances
        }

        async def run(provider: ScientificSearchProvider, query: str) -> list[SearchResult]:
            async with semaphore, per_provider[id(provider)]:
                try:
                    results = await provider.search(query, limit_per_query)
                    for rank, result in enumerate(results, start=1):
                        result.query_relevance = max(
                            result.query_relevance,
                            _query_relevance(result, query, rank),
                        )
                        if not result.best_provider_rank or rank < result.best_provider_rank:
                            result.best_provider_rank = rank
                    return results
                except Exception as exc:  # noqa: BLE001 - one provider must not stop the rest
                    self.provider_errors[provider.label] = str(exc)
                    log_event(logger, "search.provider_failed", "Provider failed",
                              level=30, provider=provider.label, error=str(exc))
                    return []

        tasks = [run(provider, query) for provider in instances for query in unique_queries]
        batches = await asyncio.gather(*tasks)

        merged = merge_results([result for batch in batches for result in batch])

        if not merged and self.provider_errors:
            raise SearchError(
                "No search provider returned results.",
                details="; ".join(f"{name}: {error}" for name, error in self.provider_errors.items()),
            )

        log_event(logger, "search.aggregated", "Search completed",
                  providers=len(instances), queries=len(unique_queries), papers=len(merged))
        return merged


def merge_results(results: list[SearchResult]) -> list[SearchResult]:
    """Fuse duplicate papers, keeping the best field from every provider."""
    merged: dict[str, SearchResult] = {}

    for result in results:
        key = result.dedup_key()
        existing = merged.get(key)
        if existing is None:
            merged[key] = result.model_copy(deep=True)
            continue
        _fuse(existing, result)

    ordered = sorted(
        merged.values(),
        key=lambda item: (
            item.query_relevance,
            item.is_open_access,
            -(item.best_provider_rank or 10_000),
            item.metadata.citation_count,
        ),
        reverse=True,
    )
    return ordered


def _fuse(target: SearchResult, other: SearchResult) -> None:
    """Merge ``other`` into ``target`` in place."""
    if other.pdf_url and not target.pdf_url:
        target.pdf_url = other.pdf_url
        target.is_open_access = True
    target.is_open_access = target.is_open_access or other.is_open_access
    target.query_relevance = max(target.query_relevance, other.query_relevance)
    ranks = [rank for rank in (target.best_provider_rank, other.best_provider_rank) if rank]
    target.best_provider_rank = min(ranks, default=0)
    _fuse_metadata(target.metadata, other.metadata)


def _query_relevance(result: SearchResult, query: str, rank: int) -> float:
    """Estimate direct topical fit while retaining provider result order."""
    query_terms = _meaningful_terms(query)
    if not query_terms:
        return round(1.0 / (rank + 1), 4)

    metadata = result.metadata
    title_terms = set(_WORD_RE.findall(metadata.title.lower()))
    detail_terms = set(
        _WORD_RE.findall(
            " ".join(
                [metadata.abstract, " ".join(metadata.keywords)]
            ).lower()
        )
    )
    title_coverage = len(query_terms & title_terms) / len(query_terms)
    detail_coverage = len(query_terms & detail_terms) / len(query_terms)
    provider_signal = 1.0 / max(1, rank)
    return round(min(1.0, title_coverage * 0.65 + detail_coverage * 0.25
                     + provider_signal * 0.1), 4)


def _meaningful_terms(text: str) -> set[str]:
    terms = {
        term for term in _WORD_RE.findall(text.lower())
        if len(term) > 1 and term not in _SEARCH_STOPWORDS
    }
    if "ai" in terms:
        terms.update({"artificial", "intelligence"})
    if "ml" in terms:
        terms.update({"machine", "learning"})
    return terms


def _fuse_metadata(target: DocumentMetadata, other: DocumentMetadata) -> None:
    """Fill empty bibliographic fields and keep the strongest numeric signals."""
    for field in (
        "title", "abstract", "doi", "url", "journal", "conference",
        "publisher", "volume", "issue", "pages",
    ):
        current = getattr(target, field)
        candidate = getattr(other, field)
        if candidate and (not current or len(str(candidate)) > len(str(current)) * 1.5):
            setattr(target, field, candidate)

    if not target.authors and other.authors:
        target.authors = list(other.authors)
    if not target.keywords and other.keywords:
        target.keywords = list(other.keywords)
    if not target.publication_year and other.publication_year:
        target.publication_year = other.publication_year

    target.citation_count = max(target.citation_count, other.citation_count)
    target.is_peer_reviewed = target.is_peer_reviewed or other.is_peer_reviewed


def _unique(values: list[str]) -> list[str]:
    """De-duplicate while preserving order."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(value)
    return unique
