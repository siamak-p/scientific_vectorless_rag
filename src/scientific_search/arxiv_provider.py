"""arXiv search provider.

arXiv is preprint-only, always open access, and needs no credential, which
makes it the default source for full-text retrieval.
"""

from __future__ import annotations

import asyncio

import arxiv

from core.constants import SearchProviderType
from core.exceptions import SearchProviderError
from core.models import DocumentMetadata
from scientific_search.base import ScientificSearchProvider, SearchResult


class ArxivProvider(ScientificSearchProvider):
    """Searches arXiv through its official client."""

    provider_type = SearchProviderType.ARXIV

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search arXiv, running the blocking client in a worker thread."""
        try:
            entries = await asyncio.to_thread(self._search_sync, query, limit)
        except Exception as exc:  # noqa: BLE001 - normalised for the UI
            raise SearchProviderError(
                self.provider_type.label, "The arXiv query could not be completed."
            ) from exc

        results = [self._to_result(entry) for entry in entries]
        self.log_results(query, results)
        return results

    def _search_sync(self, query: str, limit: int) -> list[arxiv.Result]:
        client = arxiv.Client(page_size=min(limit, 50), delay_seconds=3.0, num_retries=2)
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        return list(client.results(search))

    def _to_result(self, entry: arxiv.Result) -> SearchResult:
        identifier = entry.get_short_id()
        metadata = DocumentMetadata(
            title=self.clean_text(entry.title, 500),
            authors=[author.name for author in entry.authors],
            abstract=self.clean_text(entry.summary),
            doi=self.clean_doi(entry.doi),
            url=entry.entry_id,
            keywords=list(entry.categories),
            publication_year=entry.published.year if entry.published else None,
            journal=self.clean_text(entry.journal_ref, 300),
            is_peer_reviewed=bool(entry.journal_ref),
            provider=self.provider_type.value,
        )
        return SearchResult(
            external_id=identifier,
            metadata=metadata,
            pdf_url=entry.pdf_url or "",
            is_open_access=True,
            provider=self.provider_type,
        )
