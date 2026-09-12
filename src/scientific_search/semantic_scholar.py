"""Semantic Scholar search provider.

Provides citation counts, open-access PDF links and venue information. Works
without a key at a lower rate limit; a key is used when configured.
"""

from __future__ import annotations

from typing import Any

from config.catalog import search_endpoint
from core.constants import SearchProviderType
from core.models import DocumentMetadata
from scientific_search.base import ScientificSearchProvider, SearchResult

_FIELDS = (
    "title,abstract,year,authors,externalIds,openAccessPdf,venue,publicationVenue,"
    "citationCount,publicationTypes,journal,url,fieldsOfStudy"
)


class SemanticScholarProvider(ScientificSearchProvider):
    """Searches the Semantic Scholar Graph API."""

    provider_type = SearchProviderType.SEMANTIC_SCHOLAR

    def max_concurrency(self) -> int:
        # Unauthenticated access is limited to roughly one request per second;
        # firing every query at once guarantees a 429 for all but the first.
        return super().max_concurrency() if self.settings.semantic_scholar_api_key else 1

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search papers and map them onto the shared result model."""
        headers: dict[str, str] = {}
        if self.settings.semantic_scholar_api_key:
            headers["x-api-key"] = self.settings.semantic_scholar_api_key

        payload = await self.get_json(
            search_endpoint(self.provider_type.value),
            params={"query": query, "limit": min(limit, 100), "fields": _FIELDS},
            headers=headers,
        )

        results = [self._to_result(item) for item in payload.get("data") or []]
        self.log_results(query, results)
        return results

    def _to_result(self, item: dict[str, Any]) -> SearchResult:
        external = item.get("externalIds") or {}
        open_access = item.get("openAccessPdf") or {}
        journal = item.get("journal") or {}
        venue_info = item.get("publicationVenue") or {}
        venue_type = (venue_info.get("type") or "").lower()
        types = [entry.lower() for entry in item.get("publicationTypes") or []]

        is_conference = venue_type == "conference" or "conference" in types
        venue_name = self.clean_text(item.get("venue"), 300)

        metadata = DocumentMetadata(
            title=self.clean_text(item.get("title"), 500),
            authors=[author.get("name", "") for author in item.get("authors") or [] if author.get("name")],
            abstract=self.clean_text(item.get("abstract")),
            doi=self.clean_doi(external.get("DOI")),
            url=item.get("url") or "",
            keywords=[field for field in item.get("fieldsOfStudy") or [] if field],
            publication_year=item.get("year"),
            journal="" if is_conference else self.clean_text(journal.get("name") or venue_name, 300),
            conference=venue_name if is_conference else "",
            volume=self.clean_text(journal.get("volume"), 40),
            pages=self.clean_text(journal.get("pages"), 40),
            citation_count=int(item.get("citationCount") or 0),
            is_peer_reviewed="journalarticle" in types or is_conference,
            provider=self.provider_type.value,
        )

        return SearchResult(
            external_id=item.get("paperId") or metadata.doi,
            metadata=metadata,
            pdf_url=open_access.get("url") or "",
            is_open_access=bool(open_access.get("url")),
            provider=self.provider_type,
        )
