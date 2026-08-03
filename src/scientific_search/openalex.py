"""OpenAlex search provider.

OpenAlex is fully open, requires no key, and exposes open-access locations
that often point at a directly downloadable PDF.
"""

from __future__ import annotations

from typing import Any

from config.catalog import search_endpoint
from core.constants import SearchProviderType
from core.models import DocumentMetadata
from scientific_search.base import ScientificSearchProvider, SearchResult


class OpenAlexProvider(ScientificSearchProvider):
    """Searches the OpenAlex works index."""

    provider_type = SearchProviderType.OPENALEX

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search works and map them onto the shared result model."""
        params: dict[str, Any] = {
            "search": query,
            "per-page": min(limit, 50),
            "sort": "relevance_score:desc",
        }
        if self.settings.contact_email:
            params["mailto"] = self.settings.contact_email

        payload = await self.get_json(search_endpoint(self.provider_type.value), params=params)
        results = [self._to_result(item) for item in payload.get("results") or []]
        self.log_results(query, results)
        return results

    def _to_result(self, item: dict[str, Any]) -> SearchResult:
        best_location = item.get("best_oa_location") or item.get("primary_location") or {}
        source = best_location.get("source") or {}
        biblio = item.get("biblio") or {}
        open_access = item.get("open_access") or {}
        source_type = (source.get("type") or "").lower()

        pdf_url = best_location.get("pdf_url") or open_access.get("oa_url") or ""
        pages = _page_range(biblio)

        metadata = DocumentMetadata(
            title=self.clean_text(item.get("display_name"), 500),
            authors=[
                authorship.get("author", {}).get("display_name", "")
                for authorship in item.get("authorships") or []
                if authorship.get("author", {}).get("display_name")
            ],
            abstract=self.clean_text(_invert_abstract(item.get("abstract_inverted_index"))),
            doi=self.clean_doi(item.get("doi")),
            url=item.get("doi") or item.get("id") or "",
            keywords=[
                concept.get("display_name", "")
                for concept in (item.get("concepts") or [])[:8]
                if concept.get("display_name")
            ],
            publication_year=item.get("publication_year"),
            journal=self.clean_text(source.get("display_name"), 300) if source_type != "conference" else "",
            conference=self.clean_text(source.get("display_name"), 300) if source_type == "conference" else "",
            publisher=self.clean_text(source.get("host_organization_name"), 200),
            volume=self.clean_text(biblio.get("volume"), 40),
            issue=self.clean_text(biblio.get("issue"), 40),
            pages=pages,
            citation_count=int(item.get("cited_by_count") or 0),
            is_peer_reviewed=bool(source.get("display_name")) and item.get("type") == "article",
            provider=self.provider_type.value,
        )

        return SearchResult(
            external_id=item.get("id") or metadata.doi,
            metadata=metadata,
            pdf_url=pdf_url,
            is_open_access=bool(open_access.get("is_oa")) and bool(pdf_url),
            provider=self.provider_type,
        )


def _page_range(biblio: dict[str, Any]) -> str:
    """Join the first and last page into a range."""
    first = biblio.get("first_page") or ""
    last = biblio.get("last_page") or ""
    if first and last:
        return f"{first}-{last}"
    return first or last or ""


def _invert_abstract(index: dict[str, list[int]] | None) -> str:
    """Rebuild an abstract from OpenAlex's inverted word index."""
    if not index:
        return ""
    positions: list[tuple[int, str]] = [
        (position, word) for word, spots in index.items() for position in spots
    ]
    positions.sort()
    return " ".join(word for _, word in positions)
