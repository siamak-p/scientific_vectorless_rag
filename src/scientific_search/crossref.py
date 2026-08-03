"""Crossref search provider.

Crossref is the authority for DOIs and publisher metadata. It rarely exposes
full text, so it mainly improves the bibliographic quality of citations.
"""

from __future__ import annotations

from typing import Any

from config.catalog import search_endpoint
from core.constants import SearchProviderType
from core.models import DocumentMetadata
from scientific_search.base import ScientificSearchProvider, SearchResult


class CrossrefProvider(ScientificSearchProvider):
    """Searches the Crossref works API."""

    provider_type = SearchProviderType.CROSSREF

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search works and map them onto the shared result model."""
        params: dict[str, Any] = {
            "query.bibliographic": query,
            "rows": min(limit, 50),
            "select": (
                "DOI,title,author,abstract,container-title,issued,volume,issue,page,"
                "publisher,type,is-referenced-by-count,link,URL,subject"
            ),
        }
        if self.settings.contact_email:
            params["mailto"] = self.settings.contact_email

        payload = await self.get_json(search_endpoint(self.provider_type.value), params=params)
        items = (payload.get("message") or {}).get("items") or []
        results = [self._to_result(item) for item in items]
        self.log_results(query, results)
        return results

    def _to_result(self, item: dict[str, Any]) -> SearchResult:
        entry_type = (item.get("type") or "").lower()
        is_conference = entry_type in {"proceedings-article", "proceedings"}
        container = _first(item.get("container-title"))
        pdf_url = _pdf_link(item.get("link") or [])

        metadata = DocumentMetadata(
            title=self.clean_text(_first(item.get("title")), 500),
            authors=[_author_name(author) for author in item.get("author") or [] if _author_name(author)],
            abstract=self.clean_text(item.get("abstract")),
            doi=self.clean_doi(item.get("DOI")),
            url=item.get("URL") or "",
            keywords=[subject for subject in item.get("subject") or [] if subject],
            publication_year=_issued_year(item.get("issued")),
            journal="" if is_conference else self.clean_text(container, 300),
            conference=self.clean_text(container, 300) if is_conference else "",
            publisher=self.clean_text(item.get("publisher"), 200),
            volume=self.clean_text(item.get("volume"), 40),
            issue=self.clean_text(item.get("issue"), 40),
            pages=self.clean_text(item.get("page"), 40),
            citation_count=int(item.get("is-referenced-by-count") or 0),
            is_peer_reviewed=entry_type in {"journal-article", "proceedings-article"},
            provider=self.provider_type.value,
        )

        return SearchResult(
            external_id=metadata.doi,
            metadata=metadata,
            pdf_url=pdf_url,
            is_open_access=bool(pdf_url),
            provider=self.provider_type,
        )


def _first(value: Any) -> str:
    """Return the first element of a Crossref list field."""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _author_name(author: dict[str, Any]) -> str:
    """Join Crossref given and family names."""
    given = (author.get("given") or "").strip()
    family = (author.get("family") or "").strip()
    if given and family:
        return f"{given} {family}"
    return family or given or (author.get("name") or "").strip()


def _issued_year(issued: dict[str, Any] | None) -> int | None:
    """Extract the publication year from a Crossref date-parts structure."""
    parts = (issued or {}).get("date-parts") or []
    if parts and parts[0]:
        try:
            return int(parts[0][0])
        except (TypeError, ValueError):
            return None
    return None


def _pdf_link(links: list[dict[str, Any]]) -> str:
    """Return the first PDF full-text link, if the publisher offers one."""
    for link in links:
        if (link.get("content-type") or "").lower() == "application/pdf":
            return link.get("URL") or ""
    return ""
