"""DOAJ (Directory of Open Access Journals) search provider.

Every article DOAJ indexes is open access by definition, so full-text links
are common. The API is public and needs no key.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from config.catalog import search_endpoint
from core.constants import SearchProviderType
from core.models import DocumentMetadata
from scientific_search.base import ScientificSearchProvider, SearchResult


class DOAJProvider(ScientificSearchProvider):
    """Searches DOAJ articles."""

    provider_type = SearchProviderType.DOAJ

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search articles and map them onto the shared result model."""
        url = f"{search_endpoint(self.provider_type.value).rstrip('/')}/{quote(query, safe='')}"
        payload = await self.get_json(url, params={"pageSize": min(limit, 100)})
        results = [self._to_result(item) for item in payload.get("results") or []]
        self.log_results(query, results)
        return results

    def _to_result(self, item: dict[str, Any]) -> SearchResult:
        bib = item.get("bibjson") or {}
        journal = bib.get("journal") or {}

        doi = ""
        for identifier in bib.get("identifier") or []:
            if (identifier.get("type") or "").lower() == "doi":
                doi = self.clean_doi(identifier.get("id"))
                break

        fulltext = ""
        for link in bib.get("link") or []:
            if (link.get("type") or "").lower() == "fulltext" and link.get("url"):
                fulltext = link["url"]
                break
        pdf_url = fulltext if fulltext.lower().endswith(".pdf") or "pdf" in fulltext.lower() else ""

        year = bib.get("year")
        metadata = DocumentMetadata(
            title=self.clean_text(bib.get("title"), 500),
            authors=[a.get("name", "") for a in bib.get("author") or [] if a.get("name")],
            abstract=self.clean_text(bib.get("abstract")),
            doi=doi,
            url=fulltext or (f"https://doi.org/{doi}" if doi else ""),
            keywords=[kw for kw in bib.get("keywords") or [] if kw][:8],
            publication_year=int(year) if str(year or "").isdigit() else None,
            journal=self.clean_text(journal.get("title"), 300),
            publisher=self.clean_text(journal.get("publisher"), 200),
            volume=self.clean_text(journal.get("volume"), 40),
            issue=self.clean_text(journal.get("number"), 40),
            pages=self.clean_text(
                "-".join(p for p in (bib.get("start_page"), bib.get("end_page")) if p), 40
            ),
            is_peer_reviewed=True,
            provider=self.provider_type.value,
        )

        return SearchResult(
            external_id=item.get("id") or doi,
            metadata=metadata,
            pdf_url=pdf_url,
            is_open_access=bool(pdf_url),
            provider=self.provider_type,
        )
