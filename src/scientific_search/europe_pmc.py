"""Europe PMC search provider.

Europe PMC mirrors PubMed and PubMed Central and adds full-text links for
open-access articles, all through a key-free REST API.
"""

from __future__ import annotations

from typing import Any

from config.catalog import search_endpoint
from core.constants import SearchProviderType
from core.models import DocumentMetadata
from scientific_search.base import ScientificSearchProvider, SearchResult

_PDF_URL = "https://europepmc.org/articles/{pmcid}?pdf=render"
_ARTICLE_URL = "https://europepmc.org/article/{source}/{identifier}"


class EuropePMCProvider(ScientificSearchProvider):
    """Searches the Europe PMC REST API."""

    provider_type = SearchProviderType.EUROPE_PMC

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search articles and map them onto the shared result model."""
        params: dict[str, Any] = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": min(limit, 100),
        }
        if self.settings.contact_email:
            params["email"] = self.settings.contact_email

        payload = await self.get_json(search_endpoint(self.provider_type.value), params=params)
        items = (payload.get("resultList") or {}).get("result") or []
        results = [self._to_result(item) for item in items]
        self.log_results(query, results)
        return results

    def _to_result(self, item: dict[str, Any]) -> SearchResult:
        pmcid = item.get("pmcid") or ""
        journal_info = item.get("journalInfo") or {}
        journal = (journal_info.get("journal") or {}).get("title") or ""
        is_open_access = str(item.get("isOpenAccess") or "").upper() == "Y"

        pdf_url = ""
        for link in (item.get("fullTextUrlList") or {}).get("fullTextUrl") or []:
            if (link.get("documentStyle") or "").lower() == "pdf" and link.get("url"):
                pdf_url = link["url"]
                break
        if not pdf_url and pmcid and is_open_access:
            pdf_url = _PDF_URL.format(pmcid=pmcid)

        source = item.get("source") or "MED"
        identifier = item.get("id") or item.get("pmid") or pmcid
        authors = [
            name.strip()
            for name in (item.get("authorString") or "").split(",")
            if name.strip()
        ]
        year = item.get("pubYear")

        metadata = DocumentMetadata(
            title=self.clean_text(item.get("title"), 500),
            authors=authors,
            abstract=self.clean_text(item.get("abstractText")),
            doi=self.clean_doi(item.get("doi")),
            url=_ARTICLE_URL.format(source=source, identifier=identifier) if identifier else "",
            keywords=[kw for kw in (item.get("keywordList") or {}).get("keyword") or [] if kw][:8],
            publication_year=int(year) if str(year or "").isdigit() else None,
            journal=self.clean_text(journal, 300),
            volume=self.clean_text(journal_info.get("volume"), 40),
            issue=self.clean_text(journal_info.get("issue"), 40),
            pages=self.clean_text(item.get("pageInfo"), 40),
            citation_count=int(item.get("citedByCount") or 0),
            is_peer_reviewed=bool(journal),
            provider=self.provider_type.value,
        )

        return SearchResult(
            external_id=f"{source}:{identifier}" if identifier else metadata.doi,
            metadata=metadata,
            pdf_url=pdf_url,
            is_open_access=bool(pdf_url),
            provider=self.provider_type,
        )
