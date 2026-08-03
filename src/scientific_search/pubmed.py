"""PubMed search provider.

PubMed uses the two-step E-utilities protocol: ``esearch`` returns matching
PMIDs, ``esummary`` returns their metadata. Full text lives in PubMed Central,
so an open-access PDF link is only offered when a PMC identifier exists.
"""

from __future__ import annotations

from typing import Any

from config.catalog import search_endpoint
from core.constants import SearchProviderType
from core.models import DocumentMetadata
from scientific_search.base import ScientificSearchProvider, SearchResult

_PMC_PDF = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"
_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


class PubMedProvider(ScientificSearchProvider):
    """Searches PubMed through the NCBI E-utilities."""

    provider_type = SearchProviderType.PUBMED

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Resolve identifiers, then fetch and map their summaries."""
        base = search_endpoint(self.provider_type.value)

        identifiers = await self._esearch(base, query, limit)
        if not identifiers:
            self.log_results(query, [])
            return []

        summaries = await self.get_json(
            f"{base}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(identifiers), "retmode": "json"},
        )
        records = summaries.get("result") or {}

        results = [
            self._to_result(records[pmid])
            for pmid in identifiers
            if isinstance(records.get(pmid), dict)
        ]
        self.log_results(query, results)
        return results

    async def _esearch(self, base: str, query: str, limit: int) -> list[str]:
        payload = await self.get_json(
            f"{base}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": min(limit, 50),
                "retmode": "json",
                "sort": "relevance",
            },
        )
        return list((payload.get("esearchresult") or {}).get("idlist") or [])

    def _to_result(self, item: dict[str, Any]) -> SearchResult:
        pmid = str(item.get("uid") or "")
        article_ids = {
            (entry.get("idtype") or "").lower(): entry.get("value") or ""
            for entry in item.get("articleids") or []
        }
        pmcid = article_ids.get("pmc", "")

        metadata = DocumentMetadata(
            title=self.clean_text(item.get("title"), 500),
            authors=[
                author.get("name", "")
                for author in item.get("authors") or []
                if author.get("name")
            ],
            doi=self.clean_doi(article_ids.get("doi")),
            url=_ARTICLE_URL.format(pmid=pmid) if pmid else "",
            publication_year=_year(item.get("pubdate") or item.get("epubdate") or ""),
            journal=self.clean_text(item.get("fulljournalname") or item.get("source"), 300),
            volume=self.clean_text(item.get("volume"), 40),
            issue=self.clean_text(item.get("issue"), 40),
            pages=self.clean_text(item.get("pages"), 40),
            is_peer_reviewed=True,
            provider=self.provider_type.value,
        )

        return SearchResult(
            external_id=pmid,
            metadata=metadata,
            pdf_url=_PMC_PDF.format(pmcid=pmcid) if pmcid else "",
            is_open_access=bool(pmcid),
            provider=self.provider_type,
        )


def _year(pubdate: str) -> int | None:
    """Extract a four digit year from a PubMed date string."""
    token = pubdate.strip().split(" ")[0]
    return int(token) if token.isdigit() and len(token) == 4 else None
