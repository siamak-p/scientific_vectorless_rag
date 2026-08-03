"""Tavily web search provider.

Tavily broadens coverage to technical reports, documentation and preprint
mirrors that the academic indexes do not carry. It requires an API key and is
therefore skipped automatically when none is configured.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from config.catalog import search_endpoint
from core.constants import HTTP_TIMEOUT_SECONDS, HTTP_USER_AGENT, SearchProviderType
from core.exceptions import SearchProviderError
from core.models import DocumentMetadata
from scientific_search.base import ScientificSearchProvider, SearchResult


class TavilyProvider(ScientificSearchProvider):
    """Searches the web through the Tavily API."""

    provider_type = SearchProviderType.TAVILY

    def is_available(self) -> bool:
        """Tavily is only usable when an API key has been configured."""
        return bool(self.settings.tavily_api_key)

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Run a Tavily search restricted to advanced, content-bearing results."""
        if not self.is_available():
            raise SearchProviderError(
                self.provider_type.label,
                "A Tavily API key is required. Add one on the Settings page.",
            )

        body = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": min(limit, 20),
            "include_answer": False,
            "include_raw_content": True,
        }

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    search_endpoint(self.provider_type.value),
                    json=body,
                    headers={"User-Agent": HTTP_USER_AGENT},
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {401, 403}:
                raise SearchProviderError(
                    self.provider_type.label, "The Tavily API key was rejected."
                ) from exc
            raise SearchProviderError(
                self.provider_type.label, f"Tavily returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.HTTPError as exc:
            raise SearchProviderError(
                self.provider_type.label, "Tavily could not be reached."
            ) from exc

        results = [self._to_result(item) for item in payload.get("results") or []]
        self.log_results(query, results)
        return results

    def _to_result(self, item: dict[str, Any]) -> SearchResult:
        url = item.get("url") or ""
        content = item.get("raw_content") or item.get("content") or ""

        metadata = DocumentMetadata(
            title=self.clean_text(item.get("title"), 500),
            abstract=self.clean_text(content),
            url=url,
            doi=self.clean_doi(url),
            publisher=urlparse(url).netloc,
            is_peer_reviewed=False,
            provider=self.provider_type.value,
        )

        return SearchResult(
            external_id=url,
            metadata=metadata,
            pdf_url=url if url.lower().endswith(".pdf") else "",
            is_open_access=url.lower().endswith(".pdf"),
            provider=self.provider_type,
        )
