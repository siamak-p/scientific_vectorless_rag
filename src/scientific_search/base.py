"""Abstractions shared by every scientific search provider."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, Field

from core.constants import (
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
    SearchProviderType,
)
from core.exceptions import SearchProviderError
from core.models import AppSettings, DocumentMetadata
from observability.logger import get_logger, log_event

logger = get_logger("scientific_search.base")

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


class SearchResult(BaseModel):
    """A single paper discovered by a search provider."""

    external_id: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    pdf_url: str = ""
    is_open_access: bool = False
    provider: SearchProviderType
    query_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    best_provider_rank: int = Field(default=0, ge=0)

    def dedup_key(self) -> str:
        """Return the key used to merge the same paper across providers."""
        if self.metadata.doi:
            return f"doi:{self.metadata.doi.lower()}"
        title = re.sub(r"[^a-z0-9]+", "", self.metadata.title.lower())
        return f"title:{title}" if title else f"id:{self.provider.value}:{self.external_id}"


class ScientificSearchProvider(ABC):
    """Base class for academic search backends."""

    provider_type: SearchProviderType

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Return papers matching a query, ordered by the provider's relevance."""

    def is_available(self) -> bool:
        """Whether the provider is usable with the current configuration."""
        return True

    # -- shared HTTP helpers ---------------------------------------------

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform a GET request returning JSON with unified error handling."""
        payload = await self._request(url, params, headers)
        try:
            return payload.json()
        except ValueError as exc:
            raise SearchProviderError(self.provider_type.label, "The response was not valid JSON.") from exc

    async def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Perform a GET request returning raw text."""
        return (await self._request(url, params, headers)).text

    async def _request(
        self,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> httpx.Response:
        merged = {"User-Agent": self._user_agent(), **(headers or {})}
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
                response = await client.get(url, params=params, headers=merged)
                response.raise_for_status()
                return response
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise SearchProviderError(
                    self.provider_type.label, "The provider rate limit was reached."
                ) from exc
            raise SearchProviderError(
                self.provider_type.label, f"The provider returned HTTP {status}."
            ) from exc
        except httpx.HTTPError as exc:
            raise SearchProviderError(
                self.provider_type.label, "The provider could not be reached."
            ) from exc

    def _user_agent(self) -> str:
        """Return a polite user agent, including a contact address when known."""
        contact = (self.settings.contact_email or "").strip()
        return f"{HTTP_USER_AGENT} mailto:{contact}" if contact else HTTP_USER_AGENT

    # -- shared parsing helpers ------------------------------------------

    @staticmethod
    def clean_doi(value: str | None) -> str:
        """Extract a bare DOI from any URL or prefixed form."""
        if not value:
            return ""
        match = _DOI_RE.search(value)
        return match.group(0).rstrip(".,;") if match else ""

    @staticmethod
    def clean_text(value: str | None, limit: int = 6000) -> str:
        """Collapse whitespace and strip inline markup from provider text."""
        if not value:
            return ""
        without_tags = re.sub(r"<[^>]+>", " ", value)
        return " ".join(without_tags.split())[:limit]

    def log_results(self, query: str, results: list[SearchResult]) -> None:
        """Emit a structured record of what a provider returned."""
        log_event(
            logger,
            "search.results",
            f"{self.provider_type.label} returned {len(results)} result(s)",
            provider=self.provider_type.value,
            query=query[:120],
            open_access=sum(1 for r in results if r.is_open_access),
        )
