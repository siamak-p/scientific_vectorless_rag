"""LLM assisted extraction of bibliographic metadata from a paper."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.models import DocumentMetadata, PageContent
from llm.client import LLMClient
from observability.logger import get_logger, log_event

logger = get_logger("documents.metadata")

_INTRO_PAGES = 3
_MAX_CHARS = 9000


class _MetadataSchema(BaseModel):
    """Structured output schema for metadata extraction."""

    title: str = Field(default="", description="Paper title exactly as printed.")
    authors: list[str] = Field(default_factory=list, description="Ordered author names.")
    abstract: str = Field(default="", description="Abstract text only.")
    doi: str = Field(default="", description="Bare DOI starting with 10., or empty.")
    keywords: list[str] = Field(default_factory=list, description="Explicit keywords only.")
    publication_year: Optional[int] = Field(default=None, description="Publication year or null.")
    journal: str = Field(default="", description="Journal name for journal articles.")
    conference: str = Field(default="", description="Conference name for conference papers.")
    publisher: str = Field(default="", description="Publisher name if printed.")
    volume: str = Field(default="", description="Volume if printed.")
    issue: str = Field(default="", description="Issue if printed.")
    pages: str = Field(default="", description="Page range if printed.")
    is_peer_reviewed: bool = Field(default=False, description="True only for refereed venues.")


class MetadataExtractor:
    """Extracts bibliographic metadata from the opening pages of a paper."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def extract(
        self, pages: list[PageContent], existing: DocumentMetadata | None = None
    ) -> DocumentMetadata:
        """Extract metadata, merging it into any metadata already known.

        Metadata supplied by a search provider is authoritative and is never
        overwritten by the model; the model only fills the gaps.
        """
        base = existing.model_copy(deep=True) if existing else DocumentMetadata()

        # arXiv and the other scientific-search providers already return the
        # fields needed to identify and rank a paper. Re-reading the opening
        # pages with an LLM merely to fill optional venue fields costs one
        # request per downloaded paper and can dominate ingestion under API
        # rate limits. Incomplete search records and user uploads still take
        # the extraction path below.
        if base.title.strip() and base.authors and base.abstract.strip():
            return base

        intro = "\n\n".join(page.text for page in pages[:_INTRO_PAGES])[:_MAX_CHARS]
        if not intro.strip():
            return base

        try:
            result = await self._client.structured(
                "documents/metadata_extraction",
                _MetadataSchema,
                temperature=0.0,
                text=intro,
            )
        except Exception as exc:  # noqa: BLE001 - metadata is best effort
            log_event(logger, "metadata.extract", "Metadata extraction failed",
                      level=30, error=str(exc))
            return base

        return _merge(base, result)


def _merge(base: DocumentMetadata, extracted: _MetadataSchema) -> DocumentMetadata:
    """Fill empty fields of ``base`` from the extracted metadata."""
    merged = base.model_copy(deep=True)

    for field in ("title", "abstract", "doi", "journal", "conference",
                  "publisher", "volume", "issue", "pages"):
        if not getattr(merged, field):
            setattr(merged, field, getattr(extracted, field, "") or "")

    if not merged.authors:
        merged.authors = [a.strip() for a in extracted.authors if a.strip()]
    if not merged.keywords:
        merged.keywords = [k.strip() for k in extracted.keywords if k.strip()]
    if merged.publication_year is None:
        merged.publication_year = extracted.publication_year
    if not merged.is_peer_reviewed:
        merged.is_peer_reviewed = extracted.is_peer_reviewed

    merged.doi = merged.doi.replace("https://doi.org/", "").replace("doi:", "").strip()
    return merged
