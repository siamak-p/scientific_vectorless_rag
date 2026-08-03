"""Evidence extraction and evidence bookkeeping.

Only the pages chosen by the navigator are read, one page at a time, so every
extracted item keeps an exact page-level provenance for citation.
"""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel, Field

from core.constants import (
    EVIDENCE_CONFIDENCE_THRESHOLD,
    EXTRACTION_CONCURRENCY,
    MAX_EVIDENCE_ITEMS,
)
from core.models import DocumentRecord, EvidenceItem, PageContent
from documents.parser import section_hint
from llm.client import LLMClient
from llm.usage import approximate_tokens
from observability.logger import get_logger, log_event

logger = get_logger("retrieval.evidence")

_MAX_PAGE_CHARS = 8_000
_MAX_BATCH_TOKENS = 2_800
_MAX_FALLBACK_ITEMS = 8
_CAPACITY_ERROR_MARKERS = (
    "rate limit",
    "rate_limit",
    "tokens per minute",
    "request too large",
    "error code: 413",
    "error code: 429",
)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_PASSAGE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "what", "when", "which", "why", "with",
}


class _ExtractedItem(BaseModel):
    """One evidence item as produced by the model."""

    page_number: int | None = Field(
        default=None, description="Physical page number copied from the supplied page tag."
    )
    content: str = Field(description="Self contained statement supported by this page.")
    quote: str = Field(default="", description="Verbatim span, at most forty words.")
    section: str = Field(default="", description="Heading the content belongs to.")
    retrieval_reason: str = Field(default="", description="Why it answers the question.")
    confidence_score: float = Field(default=0.0, description="0.0 to 1.0.")
    figure_number: str = Field(default="", description="Figure label, when applicable.")
    table_number: str = Field(default="", description="Table label, when applicable.")


class _ExtractionResult(BaseModel):
    """The page-level extraction payload."""

    items: list[_ExtractedItem] = Field(default_factory=list)


class EvidenceExtractor:
    """Reads selected pages and produces citable evidence items."""

    def __init__(
        self,
        client: LLMClient,
        confidence_threshold: float = EVIDENCE_CONFIDENCE_THRESHOLD,
        concurrency: int = EXTRACTION_CONCURRENCY,
    ) -> None:
        self._client = client
        self._threshold = confidence_threshold
        self._concurrency = max(1, concurrency)
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self.rate_limit_encountered = False

    async def extract(
        self,
        document: DocumentRecord,
        pages: list[PageContent],
        question: str,
        outline: list | None = None,
    ) -> list[EvidenceItem]:
        """Extract evidence from the given pages of one document."""
        if not pages:
            return []

        title = document.label()

        async def read_once(batch: list[PageContent]) -> list[EvidenceItem]:
            async with self._semaphore:
                return await self._read_pages(
                    document, title, batch, question, outline or []
                )

        async def read_resilient(batch: list[PageContent]) -> list[EvidenceItem]:
            try:
                return await read_once(batch)
            except Exception as exc:  # noqa: BLE001 - recover without losing the pages
                if _is_capacity_error(exc):
                    if _is_rate_limit_error(exc):
                        self.rate_limit_encountered = True
                    log_event(
                        logger,
                        "evidence.extract",
                        "Provider capacity prevented extraction; using page-backed passages",
                        level=30,
                        document=title,
                        pages=[page.page_number for page in batch],
                        error=str(exc),
                    )
                    return _fallback_evidence(
                        document, title, batch, question, outline or []
                    )
                if len(batch) > 1:
                    middle = len(batch) // 2
                    log_event(
                        logger,
                        "evidence.extract",
                        "Page batch failed; retrying smaller batches",
                        level=30,
                        document=title,
                        pages=[page.page_number for page in batch],
                        error=str(exc),
                    )
                    left, right = await asyncio.gather(
                        read_resilient(batch[:middle]),
                        read_resilient(batch[middle:]),
                    )
                    return left + right
                log_event(
                    logger,
                    "evidence.extract",
                    "Single-page extraction failed; using an exact page-backed passage",
                    level=30,
                    document=title,
                    page=batch[0].page_number,
                    error=str(exc),
                )
                return _fallback_evidence(
                    document, title, batch, question, outline or []
                )

        batches = await asyncio.gather(
            *(read_resilient(batch) for batch in _page_batches(pages))
        )

        evidence: list[EvidenceItem] = []
        for batch in batches:
            evidence.extend(batch)

        if not evidence:
            evidence = _fallback_evidence(
                document, title, pages, question, outline or []
            )

        log_event(
            logger,
            "evidence.extract",
            "Evidence extracted",
            document=title,
            pages=len(pages),
            items=len(evidence),
        )
        return evidence

    async def extract_from_text(
        self,
        document: DocumentRecord,
        text: str,
        question: str,
    ) -> list[EvidenceItem]:
        """Extract evidence from a text-only source such as a web result."""
        if not text.strip():
            return []
        page = PageContent(page_number=1, text=text, char_count=len(text))
        async with self._semaphore:
            return await self._read_pages(
                document, document.label(), [page], question, []
            )

    # -- internals -------------------------------------------------------

    async def _read_pages(
        self,
        document: DocumentRecord,
        title: str,
        pages: list[PageContent],
        question: str,
        outline: list,
    ) -> list[EvidenceItem]:
        usable = [page for page in pages if page.text.strip()]
        if not usable:
            return []

        page_hints = {
            page.page_number: section_hint(outline, page.page_number) if outline else ""
            for page in usable
        }
        rendered_pages = "\n\n".join(
            f"=== PAGE {page.page_number} ===\n"
            f"Section hint: {page_hints[page.page_number] or 'not available'}\n"
            f"{page.text[:_MAX_PAGE_CHARS]}"
            for page in usable
        )

        result = await self._client.structured(
            "reasoning/evidence_extraction",
            _ExtractionResult,
            temperature=0.0,
            question=question,
            document_title=title,
            pages=rendered_pages,
        )

        valid_page_numbers = {page.page_number for page in usable}
        only_page = usable[0].page_number if len(usable) == 1 else None
        items: list[EvidenceItem] = []
        for raw in result.items:
            score = max(0.0, min(1.0, float(raw.confidence_score)))
            if score < self._threshold or not raw.content.strip():
                continue
            page_number = raw.page_number or only_page
            if page_number not in valid_page_numbers:
                continue
            hint = page_hints[page_number]
            items.append(
                EvidenceItem(
                    document_id=document.id,
                    document_title=title,
                    page_number=page_number,
                    section=raw.section.strip() or hint,
                    content=raw.content.strip(),
                    quote=raw.quote.strip(),
                    retrieval_reason=raw.retrieval_reason.strip(),
                    confidence_score=round(score, 3),
                    sub_question=question,
                    figure_number=raw.figure_number.strip(),
                    table_number=raw.table_number.strip(),
                )
            )
        return items


def _page_batches(pages: list[PageContent]) -> list[list[PageContent]]:
    """Group pages under a conservative input-token budget."""
    batches: list[list[PageContent]] = []
    current: list[PageContent] = []
    current_tokens = 0

    for page in pages:
        page_tokens = approximate_tokens(page.text[:_MAX_PAGE_CHARS])
        if current and current_tokens + page_tokens > _MAX_BATCH_TOKENS:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(page)
        current_tokens += page_tokens

    if current:
        batches.append(current)
    return batches


def _is_capacity_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CAPACITY_ERROR_MARKERS)


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in (
        "rate limit", "rate_limit", "quota", "tokens per minute", "tpm", "429"
    ))


def _fallback_evidence(
    document: DocumentRecord,
    title: str,
    pages: list[PageContent],
    question: str,
    outline: list,
) -> list[EvidenceItem]:
    """Select exact source passages deterministically when the model cannot extract.

    These are verbatim page passages, not generated claims, so answer generation
    remains grounded and every item retains its physical page provenance.
    """
    query_terms = _meaningful_terms(question)
    candidates: list[tuple[float, PageContent, str]] = []
    for page in pages:
        for passage in _PASSAGE_RE.split(page.text):
            cleaned = " ".join(passage.split()).strip()
            if len(cleaned) < 40:
                continue
            passage_terms = set(_WORD_RE.findall(cleaned.lower()))
            matches = len(query_terms & passage_terms)
            if matches == 0:
                continue
            score = matches / max(1, len(query_terms))
            candidates.append((score, page, cleaned[:1_200]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    evidence: list[EvidenceItem] = []
    seen: set[tuple[int, str]] = set()
    for score, page, passage in candidates:
        fingerprint = (page.page_number, passage[:160].lower())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        hint = section_hint(outline, page.page_number) if outline else ""
        evidence.append(
            EvidenceItem(
                document_id=document.id,
                document_title=title,
                page_number=page.page_number,
                section=hint,
                content=passage,
                quote=" ".join(passage.split()[:40]),
                retrieval_reason="Exact page passage selected after structured extraction failed.",
                confidence_score=round(min(0.7, 0.45 + score * 0.25), 3),
                sub_question=question,
            )
        )
        if len(evidence) >= _MAX_FALLBACK_ITEMS:
            break
    return evidence


def _meaningful_terms(text: str) -> set[str]:
    terms = {
        term for term in _WORD_RE.findall(text.lower())
        if len(term) > 1 and term not in _FALLBACK_STOPWORDS
    }
    if "ai" in terms:
        terms.update({"artificial", "intelligence"})
    if "ml" in terms:
        terms.update({"machine", "learning"})
    return terms


def deduplicate(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    """Drop near-identical evidence produced by overlapping sub-questions."""
    seen: set[tuple[str, int, str]] = set()
    unique: list[EvidenceItem] = []

    for item in sorted(evidence, key=lambda e: e.confidence_score, reverse=True):
        fingerprint = (
            item.document_id,
            item.page_number,
            " ".join(item.content.lower().split())[:160],
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(item)

    return unique


def select_top(evidence: list[EvidenceItem], limit: int = MAX_EVIDENCE_ITEMS) -> list[EvidenceItem]:
    """Keep the strongest evidence while preserving document diversity.

    A single verbose paper must not crowd out the others, so items are taken
    round-robin across documents in descending confidence order.
    """
    by_document: dict[str, list[EvidenceItem]] = {}
    for item in sorted(evidence, key=lambda e: e.confidence_score, reverse=True):
        by_document.setdefault(item.document_id, []).append(item)

    selected: list[EvidenceItem] = []
    while len(selected) < limit and any(by_document.values()):
        for items in by_document.values():
            if not items:
                continue
            selected.append(items.pop(0))
            if len(selected) >= limit:
                break

    return sorted(selected, key=lambda e: e.confidence_score, reverse=True)


def format_evidence_block(evidence: list[EvidenceItem]) -> str:
    """Render evidence as the numbered context block used by answer prompts."""
    lines: list[str] = []
    for index, item in enumerate(evidence, start=1):
        location = f"page {item.page_number}"
        if item.section:
            location += f", section {item.section}"
        lines.append(
            f"E{index} | id={item.id} | {item.document_title} | {location}\n"
            f"{item.content}"
            + (f"\nVerbatim: {item.quote}" if item.quote else "")
        )
    return "\n\n".join(lines) if lines else "No evidence was retrieved."
