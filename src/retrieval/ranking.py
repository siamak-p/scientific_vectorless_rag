"""Paper ranking.

Papers are scored before any of them is read, so the expensive navigation and
extraction budget is only spent on the strongest evidence. Scoring combines a
cheap lexical signal with an optional LLM relevance judgement, plus recency,
citation count and venue quality.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from core.constants import DocumentSource
from core.models import DocumentRecord, PaperRankingEntry
from llm.client import LLMClient
from observability.logger import get_logger, log_event

logger = get_logger("retrieval.ranking")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "compare",
    "comparison", "do", "does", "for", "from", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "their", "there", "these", "this", "to", "using", "we",
    "what", "when", "which", "why", "with",
}

_WEIGHTS = {
    "relevance": 0.45,
    "coverage": 0.20,
    "recency": 0.15,
    "quality": 0.12,
    "citations": 0.08,
}

_UPLOAD_BONUS = 0.05
_RECENCY_HALF_LIFE_YEARS = 6.0

# When only the top `keep` papers will ever be read, judging every candidate
# with the LLM is wasted work: papers that lose badly on the free lexical,
# recency, venue and citation signals are exceedingly unlikely to climb high
# enough to be selected anyway. The shortlist that gets the extra, more
# accurate LLM judgement scales with `keep` itself (see `_judge_budget`), so
# asking for more papers automatically widens the shortlist too - nothing
# about the cutoff is a fixed constant independent of what the user asked for.
_JUDGE_BUDGET_MULTIPLIER = 3
_JUDGE_BUDGET_MARGIN = 5


class _RelevanceJudgement(BaseModel):
    """LLM judgement of how well a paper fits the question."""

    relevance_score: float = Field(default=0.0, description="0.0 to 1.0.")
    coverage_score: float = Field(default=0.0, description="0.0 to 1.0.")
    reason: str = Field(default="", description="One short sentence.")


class _PaperJudgement(_RelevanceJudgement):
    """One relevance judgement linked to the exact document id supplied."""

    document_id: str


class _BatchJudgement(BaseModel):
    """Comparative relevance judgements for the shortlisted papers."""

    judgements: list[_PaperJudgement] = Field(default_factory=list)


class PaperRanker:
    """Ranks candidate papers for one research question."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client
        self.rate_limit_encountered = False

    async def rank(
        self,
        question: str,
        documents: list[DocumentRecord],
        top_k: int | None = None,
    ) -> list[PaperRankingEntry]:
        """Return every document scored and sorted by overall quality.

        Args:
            question: The question papers are being ranked against.
            documents: Every candidate discovered so far.
            top_k: How many papers the caller actually intends to keep. Used
                only to size the LLM-judgement shortlist (see
                `_judge_budget`) - every document is still scored and
                returned, so callers that need a bigger or smaller slice than
                `top_k` remain free to take it from the full result.
        """
        if not documents:
            return []

        judge_at = self._preselect_for_judgement(question, documents, top_k)
        judgements = await self._judge(question, documents, judge_at)

        entries: list[PaperRankingEntry] = []
        for document, judgement in zip(documents, judgements):
            entries.append(self._score(question, document, judgement))

        entries.sort(key=lambda entry: entry.overall_score, reverse=True)
        log_event(logger, "ranking.complete", "Papers ranked",
                  question=question[:120], papers=len(entries), llm_judged=len(judge_at))
        return entries

    def _preselect_for_judgement(
        self, question: str, documents: list[DocumentRecord], top_k: int | None
    ) -> set[int]:
        """Pick which candidates are worth an LLM judgement call.

        The shortlist is sized from `top_k` (the number of papers the caller
        will actually keep), not a fixed constant, so asking for more papers
        widens the shortlist and never starves the ranking of real
        alternatives. When `top_k` is unknown, every candidate is judged.
        """
        # A judgement cannot change which papers are read when every candidate
        # is already inside the reading budget, which is the common case for a
        # small chat corpus. The free signals still order the trace.
        if top_k is not None and len(documents) <= max(1, top_k):
            return set()

        budget = _judge_budget(top_k, len(documents))
        if budget >= len(documents):
            return set(range(len(documents)))

        cheap_scores = [self._score(question, document, None) for document in documents]
        order = sorted(
            range(len(documents)),
            key=lambda index: cheap_scores[index].overall_score,
            reverse=True,
        )
        return set(order[:budget])

    # -- scoring ---------------------------------------------------------

    def _score(
        self,
        question: str,
        document: DocumentRecord,
        judgement: _RelevanceJudgement | None,
    ) -> PaperRankingEntry:
        metadata = document.metadata

        lexical = lexical_overlap(question, _document_terms(document))
        relevance = _clamp(judgement.relevance_score) if judgement else lexical
        coverage = _clamp(judgement.coverage_score) if judgement else lexical
        reason = judgement.reason if judgement else "Scored from title and abstract overlap."

        recency = recency_score(metadata.publication_year)
        quality = venue_quality(document)
        citations = citation_score(metadata.citation_count)

        overall = (
            _WEIGHTS["relevance"] * relevance
            + _WEIGHTS["coverage"] * coverage
            + _WEIGHTS["recency"] * recency
            + _WEIGHTS["quality"] * quality
            + _WEIGHTS["citations"] * citations
        )
        if document.source is DocumentSource.UPLOADED:
            overall += _UPLOAD_BONUS

        return PaperRankingEntry(
            document_id=document.id,
            title=document.label(),
            source=document.source,
            relevance_score=round(relevance, 3),
            recency_score=round(recency, 3),
            citation_count=metadata.citation_count,
            quality_score=round(quality, 3),
            coverage_score=round(coverage, 3),
            overall_score=round(min(1.0, overall), 3),
            reason=reason,
        )

    async def _judge(
        self,
        question: str,
        documents: list[DocumentRecord],
        judge_at: set[int],
    ) -> list[_RelevanceJudgement | None]:
        """Ask the model to judge relevance for the preselected candidates.

        Candidates outside ``judge_at`` fall back to lexical scoring, same as
        when the LLM call itself fails - both are handled identically by
        ``_score``.
        """
        if self._client is None or not judge_at:
            return [None] * len(documents)

        indices = sorted(judge_at)
        rendered: list[str] = []
        for index in indices:
            document = documents[index]
            metadata = document.metadata
            rendered.append(
                f"DOCUMENT {document.id}\n"
                f"title: {document.label()}\n"
                f"authors: {metadata.author_string()}\n"
                f"year: {metadata.publication_year or 'unknown'}\n"
                f"venue: {metadata.journal or metadata.conference or 'unknown'}\n"
                f"keywords: {', '.join(metadata.keywords) or 'none'}\n"
                f"abstract: {(metadata.abstract or 'not available')[:2500]}"
            )

        try:
            result = await self._client.structured(
                "retrieval/paper_relevance_batch",
                _BatchJudgement,
                temperature=0.0,
                question=question,
                papers="\n\n".join(rendered),
                document_ids=[documents[index].id for index in indices],
            )
        except Exception as exc:  # noqa: BLE001 - lexical fallback
            text = str(exc).lower()
            if any(marker in text for marker in (
                "rate limit", "rate_limit", "quota", "tokens per minute", "tpm", "429"
            )):
                self.rate_limit_encountered = True
            log_event(
                logger,
                "ranking.judge",
                "Batch relevance judgement failed; using lexical scores",
                level=30,
                papers=len(indices),
                error=str(exc),
            )
            return [None] * len(documents)

        by_document = {item.document_id: item for item in result.judgements}
        return [by_document.get(document.id) for document in documents]


# ---------------------------------------------------------------------------
# Scoring primitives (pure functions, unit tested directly)
# ---------------------------------------------------------------------------

def _judge_budget(top_k: int | None, total: int) -> int:
    """How many candidates deserve an LLM relevance judgement.

    Scales with `top_k` (how many papers will actually be kept) instead of a
    fixed number, so requesting more papers automatically widens the
    shortlist: a comfortable margin above `top_k` is judged so a cheap-score
    mistake cannot silently exclude a genuinely better paper. When `top_k`
    is unknown, every candidate is judged, same as before this shortlist
    existed.
    """
    if top_k is None or top_k <= 0:
        return total
    return min(total, max(top_k * _JUDGE_BUDGET_MULTIPLIER, top_k + _JUDGE_BUDGET_MARGIN))


def lexical_overlap(query: str, terms: set[str]) -> float:
    """Return the fraction of meaningful query terms present in ``terms``."""
    query_terms = _tokenise(query) - _STOPWORDS
    if not query_terms:
        return 0.5
    overlap = len(query_terms & terms) / len(query_terms)
    return round(min(1.0, 0.15 + 0.85 * overlap), 3)


def recency_score(year: int | None) -> float:
    """Score publication recency with exponential decay."""
    if not year:
        return 0.4
    current = datetime.now(timezone.utc).year
    age = max(0, current - year)
    return round(math.exp(-age / _RECENCY_HALF_LIFE_YEARS), 3)


def citation_score(citations: int) -> float:
    """Score citation impact on a saturating logarithmic scale."""
    if citations <= 0:
        return 0.0
    return round(min(1.0, math.log10(1 + citations) / 4.0), 3)


def venue_quality(document: DocumentRecord) -> float:
    """Score venue quality from the available bibliographic signals."""
    metadata = document.metadata
    score = 0.2
    if metadata.is_peer_reviewed:
        score += 0.35
    if metadata.journal or metadata.conference:
        score += 0.2
    if metadata.doi:
        score += 0.15
    if metadata.venue_rank:
        score += min(0.1, metadata.venue_rank / 10.0)
    return round(min(1.0, score), 3)


def _document_terms(document: DocumentRecord) -> set[str]:
    metadata = document.metadata
    blob = " ".join(
        [
            metadata.title,
            metadata.abstract,
            " ".join(metadata.keywords),
            document.filename,
        ]
    )
    return _tokenise(blob)


def _tokenise(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
