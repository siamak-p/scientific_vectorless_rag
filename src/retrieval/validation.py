"""Hallucination control.

Two independent mechanisms protect the answer:

1. a deterministic marker audit, which removes any citation pointing at
   evidence that does not exist and flags factual sentences with no citation;
2. an LLM audit, which classifies each claim against the evidence and reports
   which sentences must be removed.

Unsupported sentences are deleted rather than rewritten, so the answer can
only ever shrink towards what the evidence actually supports.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from core.constants import ClaimVerdict
from core.models import ClaimValidation, EvidenceItem, ValidationReport
from llm.client import LLMClient
from observability.logger import get_logger, log_event

logger = get_logger("retrieval.validation")

_MARKER_RE = re.compile(r"\[\s*E\s*(\d{1,3})\s*\]", re.IGNORECASE)
# Sentence-final punctuation of any script (Latin, Arabic/Persian, Devanagari,
# CJK) followed by whitespace; the old Latin-uppercase lookahead never split
# non-Latin answers, so a whole paragraph counted as one sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?\u061f\u0964\u3002\uff01\uff1f])\s+")
_HEADING_RE = re.compile(r"^\s*(#{1,6}\s|\||-{3,}|\*\s|\d+\.\s*$)")
_MIN_FACTUAL_WORDS = 6
_AUDIT_FALLBACK_NOTE = (
    "The claim-level audit could not run, so this score counts citation markers only "
    "and does not judge whether the cited evidence supports each statement."
)
_UNCITED_NOTICE = (
    "> **Grounding notice.** Some statements were removed because the retrieved "
    "evidence did not support them."
)


class _ClaimAudit(BaseModel):
    """One audited claim."""

    claim: str = Field(description="The claim as written in the answer.")
    verdict: ClaimVerdict = Field(description="supported, partially_supported or unsupported.")
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = Field(default="")


class _AuditResult(BaseModel):
    """The full audit payload."""

    claims: list[_ClaimAudit] = Field(default_factory=list)
    removed_statements: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    grounding_score: float = Field(default=1.0)
    needs_more_information: bool = Field(default=False)


class AnswerValidator:
    """Validates a generated answer against the evidence that produced it."""

    def __init__(self, client: LLMClient, strict: bool = True) -> None:
        self._client = client
        self._strict = strict

    async def validate(
        self, question: str, answer: str, evidence: list[EvidenceItem]
    ) -> tuple[str, ValidationReport]:
        """Audit an answer and return the corrected text plus its report."""
        if not answer.strip():
            return answer, ValidationReport(grounding_score=0.0, needs_more_information=True)

        cleaned, invalid_markers = strip_invalid_markers(answer, len(evidence))

        if not evidence:
            return cleaned, ValidationReport(
                grounding_score=0.0,
                needs_more_information=True,
                uncertainty_notes=[
                    "No evidence was retrieved, so this answer is not grounded in any source."
                ],
            )

        if not self._strict:
            return cleaned, _marker_only_report(cleaned, invalid_markers)

        try:
            audit = await self._client.structured(
                "citation/claim_validation",
                _AuditResult,
                temperature=0.0,
                question=question,
                answer=cleaned,
                evidence=_render_evidence(evidence),
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the marker audit
            log_event(logger, "validation.audit", "Claim audit failed",
                      level=30, error=str(exc))
            return cleaned, _marker_only_report(cleaned, invalid_markers, audit_failed=True)

        corrected = remove_statements(cleaned, audit.removed_statements)
        if audit.removed_statements:
            corrected = f"{corrected}\n\n{_UNCITED_NOTICE}"

        report = ValidationReport(
            claims=[
                ClaimValidation(
                    claim=item.claim,
                    verdict=item.verdict,
                    supporting_evidence_ids=item.supporting_evidence_ids,
                    explanation=item.explanation,
                )
                for item in audit.claims
            ],
            removed_statements=audit.removed_statements,
            uncertainty_notes=audit.uncertainty_notes,
            grounding_score=max(0.0, min(1.0, audit.grounding_score)),
            needs_more_information=audit.needs_more_information,
        )

        log_event(
            logger,
            "validation.audit",
            "Answer audited",
            grounding_score=report.grounding_score,
            unsupported=report.unsupported_count,
            removed=len(report.removed_statements),
        )
        return corrected, report


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def strip_invalid_markers(answer: str, evidence_count: int) -> tuple[str, int]:
    """Remove citation markers that point outside the evidence list."""
    removed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal removed
        index = int(match.group(1))
        if 1 <= index <= evidence_count:
            return match.group(0)
        removed += 1
        return ""

    cleaned = _MARKER_RE.sub(replace, answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), removed


def factual_sentences(answer: str) -> list[str]:
    """Return the sentences long enough to state a fact, skipping headings."""
    sentences: list[str] = []
    for block in answer.split("\n"):
        stripped = block.strip()
        if not stripped or _HEADING_RE.match(stripped):
            continue
        for sentence in _SENTENCE_RE.split(stripped):
            candidate = sentence.strip()
            if len(candidate.split()) >= _MIN_FACTUAL_WORDS:
                sentences.append(candidate)
    return sentences


def uncited_sentences(answer: str) -> list[str]:
    """Return factual looking sentences that carry no citation marker."""
    return [s for s in factual_sentences(answer) if not _MARKER_RE.search(s)]


def remove_statements(answer: str, statements: list[str]) -> str:
    """Delete specific sentences from the answer, preserving its structure."""
    if not statements:
        return answer

    result = answer
    for statement in statements:
        needle = statement.strip()
        if len(needle) < 12:
            continue
        result = result.replace(needle, "")

    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _marker_only_report(
    answer: str, invalid_markers: int, audit_failed: bool = False
) -> ValidationReport:
    """Build a report from the deterministic checks alone.

    Numerator and denominator count the same sentences: the old version divided
    per-line uncited sentences by a whole-answer split that never fired on
    non-Latin text, reporting a grounding of 0 for a fully cited Persian answer.
    """
    sentences = factual_sentences(answer)
    uncited = [s for s in sentences if not _MARKER_RE.search(s)]
    grounding = 1.0 - (len(uncited) / len(sentences)) if sentences else 1.0

    notes: list[str] = []
    if audit_failed:
        notes.append(_AUDIT_FALLBACK_NOTE)
    if invalid_markers:
        notes.append(
            f"{invalid_markers} citation marker(s) referred to evidence that does not exist "
            "and were removed."
        )
    if uncited:
        notes.append(f"{len(uncited)} statement(s) carry no citation and should be treated as unverified.")

    return ValidationReport(
        claims=[
            ClaimValidation(
                claim=sentence,
                verdict=ClaimVerdict.UNSUPPORTED,
                explanation="No citation marker was attached to this statement.",
            )
            for sentence in uncited
        ],
        uncertainty_notes=notes,
        grounding_score=round(max(0.0, grounding), 3),
        # Missing markers are a citation-discipline problem, not proof that the
        # evidence was insufficient; only the claim-level audit can say that.
        needs_more_information=False,
    )


def _render_evidence(evidence: list[EvidenceItem]) -> str:
    return "\n\n".join(
        f"E{index} | id={item.id} | {item.document_title}, page {item.page_number}\n{item.content}"
        for index, item in enumerate(evidence, start=1)
    )
