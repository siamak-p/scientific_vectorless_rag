"""Citation construction, renumbering and bibliographic formatting.

The answer generator emits inline evidence markers such as ``[E3]``. This
module maps those markers back to the exact evidence item and document they
came from, renumbers them into a clean sequential reference list, and renders
them in APA, IEEE or BibTeX.
"""

from __future__ import annotations

import re
import unicodedata

from core.constants import UNAVAILABLE_METADATA, CitationFormat
from core.models import Citation, DocumentRecord, EvidenceItem
from observability.logger import get_logger, log_event

logger = get_logger("retrieval.citations")

_MARKER_RE = re.compile(r"\[\s*E\s*(\d{1,3})\s*\]", re.IGNORECASE)


def build_citations(
    answer: str,
    evidence: list[EvidenceItem],
    documents: dict[str, DocumentRecord],
) -> tuple[str, list[Citation]]:
    """Resolve inline evidence markers into a numbered reference list.

    Args:
        answer: Generated text containing ``[E<n>]`` markers.
        evidence: The evidence list in the exact order shown to the model.
        documents: Document records keyed by id, used for bibliographic fields.

    Returns:
        The answer with markers renumbered to ``[1]``, ``[2]``, ... and the
        matching citations. Reference numbers identify distinct papers; each
        inline marker additionally carries the exact physical PDF page, e.g.
        ``[2, p. 7]``. Markers pointing at non-existent evidence are removed,
        which is the first line of defence against fabricated references.
    """
    by_index = {index: item for index, item in enumerate(evidence, start=1)}
    document_order: dict[str, int] = {}
    seen_evidence: set[int] = set()
    citations: list[Citation] = []
    dropped = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal dropped
        source_index = int(match.group(1))
        item = by_index.get(source_index)
        if item is None:
            dropped += 1
            return ""

        document_key = item.document_id or item.document_title
        if document_key not in document_order:
            document_order[document_key] = len(document_order) + 1
        if source_index not in seen_evidence:
            seen_evidence.add(source_index)
            citations.append(_to_citation(item, documents.get(item.document_id)))
        return f"[{document_order[document_key]}, p. {item.page_number}]"

    rewritten = _MARKER_RE.sub(replace, answer)
    rewritten = _tidy(rewritten)

    if dropped:
        log_event(logger, "citations.build", "Removed markers with no evidence",
                  level=30, dropped=dropped)

    return rewritten, citations


def _to_citation(item: EvidenceItem, document: DocumentRecord | None) -> Citation:
    """Build a citation from one evidence item and its source document."""
    metadata = document.metadata if document else None
    return Citation(
        evidence_id=item.id,
        document_id=item.document_id,
        paper_title=(metadata.title if metadata and metadata.title else item.document_title),
        authors=list(metadata.authors) if metadata else [],
        doi=metadata.doi if metadata else "",
        url=metadata.url if metadata else "",
        journal=metadata.journal if metadata else "",
        conference=metadata.conference if metadata else "",
        publisher=metadata.publisher if metadata else "",
        publication_year=metadata.publication_year if metadata else None,
        volume=metadata.volume if metadata else "",
        issue=metadata.issue if metadata else "",
        pages=metadata.pages if metadata else "",
        section=item.section,
        subsection=item.subsection,
        pdf_page_number=item.page_number,
        supporting_quote=item.quote,
        figure_number=item.figure_number,
        table_number=item.table_number,
        confidence_score=item.confidence_score,
    )


def _tidy(text: str) -> str:
    """Clean up whitespace left behind by removed markers."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_citation(citation: Citation, style: CitationFormat, index: int = 1) -> str:
    """Render one citation in the requested style."""
    if style is CitationFormat.APA:
        return _format_apa(citation)
    if style is CitationFormat.IEEE:
        return _format_ieee(citation, index)
    return _format_bibtex(citation, index)


def format_bibliography(
    citations: list[Citation], style: CitationFormat
) -> str:
    """Render a full reference list, de-duplicated by document."""
    unique = _unique_by_document(citations)
    if not unique:
        return "No references were generated for this answer."

    if style is CitationFormat.BIBTEX:
        return "\n\n".join(
            _format_bibtex(citation, index) for index, citation in enumerate(unique, start=1)
        )

    lines = []
    for index, citation in enumerate(unique, start=1):
        rendered = format_citation(citation, style, index)
        lines.append(rendered if style is CitationFormat.IEEE else f"{index}. {rendered}")
    return "\n\n".join(lines)


def _unique_by_document(citations: list[Citation]) -> list[Citation]:
    seen: set[str] = set()
    unique: list[Citation] = []
    for citation in citations:
        key = citation.document_id or citation.paper_title
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return unique


def _format_apa(citation: Citation) -> str:
    authors = _apa_authors(citation.authors)
    year = citation.publication_year or "n.d."
    title = citation.paper_title or UNAVAILABLE_METADATA
    parts = [f"{authors} ({year}). {title}."]

    venue = citation.journal or citation.conference
    if venue:
        segment = f"*{venue}*"
        if citation.volume:
            segment += f", {citation.volume}"
            if citation.issue:
                segment += f"({citation.issue})"
        if citation.pages:
            segment += f", {citation.pages}"
        parts.append(segment + ".")
    elif citation.publisher:
        parts.append(f"{citation.publisher}.")
    else:
        parts.append(f"{UNAVAILABLE_METADATA}.")

    if citation.doi:
        parts.append(f"https://doi.org/{citation.doi}")
    elif citation.url:
        parts.append(citation.url)

    return " ".join(parts)


def _format_ieee(citation: Citation, index: int) -> str:
    authors = _ieee_authors(citation.authors)
    title = citation.paper_title or UNAVAILABLE_METADATA
    parts = [f'[{index}] {authors}, "{title},"']

    venue = citation.journal or citation.conference
    parts.append(f"*{venue}*," if venue else f"{UNAVAILABLE_METADATA},")

    if citation.volume:
        parts.append(f"vol. {citation.volume},")
    if citation.issue:
        parts.append(f"no. {citation.issue},")
    if citation.pages:
        parts.append(f"pp. {citation.pages},")
    parts.append(f"{citation.publication_year}." if citation.publication_year else "n.d.")

    if citation.doi:
        parts.append(f"doi: {citation.doi}.")

    return " ".join(parts)


def _format_bibtex(citation: Citation, index: int) -> str:
    entry_type = "article" if citation.journal else ("inproceedings" if citation.conference else "misc")
    key = _bibtex_key(citation, index)

    fields: list[tuple[str, str]] = [
        ("title", citation.paper_title or UNAVAILABLE_METADATA),
        ("author", " and ".join(citation.authors) if citation.authors else UNAVAILABLE_METADATA),
    ]
    if citation.publication_year:
        fields.append(("year", str(citation.publication_year)))
    if citation.journal:
        fields.append(("journal", citation.journal))
    if citation.conference:
        fields.append(("booktitle", citation.conference))
    if citation.publisher:
        fields.append(("publisher", citation.publisher))
    if citation.volume:
        fields.append(("volume", citation.volume))
    if citation.issue:
        fields.append(("number", citation.issue))
    if citation.pages:
        fields.append(("pages", citation.pages))
    if citation.doi:
        fields.append(("doi", citation.doi))
    if citation.url:
        fields.append(("url", citation.url))

    body = ",\n".join(f"  {name} = {{{_escape_bibtex(value)}}}" for name, value in fields)
    return f"@{entry_type}{{{key},\n{body}\n}}"


def _bibtex_key(citation: Citation, index: int) -> str:
    surname = ""
    if citation.authors:
        surname = citation.authors[0].split()[-1]
    surname = _ascii_slug(surname) or "anonymous"

    word = ""
    for token in citation.paper_title.split():
        cleaned = _ascii_slug(token)
        if len(cleaned) > 3:
            word = cleaned.lower()
            break

    year = citation.publication_year or index
    return f"{surname.lower()}{year}{word}"


def _apa_authors(authors: list[str]) -> str:
    if not authors:
        return UNAVAILABLE_METADATA
    formatted = [_apa_name(name) for name in authors[:20]]
    if len(formatted) == 1:
        return formatted[0]
    if len(authors) > 20:
        return ", ".join(formatted) + ", ... et al."
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def _apa_name(name: str) -> str:
    parts = name.strip().split()
    if len(parts) < 2:
        return name.strip()
    initials = " ".join(f"{part[0].upper()}." for part in parts[:-1] if part)
    return f"{parts[-1]}, {initials}"


def _ieee_authors(authors: list[str]) -> str:
    if not authors:
        return UNAVAILABLE_METADATA
    formatted = []
    for name in authors[:6]:
        parts = name.strip().split()
        if len(parts) < 2:
            formatted.append(name.strip())
            continue
        initials = ". ".join(part[0].upper() for part in parts[:-1]) + "."
        formatted.append(f"{initials} {parts[-1]}")
    if len(authors) > 6:
        formatted.append("et al.")
    return ", ".join(formatted)


def _escape_bibtex(value: str) -> str:
    return value.replace("{", "").replace("}", "").replace("\\", "")


def _ascii_slug(value: str) -> str:
    normalised = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalised if ch.isalnum() and ch.isascii())
