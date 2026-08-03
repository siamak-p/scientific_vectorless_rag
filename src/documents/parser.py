"""Deterministic structural parsing of extracted PDF text.

Everything in this module is rule based and costs no tokens. It supplies the
figures, tables, references and heading candidates that the tree builder and
the evidence extractor rely on.
"""

from __future__ import annotations

import re

from core.models import FigureRef, OutlineEntry, PageContent, TableRef

_FIGURE_RE = re.compile(
    r"^\s*(fig(?:ure)?\.?\s*(?:S?\d+[a-z]?))\s*[.:\u2014-]?\s*(.{0,300})",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(
    r"^\s*(tab(?:le)?\.?\s*(?:S?[\dIVX]+[a-z]?))\s*[.:\u2014-]?\s*(.{0,300})",
    re.IGNORECASE,
)
_REFERENCE_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|literature cited|works cited)\s*$", re.IGNORECASE
)
_REFERENCE_ENTRY_RE = re.compile(r"^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}\.)\s+(.{20,})")
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][^.]{2,80})\s*$"
)

_CANONICAL_SECTIONS = (
    "abstract",
    "introduction",
    "background",
    "related work",
    "materials and methods",
    "methods",
    "methodology",
    "approach",
    "model",
    "experimental setup",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "ablation study",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "acknowledgements",
    "acknowledgments",
    "references",
    "appendix",
)

_MAX_HEADING_WORDS = 12


def extract_figures(pages: list[PageContent]) -> list[FigureRef]:
    """Return every figure caption detected across the document."""
    figures: list[FigureRef] = []
    seen: set[str] = set()

    for page in pages:
        for line in page.text.splitlines():
            match = _FIGURE_RE.match(line)
            if not match:
                continue
            label = _normalise_label(match.group(1), "Figure")
            if label in seen:
                continue
            seen.add(label)
            figures.append(
                FigureRef(
                    label=label,
                    caption=match.group(2).strip(),
                    page_number=page.page_number,
                )
            )
    return figures


def extract_tables(pages: list[PageContent]) -> list[TableRef]:
    """Return every table caption detected across the document."""
    tables: list[TableRef] = []
    seen: set[str] = set()

    for page in pages:
        for line in page.text.splitlines():
            match = _TABLE_RE.match(line)
            if not match:
                continue
            label = _normalise_label(match.group(1), "Table")
            if label in seen:
                continue
            seen.add(label)
            tables.append(
                TableRef(
                    label=label,
                    caption=match.group(2).strip(),
                    page_number=page.page_number,
                )
            )
    return tables


def extract_references(pages: list[PageContent]) -> list[str]:
    """Return the bibliography entries found after the references heading."""
    started = False
    entries: list[str] = []
    buffer: list[str] = []

    for page in pages:
        for line in page.text.splitlines():
            stripped = line.strip()
            if not started:
                if _REFERENCE_HEADING_RE.match(stripped):
                    started = True
                continue

            if not stripped:
                continue

            if _REFERENCE_ENTRY_RE.match(stripped):
                if buffer:
                    entries.append(" ".join(buffer).strip())
                buffer = [stripped]
            elif buffer:
                buffer.append(stripped)

    if buffer:
        entries.append(" ".join(buffer).strip())

    return [entry for entry in entries if len(entry) > 25][:200]


def infer_headings(pages: list[PageContent]) -> list[OutlineEntry]:
    """Infer a heading outline from the raw text without calling an LLM.

    Two signals are used: explicit decimal numbering such as ``3.2 Training``
    and short standalone lines matching canonical scientific section names.
    """
    entries: list[OutlineEntry] = []
    seen: set[tuple[str, int]] = set()

    for page in pages:
        for line in page.text.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > 120:
                continue

            numbered = _NUMBERED_HEADING_RE.match(stripped)
            if numbered:
                number, title = numbered.group(1), numbered.group(2).strip()
                level = number.count(".") + 1
                key = (title.lower(), page.page_number)
                if key not in seen:
                    seen.add(key)
                    entries.append(
                        OutlineEntry(title=title, level=min(level, 3), page_start=page.page_number)
                    )
                continue

            canonical = _canonical_heading(stripped)
            if canonical:
                key = (canonical.lower(), page.page_number)
                if key not in seen:
                    seen.add(key)
                    entries.append(
                        OutlineEntry(title=canonical, level=1, page_start=page.page_number)
                    )

    entries.sort(key=lambda e: (e.page_start, e.level))
    return entries


def section_hint(outline: list[OutlineEntry], page_number: int) -> str:
    """Return the title of the section that a page most likely belongs to."""
    current = ""
    for entry in outline:
        if entry.page_start <= page_number:
            current = entry.title
        else:
            break
    return current


def build_page_digest(pages: list[PageContent], char_budget: int) -> str:
    """Return page-tagged text truncated to a character budget.

    The ``Page N`` markers let the LLM anchor each heading to a physical page.
    """
    parts: list[str] = []
    used = 0

    for page in pages:
        block = f"\nPage {page.page_number}\n{page.text}\n"
        if used + len(block) > char_budget:
            remaining = max(0, char_budget - used)
            if remaining > 200:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used += len(block)

    return "".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_label(raw: str, kind: str) -> str:
    """Normalise ``fig. 3a`` style labels into ``Figure 3a``."""
    number = re.sub(r"^(fig(ure)?|tab(le)?)\.?\s*", "", raw.strip(), flags=re.IGNORECASE)
    return f"{kind} {number.strip()}"


def _canonical_heading(line: str) -> str:
    """Return the canonical section name when a line is a known heading."""
    cleaned = re.sub(r"^[\dIVX]+[.)]?\s*", "", line).strip(" .:-")
    if not cleaned or len(cleaned.split()) > _MAX_HEADING_WORDS:
        return ""

    lowered = cleaned.lower()
    for canonical in _CANONICAL_SECTIONS:
        if lowered == canonical:
            return cleaned.title() if cleaned.isupper() else cleaned
    return ""
