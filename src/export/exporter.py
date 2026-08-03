"""Export of conversations and research reports.

Markdown is generated first and used as the single source of truth; the PDF
renderer consumes the same structure, so both formats always agree.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from core.constants import EXPORT_DIR, CitationFormat, ExportFormat, ensure_directories
from core.exceptions import ExportError
from core.models import Chat, ChatMessage, GeneratedAnswer
from observability.logger import get_logger, log_event
from retrieval.citations import format_bibliography

logger = get_logger("export.exporter")

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`]+)`")


def export_chat(chat: Chat, messages: list[ChatMessage], fmt: ExportFormat) -> Path:
    """Export a full conversation and return the file that was written."""
    markdown = chat_to_markdown(chat, messages)
    return _write(_slug(chat.title), markdown, fmt)


def export_answer(chat: Chat, answer: GeneratedAnswer, fmt: ExportFormat) -> Path:
    """Export a single research answer with its citations and trace."""
    markdown = answer_to_markdown(chat, answer)
    return _write(f"{_slug(chat.title)}-report", markdown, fmt)


def export_references(
    chat: Chat, answer: GeneratedAnswer, style: CitationFormat = CitationFormat.BIBTEX
) -> Path:
    """Export just the bibliography, by default as BibTeX."""
    body = format_bibliography(answer.citations, style)
    suffix = "bib" if style is CitationFormat.BIBTEX else "txt"
    return _write_text(f"{_slug(chat.title)}-references", body, suffix)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def chat_to_markdown(chat: Chat, messages: list[ChatMessage]) -> str:
    """Render a conversation as Markdown."""
    lines = [
        f"# {chat.title}",
        "",
        f"*Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}*  ",
        f"*Model: {chat.provider or 'unknown'} / {chat.model or 'unknown'}*",
        "",
        "---",
        "",
    ]
    for message in messages:
        speaker = "User" if message.role == "user" else "Assistant"
        lines.extend([f"## {speaker}", "", message.content.strip(), "", "---", ""])
    return "\n".join(lines)


def answer_to_markdown(chat: Chat, answer: GeneratedAnswer) -> str:
    """Render one answer, its evidence and its audit trail as Markdown."""
    lines = [
        f"# {chat.title}",
        "",
        f"*Generated {answer.created_at.strftime('%Y-%m-%d %H:%M')} — mode: {answer.mode.value}*",
        "",
        answer.answer.strip(),
        "",
    ]

    if answer.evidence_items:
        lines.extend(["## Evidence", ""])
        for index, item in enumerate(answer.evidence_items, start=1):
            lines.append(
                f"**E{index}** — {item.document_title}, page {item.page_number}"
                + (f", {item.section}" if item.section else "")
            )
            lines.extend(["", item.content.strip(), ""])
            if item.quote:
                lines.extend([f"> {item.quote.strip()}", ""])

    if answer.validation is not None:
        report = answer.validation
        lines.extend(
            [
                "## Verification",
                "",
                f"- Grounding score: {report.grounding_score:.2f}",
                f"- Unsupported claims removed: {len(report.removed_statements)}",
                "",
            ]
        )
        for note in report.uncertainty_notes:
            lines.append(f"- {note}")
        lines.append("")

    if answer.research_trace is not None and answer.research_trace.iterations:
        lines.extend(["## Research trace", ""])
        for iteration in answer.research_trace.iterations:
            lines.append(f"**Iteration {iteration.index}** — {iteration.focus}")
            lines.append(f"- New papers: {len(iteration.new_document_ids)}")
            lines.append(f"- Evidence found: {iteration.evidence_count}")
            if iteration.knowledge_gaps:
                lines.append("- Gaps: " + "; ".join(iteration.knowledge_gaps))
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write(stem: str, markdown: str, fmt: ExportFormat) -> Path:
    """Write the markdown in the requested format."""
    if fmt is ExportFormat.MARKDOWN:
        return _write_text(stem, markdown, "md")
    if fmt is ExportFormat.PDF:
        return _write_pdf(stem, markdown)
    raise ExportError(f"The export format '{fmt.value}' is not supported.")


def _write_text(stem: str, body: str, suffix: str) -> Path:
    """Write a plain text artefact into the export directory."""
    ensure_directories()
    path = _unique_path(stem, suffix)
    try:
        path.write_text(body, encoding="utf-8")
    except OSError as exc:
        raise ExportError("The export file could not be written.", details=str(exc)) from exc
    log_event(logger, "export.write", "Export written", path=str(path))
    return path


def _write_pdf(stem: str, markdown: str) -> Path:
    """Render the markdown into a paginated PDF."""
    ensure_directories()
    path = _unique_path(stem, "pdf")

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ResearchBody",
        parent=styles["BodyText"],
        alignment=TA_JUSTIFY,
        fontSize=10.5,
        leading=15,
        spaceAfter=6,
    )
    quote = ParagraphStyle(
        "ResearchQuote", parent=body, leftIndent=18, textColor="#444444", fontName="Helvetica-Oblique"
    )

    story = []
    for block in markdown.split("\n"):
        story.extend(_render_block(block, styles, body, quote))

    try:
        document = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=stem,
        )
        document.build(story or [Paragraph("This export is empty.", body)])
    except Exception as exc:  # noqa: BLE001 - reportlab raises broadly
        raise ExportError("The PDF export could not be produced.", details=str(exc)) from exc

    log_event(logger, "export.write", "Export written", path=str(path))
    return path


def _render_block(block: str, styles, body, quote) -> list:
    """Convert one Markdown line into flowables."""
    text = block.rstrip()
    if not text:
        return [Spacer(1, 6)]

    if text.strip() == "---":
        return [Spacer(1, 10)]
    if text.strip() == "\f":
        return [PageBreak()]

    heading = _HEADING_RE.match(text)
    if heading:
        level = min(len(heading.group(1)), 4)
        return [Spacer(1, 8), Paragraph(_inline(heading.group(2)), styles[f"Heading{level}"])]

    if text.lstrip().startswith(">"):
        return [Paragraph(_inline(text.lstrip("> ").strip()), quote)]

    bullet = _BULLET_RE.match(text)
    if bullet:
        return [Paragraph(f"&bull; {_inline(bullet.group(1))}", body)]

    return [Paragraph(_inline(text), body)]


def _inline(text: str) -> str:
    """Translate inline Markdown into the mini-HTML reportlab understands."""
    escaped = html.escape(text)
    escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)
    escaped = _ITALIC_RE.sub(r"<i>\1</i>", escaped)
    return _CODE_RE.sub(r"<font face='Courier'>\1</font>", escaped)


def _unique_path(stem: str, suffix: str) -> Path:
    """Return a non-colliding path inside the export directory."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return EXPORT_DIR / f"{stem}-{stamp}.{suffix}"


def _slug(value: str) -> str:
    """Return a filesystem-safe slug for a title."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return (cleaned or "scientific-rag")[:60]
