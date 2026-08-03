"""PDF reading, validation and text extraction built on PyMuPDF."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # PyMuPDF

from core.constants import MAX_UPLOAD_SIZE_BYTES, PDF_MAGIC_BYTES
from core.exceptions import (
    FileTooLargeError,
    InvalidPDFError,
    PDFExtractionError,
    UnsupportedFileError,
)
from core.models import OutlineEntry, PageContent
from observability.logger import get_logger, log_event

logger = get_logger("documents.pdf_loader")

_HASH_CHUNK = 1 << 20  # 1 MiB


def validate_upload(filename: str, data: bytes) -> None:
    """Validate an uploaded file before it is written to disk.

    Args:
        filename: The client supplied name, used only for messages.
        data: The raw bytes of the upload.

    Raises:
        UnsupportedFileError: The file is not a PDF.
        FileTooLargeError: The file exceeds the configured size limit.
        InvalidPDFError: The bytes do not start with the PDF magic header.
    """
    if not filename.lower().endswith(".pdf"):
        raise UnsupportedFileError(filename)

    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(filename, len(data), MAX_UPLOAD_SIZE_BYTES)

    if not data.startswith(PDF_MAGIC_BYTES):
        raise InvalidPDFError(filename, "The file does not contain a PDF header.")


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe name, stripping any directory components."""
    stem = Path(filename).name
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in stem).strip()
    return (cleaned or "document.pdf")[:180]


def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


class PDFLoader:
    """Reads a single PDF file from disk."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise InvalidPDFError(self.file_path.name, "The file no longer exists on disk.")
        self._hash: str | None = None

    # -- identity --------------------------------------------------------

    def file_hash(self) -> str:
        """Return the SHA-256 digest of the file, used as the cache key."""
        if self._hash is None:
            digest = hashlib.sha256()
            try:
                with self.file_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
                        digest.update(chunk)
            except OSError as exc:
                raise InvalidPDFError(self.file_path.name, f"Could not read the file: {exc}") from exc
            self._hash = digest.hexdigest()
        return self._hash

    # -- structure -------------------------------------------------------

    def page_count(self) -> int:
        """Return the number of pages in the document."""
        try:
            with fitz.open(self.file_path) as document:
                return document.page_count
        except Exception as exc:  # noqa: BLE001 - normalised for the UI
            raise InvalidPDFError(self.file_path.name, f"Could not open the document: {exc}") from exc

    def embedded_outline(self) -> list[OutlineEntry]:
        """Return the PDF's own bookmark outline, when it has one.

        Using the embedded outline avoids an LLM call entirely and is exact,
        which is the cheapest possible way to build a PageIndex tree.
        """
        try:
            with fitz.open(self.file_path) as document:
                raw = document.get_toc(simple=True) or []
        except Exception as exc:  # noqa: BLE001 - outline is optional
            log_event(
                logger,
                "pdf.outline",
                "Embedded outline unavailable",
                level=20,
                file=self.file_path.name,
                error=str(exc),
            )
            return []

        entries: list[OutlineEntry] = []
        for item in raw:
            if len(item) < 3:
                continue
            level, title, page = int(item[0]), str(item[1]).strip(), int(item[2])
            if not title or page < 1:
                continue
            entries.append(OutlineEntry(title=title, level=max(1, level), page_start=page))
        return entries

    # -- text ------------------------------------------------------------

    def load_pages(self, start_page: int = 1, end_page: int | None = None) -> list[PageContent]:
        """Extract text from an inclusive, 1-indexed page range.

        Raises:
            InvalidPDFError: The document could not be opened.
            PDFExtractionError: A page could not be decoded.
        """
        pages: list[PageContent] = []
        try:
            with fitz.open(self.file_path) as document:
                total = document.page_count
                first = max(0, start_page - 1)
                last = total - 1 if end_page is None else min(total - 1, end_page - 1)

                if first > last:
                    return []

                for index in range(first, last + 1):
                    try:
                        text = document.load_page(index).get_text("text").strip()
                    except Exception as exc:  # noqa: BLE001 - per page failure
                        raise PDFExtractionError(self.file_path.name, index + 1) from exc
                    pages.append(
                        PageContent(page_number=index + 1, text=text, char_count=len(text))
                    )
        except (PDFExtractionError, InvalidPDFError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalised for the UI
            raise InvalidPDFError(self.file_path.name, f"Could not process the document: {exc}") from exc

        return pages

    def load_all_pages(self) -> list[PageContent]:
        """Extract text from the whole document."""
        return self.load_pages()

    def is_scanned(self, pages: list[PageContent]) -> bool:
        """Heuristically detect a scanned PDF that carries no extractable text."""
        if not pages:
            return True
        total_chars = sum(page.char_count for page in pages)
        return total_chars < 100 * len(pages)
