"""Content-addressed cache for processed documents.

Cache entries are keyed by the SHA-256 digest of the PDF, so the same paper is
never parsed, summarised or tree-built twice — even when it is uploaded into a
different chat or arrives from a different search provider.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from core.constants import CACHE_DIR, ensure_directories
from core.models import DocumentMetadata, DocumentTree, ParsedDocument
from observability.logger import get_logger, log_event

logger = get_logger("documents.cache")

_SCHEMA_VERSION = 2


class DocumentCache:
    """A small JSON cache on the local filesystem."""

    def __init__(self, root: Path | None = None) -> None:
        ensure_directories()
        self._root = root or CACHE_DIR
        self._root.mkdir(parents=True, exist_ok=True)

    # -- paths -----------------------------------------------------------

    def _entry_dir(self, file_hash: str) -> Path:
        directory = self._root / file_hash[:2] / file_hash
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def pdf_path(self, file_hash: str) -> Path:
        """Return the canonical cached location of a PDF."""
        return self._entry_dir(file_hash) / "document.pdf"

    # -- generic read/write ---------------------------------------------

    def _read(self, file_hash: str, name: str) -> dict | None:
        path = self._entry_dir(file_hash) / f"{name}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log_event(logger, "cache.read", f"Discarding unreadable cache entry {name}",
                      level=30, file_hash=file_hash, error=str(exc))
            path.unlink(missing_ok=True)
            return None

        if payload.get("schema_version") != _SCHEMA_VERSION:
            path.unlink(missing_ok=True)
            return None
        return payload.get("data")

    def _write(self, file_hash: str, name: str, data: dict) -> None:
        path = self._entry_dir(file_hash) / f"{name}.json"
        temp = path.with_suffix(".tmp")
        try:
            temp.write_text(
                json.dumps({"schema_version": _SCHEMA_VERSION, "data": data}, default=str),
                encoding="utf-8",
            )
            os.replace(temp, path)
        except OSError as exc:
            log_event(logger, "cache.write", f"Could not persist cache entry {name}",
                      level=30, file_hash=file_hash, error=str(exc))

    # -- typed accessors -------------------------------------------------

    def get_parsed(self, file_hash: str) -> ParsedDocument | None:
        """Return the cached parse result for a PDF."""
        raw = self._read(file_hash, "parsed")
        if raw is None:
            return None
        try:
            return ParsedDocument.model_validate(raw)
        except ValueError:
            return None

    def save_parsed(self, parsed: ParsedDocument) -> None:
        """Cache the parse result for a PDF."""
        self._write(parsed.file_hash, "parsed", parsed.model_dump(mode="json"))

    def get_metadata(self, file_hash: str) -> DocumentMetadata | None:
        """Return cached bibliographic metadata for a PDF."""
        raw = self._read(file_hash, "metadata")
        if raw is None:
            return None
        try:
            return DocumentMetadata.model_validate(raw)
        except ValueError:
            return None

    def save_metadata(self, file_hash: str, metadata: DocumentMetadata) -> None:
        """Cache bibliographic metadata for a PDF."""
        self._write(file_hash, "metadata", metadata.model_dump(mode="json"))

    def get_tree(self, file_hash: str) -> DocumentTree | None:
        """Return the cached PageIndex tree for a PDF."""
        raw = self._read(file_hash, "tree")
        if raw is None:
            return None
        try:
            return DocumentTree.model_validate(raw)
        except ValueError:
            return None

    def save_tree(self, file_hash: str, tree: DocumentTree) -> None:
        """Cache the PageIndex tree for a PDF."""
        self._write(file_hash, "tree", tree.model_dump(mode="json"))

    def store_pdf(self, file_hash: str, source: Path) -> Path:
        """Copy a PDF into the cache and return its canonical path."""
        target = self.pdf_path(file_hash)
        if not target.exists():
            shutil.copy2(source, target)
        return target

    def store_bytes(self, file_hash: str, data: bytes) -> Path:
        """Write PDF bytes into the cache and return the canonical path."""
        target = self.pdf_path(file_hash)
        if not target.exists():
            target.write_bytes(data)
        return target

    def has_pdf(self, file_hash: str) -> bool:
        """Whether a PDF with this digest is already cached."""
        return self.pdf_path(file_hash).is_file()

    def clear(self, file_hash: str) -> None:
        """Remove every cached artefact for one document."""
        shutil.rmtree(self._entry_dir(file_hash), ignore_errors=True)

    def stats(self) -> dict[str, int]:
        """Return cache size information for the settings page."""
        files = list(self._root.rglob("*"))
        return {
            "documents": sum(1 for path in files if path.name == "document.pdf"),
            "bytes": sum(path.stat().st_size for path in files if path.is_file()),
        }

    def purge(self) -> None:
        """Delete the entire cache."""
        shutil.rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)
        log_event(logger, "cache.purge", "Document cache cleared")


document_cache = DocumentCache()
