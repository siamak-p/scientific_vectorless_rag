"""Legal open-access PDF retrieval.

Only links the provider itself advertises as open access are followed, and
the response must be a real PDF. Paywalled or HTML landing pages are skipped
rather than scraped.
"""

from __future__ import annotations

import httpx

from core.constants import (
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
    MAX_UPLOAD_SIZE_BYTES,
    PDF_MAGIC_BYTES,
)
from core.exceptions import PaperDownloadError
from documents.cache import document_cache
from documents.pdf_loader import hash_bytes, safe_filename
from observability.logger import get_logger, log_event
from scientific_search.base import SearchResult

logger = get_logger("scientific_search.downloader")


class PaperDownloader:
    """Downloads open-access PDFs into the content-addressed cache."""

    def __init__(self, timeout: float = HTTP_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout

    async def download(self, result: SearchResult) -> tuple[str, bytes]:
        """Download one paper.

        Returns:
            The SHA-256 of the file and its bytes.

        Raises:
            PaperDownloadError: If the paper is not openly downloadable.
        """
        title = result.metadata.display_title()

        if not result.pdf_url or not result.is_open_access:
            raise PaperDownloadError(
                f"No open-access PDF is available for '{title}'.",
                details="Only openly licensed full texts are downloaded.",
            )

        data = await self._fetch(result.pdf_url, title)
        file_hash = hash_bytes(data)

        if not document_cache.has_pdf(file_hash):
            document_cache.store_bytes(file_hash, data)

        log_event(logger, "download.complete", "Paper downloaded",
                  title=title[:120], bytes=len(data), file_hash=file_hash[:12])
        return file_hash, data

    def filename_for(self, result: SearchResult) -> str:
        """Build a stable, filesystem-safe filename for a downloaded paper."""
        title = result.metadata.display_title()[:80] or result.external_id or "paper"
        return safe_filename(f"{title}.pdf")

    async def _fetch(self, url: str, title: str) -> bytes:
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=True
            ) as client:
                response = await client.get(url, headers={"User-Agent": HTTP_USER_AGENT})
                response.raise_for_status()
                data = response.content
        except httpx.HTTPStatusError as exc:
            raise PaperDownloadError(
                f"'{title}' could not be downloaded.",
                details=f"The server returned HTTP {exc.response.status_code}.",
            ) from exc
        except httpx.HTTPError as exc:
            raise PaperDownloadError(
                f"'{title}' could not be downloaded.",
                details="The download source could not be reached.",
            ) from exc

        if not data.startswith(PDF_MAGIC_BYTES):
            raise PaperDownloadError(
                f"'{title}' is not available as a PDF.",
                details="The link returned a landing page instead of a full text.",
            )
        if len(data) > MAX_UPLOAD_SIZE_BYTES:
            raise PaperDownloadError(
                f"'{title}' is too large to process.",
                details=f"The file exceeds {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
            )
        return data
