"""Document ingestion pipeline.

One entry point turns raw bytes into a queryable document: validate, store by
content hash, parse, extract metadata, build the PageIndex tree, and attach
that tree to the chat's forest.

Everything expensive is keyed by the SHA-256 of the file, so re-adding the
same paper — in this chat or any other — reuses the cached parse, metadata and
tree instead of paying for them again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.constants import (
    DocumentSource,
    DocumentStatus,
    DOCUMENT_BUILD_CONCURRENCY,
    UPLOAD_DIR,
    ensure_directories,
)
from core.exceptions import DocumentError
from core.models import (
    ChatIndex,
    DocumentMetadata,
    DocumentRecord,
    DocumentTree,
    ParsedDocument,
    new_id,
)
from documents.cache import DocumentCache, document_cache
from documents.metadata_extractor import MetadataExtractor
from documents.parser import extract_figures, extract_references, extract_tables, infer_headings
from documents.pdf_loader import PDFLoader, hash_bytes, safe_filename, validate_upload
from llm.client import LLMClient
from memory.storage import (
    ChatIndexRepository,
    DocumentRepository,
    chat_index_repository,
    document_repository,
)
from observability.logger import get_logger, log_event
from retrieval.pageindex.tree_builder import TreeBuilder

logger = get_logger("documents.processor")


class ProcessingResult:
    """Outcome of ingesting one document."""

    def __init__(
        self,
        document: DocumentRecord,
        reused: bool,
        tree: DocumentTree | None,
        error: DocumentError | None = None,
    ) -> None:
        self.document = document
        self.reused = reused
        self.tree = tree
        self.error = error

    @property
    def succeeded(self) -> bool:
        """Whether the document is ready to be queried."""
        return self.document.status is DocumentStatus.INDEXED


class DocumentProcessor:
    """Ingests PDFs into a chat's knowledge base."""

    def __init__(
        self,
        client: LLMClient,
        cache: DocumentCache = document_cache,
        documents: DocumentRepository = document_repository,
        indexes: ChatIndexRepository = chat_index_repository,
    ) -> None:
        self._client = client
        self._cache = cache
        self._documents = documents
        self._indexes = indexes
        self._metadata_extractor = MetadataExtractor(client)
        self._tree_builder = TreeBuilder(client)

    async def process_upload(
        self,
        chat_id: str,
        filename: str,
        data: bytes,
        source: DocumentSource = DocumentSource.UPLOADED,
        known_metadata: DocumentMetadata | None = None,
    ) -> ProcessingResult:
        """Validate, store and index one uploaded PDF."""
        results = await self.process_uploads(chat_id, [(filename, data, source, known_metadata)])
        result = results[0]
        if result.error is not None:
            raise result.error
        return result

    async def process_uploads(
        self,
        chat_id: str,
        uploads: list[tuple[str, bytes, DocumentSource, DocumentMetadata | None]],
    ) -> list[ProcessingResult]:
        """Validate, store and index several PDFs for one chat.

        Registering each paper (dedup check, persisting bytes, creating the
        record) is cheap and sequential, but the expensive stage - parsing,
        metadata extraction and PageIndex tree construction - runs for every
        new paper concurrently. The chat's forest is then updated a single
        time with every new tree attached together, instead of once per
        paper, so N papers cost one index load/save instead of N. This is
        the shared path for auto-searched papers, user uploads, and a turn
        that has both at once.
        """
        if not uploads:
            return []

        results: list[ProcessingResult | None] = [None] * len(uploads)
        pending: list[tuple[int, DocumentRecord, DocumentMetadata | None]] = []

        for position, (filename, data, source, known_metadata) in enumerate(uploads):
            validate_upload(filename, data)
            file_hash = hash_bytes(data)

            existing = await self._documents.find_by_hash(chat_id, file_hash)
            if existing is not None and existing.status is DocumentStatus.INDEXED:
                log_event(logger, "process.skip", "Document already present in this chat",
                          chat_id=chat_id, filename=filename)
                results[position] = ProcessingResult(
                    existing, reused=True, tree=self._cache.get_tree(file_hash)
                )
                continue

            path = await asyncio.to_thread(self._persist, file_hash, filename, data)
            record = DocumentRecord(
                id=new_id(),
                chat_id=chat_id,
                filename=safe_filename(filename),
                file_path=str(path),
                file_hash=file_hash,
                source=source,
                status=DocumentStatus.PROCESSING,
                metadata=known_metadata or DocumentMetadata(),
            )
            await self._documents.add(record)
            pending.append((position, record, known_metadata))

        if pending:
            semaphore = asyncio.Semaphore(DOCUMENT_BUILD_CONCURRENCY)

            async def build_one(
                record: DocumentRecord, known_metadata: DocumentMetadata | None
            ) -> tuple[DocumentTree, int]:
                async with semaphore:
                    return await self._build(record, known_metadata)

            outcomes = await asyncio.gather(
                *(build_one(record, known_metadata) for _, record, known_metadata in pending),
                return_exceptions=True,
            )

            built: list[tuple[DocumentRecord, DocumentTree, int]] = []
            for (position, record, _), outcome in zip(pending, outcomes):
                if isinstance(outcome, BaseException):
                    error = (
                        outcome
                        if isinstance(outcome, DocumentError)
                        else DocumentError(
                            f"'{record.filename}' could not be processed.", details=str(outcome)
                        )
                    )
                    results[position] = ProcessingResult(
                        record, reused=False, tree=None, error=error
                    )
                    continue
                tree, page_count = outcome
                built.append((record, tree, page_count))
                results[position] = ProcessingResult(record, reused=False, tree=tree)

            if built:
                index = await self._indexes.load(chat_id)
                for record, tree, _ in built:
                    index.attach(tree)
                await self._indexes.save(index)

                for record, tree, page_count in built:
                    record.status = DocumentStatus.INDEXED
                    await self._documents.update_status(
                        record.id,
                        DocumentStatus.INDEXED,
                        page_count=page_count,
                        metadata=record.metadata,
                    )
                    log_event(logger, "process.complete", "Document indexed",
                              chat_id=chat_id, document=record.label(), pages=page_count)

        return [result for result in results if result is not None]

    async def _build(
        self, record: DocumentRecord, known_metadata: DocumentMetadata | None
    ) -> tuple[DocumentTree, int]:
        """Parse, extract metadata and build the PageIndex tree for one document.

        Does not touch the chat's forest - callers attach the returned tree
        themselves, once, after every paper in the batch has been built.
        """
        try:
            parsed = await self._parse(record)
            record.page_count = parsed.page_count

            metadata = await self._metadata(record, parsed, known_metadata)
            record.metadata = metadata

            tree = await self._tree(record, parsed, metadata)
            return tree, parsed.page_count

        except DocumentError as exc:
            await self._fail(record, exc.user_message)
            raise
        except Exception as exc:  # noqa: BLE001 - normalised for the UI
            await self._fail(record, "The document could not be processed.")
            raise DocumentError(
                f"'{record.filename}' could not be processed.", details=str(exc)
            ) from exc

    async def remove(self, chat_id: str, document_id: str) -> None:
        """Detach a document from a chat and prune its subtree.

        The cached parse is intentionally kept, so the same paper can be
        re-added instantly later.
        """
        index = await self._indexes.load(chat_id)
        index.detach(document_id)
        await self._indexes.save(index)
        await self._documents.delete(document_id)
        log_event(logger, "process.remove", "Document removed from chat",
                  chat_id=chat_id, document_id=document_id)

    async def rebuild_index(self, chat_id: str) -> ChatIndex:
        """Rebuild a chat's forest from its indexed documents and cached trees."""
        index = ChatIndex(chat_id=chat_id)
        for document in await self._documents.list_indexed(chat_id):
            tree = self._cache.get_tree(document.file_hash)
            if tree is not None:
                index.attach(tree.model_copy(deep=True, update={"document_id": document.id}))
        await self._indexes.save(index)
        return index

    # -- stages ----------------------------------------------------------

    def _persist(self, file_hash: str, filename: str, data: bytes) -> Path:
        """Write the PDF into the cache and mirror it under the uploads folder."""
        ensure_directories()
        cached = self._cache.store_bytes(file_hash, data)

        readable = UPLOAD_DIR / f"{file_hash[:12]}_{safe_filename(filename)}"
        if not readable.exists():
            readable.write_bytes(data)
        return cached

    async def _parse(self, record: DocumentRecord) -> ParsedDocument:
        """Return the parsed document, reusing the cache when possible."""
        cached = self._cache.get_parsed(record.file_hash)
        if cached is not None:
            log_event(logger, "process.cache_hit", "Reusing cached parse for this file",
                      chat_id=record.chat_id, document=record.filename, file_hash=record.file_hash)
            return cached.model_copy(update={"document_id": record.id})

        parsed = await asyncio.to_thread(self._parse_sync, record)
        self._cache.save_parsed(parsed)
        return parsed

    def _parse_sync(self, record: DocumentRecord) -> ParsedDocument:
        """Blocking PDF extraction, run in a worker thread."""
        loader = PDFLoader(record.file_path)
        pages = loader.load_all_pages()
        outline = loader.embedded_outline()

        return ParsedDocument(
            document_id=record.id,
            file_hash=record.file_hash,
            page_count=len(pages),
            pages=pages,
            outline=outline,
            figures=extract_figures(pages),
            tables=extract_tables(pages),
            references=extract_references(pages),
        )

    async def _metadata(
        self,
        record: DocumentRecord,
        parsed: ParsedDocument,
        known_metadata: DocumentMetadata | None,
    ) -> DocumentMetadata:
        """Return bibliographic metadata, reusing the cache when possible."""
        cached = self._cache.get_metadata(record.file_hash)
        if cached is not None and cached.title:
            log_event(logger, "process.cache_hit", "Reusing cached metadata for this file",
                      chat_id=record.chat_id, document=record.filename, file_hash=record.file_hash)
            return cached

        metadata = await self._metadata_extractor.extract(
            parsed.pages, existing=known_metadata or record.metadata
        )
        if not metadata.title:
            metadata.title = Path(record.filename).stem.replace("_", " ")

        self._cache.save_metadata(record.file_hash, metadata)
        return metadata

    async def _tree(
        self,
        record: DocumentRecord,
        parsed: ParsedDocument,
        metadata: DocumentMetadata,
    ) -> DocumentTree:
        """Return the PageIndex tree, reusing the cache when possible."""
        title = metadata.display_title() or record.filename

        cached = self._cache.get_tree(record.file_hash)
        if cached is not None:
            log_event(logger, "process.cache_hit", "Reusing cached PageIndex tree for this file - not rebuilding",
                      chat_id=record.chat_id, document=record.filename, file_hash=record.file_hash)
            return _rebind(cached, record.id, title)

        outline = parsed.outline or infer_headings(parsed.pages)
        tree = await self._tree_builder.build(
            document_id=record.id,
            document_title=title,
            file_hash=record.file_hash,
            pages=parsed.pages,
            embedded_outline=outline,
        )
        self._cache.save_tree(record.file_hash, tree)
        return tree

    async def _fail(self, record: DocumentRecord, message: str) -> None:
        """Mark a document as failed without losing the reason."""
        record.status = DocumentStatus.FAILED
        record.error_message = message
        await self._documents.update_status(
            record.id, DocumentStatus.FAILED, error_message=message
        )
        log_event(logger, "process.failed", "Document processing failed",
                  level=40, document=record.filename, reason=message)


def _rebind(tree: DocumentTree, document_id: str, title: str) -> DocumentTree:
    """Re-point a cached tree at the document record of the current chat."""
    clone = tree.model_copy(deep=True)
    clone.document_id = document_id
    clone.document_title = title
    clone.root.title = title
    for node in clone.root.iter_nodes():
        node.document_id = document_id
    return clone
