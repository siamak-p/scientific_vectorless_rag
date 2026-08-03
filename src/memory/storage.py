"""Repository layer.

All persistence goes through these repositories so the rest of the
application never writes SQL. Every query is scoped by ``chat_id``, which is
what enforces conversation isolation: one chat can never read another chat's
messages, documents or index.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, or_, select, update

from core.constants import DocumentSource, DocumentStatus
from core.exceptions import ChatNotFoundError, DatabaseError
from core.models import (
    Chat,
    ChatIndex,
    ChatMessage,
    ChatSummary,
    DocumentMetadata,
    DocumentRecord,
    new_id,
    utc_now,
)
from memory.database import (
    ChatIndexModel,
    ChatModel,
    ChatSummaryModel,
    Database,
    DocumentModel,
    MessageModel,
    database,
)
from observability.logger import get_logger, log_event

logger = get_logger("memory.storage")


class ChatRepository:
    """Create, rename, search and delete conversations."""

    def __init__(self, db: Database = database) -> None:
        self._db = db

    async def create(self, chat: Chat | None = None) -> Chat:
        """Create a new conversation."""
        record = chat or Chat()
        async with await self._db.session() as session:
            session.add(_chat_to_model(record))
            await _commit(session, "The chat could not be created.")
        log_event(logger, "chat.create", "Chat created", chat_id=record.id)
        return record

    async def get(self, chat_id: str) -> Chat:
        """Return one conversation."""
        async with await self._db.session() as session:
            model = await session.get(ChatModel, chat_id)
            if model is None:
                raise ChatNotFoundError(f"Chat '{chat_id}' no longer exists.")
            return _model_to_chat(model)

    async def list(self, limit: int = 200) -> list[Chat]:
        """Return conversations, most recently updated first."""
        async with await self._db.session() as session:
            result = await session.execute(
                select(ChatModel).order_by(ChatModel.updated_at.desc()).limit(limit)
            )
            return [_model_to_chat(model) for model in result.scalars()]

    async def search(self, term: str, limit: int = 50) -> list[Chat]:
        """Find conversations by title or message content."""
        needle = f"%{term.strip()}%"
        if not term.strip():
            return await self.list(limit)

        async with await self._db.session() as session:
            matching_ids = select(MessageModel.chat_id).where(MessageModel.content.ilike(needle))
            result = await session.execute(
                select(ChatModel)
                .where(or_(ChatModel.title.ilike(needle), ChatModel.id.in_(matching_ids)))
                .order_by(ChatModel.updated_at.desc())
                .limit(limit)
            )
            return [_model_to_chat(model) for model in result.scalars()]

    async def rename(self, chat_id: str, title: str) -> None:
        """Change a conversation title."""
        await self._update(chat_id, {"title": title.strip()[:300] or "New chat"})

    async def update_model_settings(
        self,
        chat_id: str,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> None:
        """Persist the per-chat generation settings."""
        await self._update(
            chat_id,
            {
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            },
        )

    async def touch(self, chat_id: str) -> None:
        """Bump the updated timestamp so the chat rises in the sidebar."""
        await self._update(chat_id, {})

    async def delete(self, chat_id: str) -> None:
        """Delete a conversation with all of its messages and documents."""
        async with await self._db.session() as session:
            await session.execute(delete(ChatModel).where(ChatModel.id == chat_id))
            await _commit(session, "The chat could not be deleted.")
        log_event(logger, "chat.delete", "Chat deleted", chat_id=chat_id)

    async def _update(self, chat_id: str, values: dict[str, Any]) -> None:
        async with await self._db.session() as session:
            result = await session.execute(
                update(ChatModel)
                .where(ChatModel.id == chat_id)
                .values(**values, updated_at=utc_now())
            )
            if result.rowcount == 0:
                raise ChatNotFoundError(f"Chat '{chat_id}' no longer exists.")
            await _commit(session, "The chat could not be updated.")


class MessageRepository:
    """Append and read conversation turns."""

    def __init__(self, db: Database = database) -> None:
        self._db = db

    async def add(
        self,
        chat_id: str,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Append one message to a conversation."""
        message = ChatMessage(
            id=new_id(), chat_id=chat_id, role=role, content=content, payload=payload or {}
        )
        async with await self._db.session() as session:
            session.add(
                MessageModel(
                    id=message.id,
                    chat_id=message.chat_id,
                    role=message.role,
                    content=message.content,
                    payload=message.payload,
                    created_at=message.created_at,
                )
            )
            await session.execute(
                update(ChatModel).where(ChatModel.id == chat_id).values(updated_at=utc_now())
            )
            await _commit(session, "The message could not be saved.")
        return message

    async def list(self, chat_id: str, limit: int | None = None) -> list[ChatMessage]:
        """Return a conversation's messages in chronological order."""
        async with await self._db.session() as session:
            query = (
                select(MessageModel)
                .where(MessageModel.chat_id == chat_id)
                .order_by(MessageModel.created_at.asc(), MessageModel.id.asc())
            )
            result = await session.execute(query)
            messages = [_model_to_message(model) for model in result.scalars()]
        return messages[-limit:] if limit else messages

    async def recent(self, chat_id: str, window: int) -> list[ChatMessage]:
        """Return the last ``window`` messages of a conversation."""
        return await self.list(chat_id, limit=window)

    async def count(self, chat_id: str) -> int:
        """Return how many messages a conversation holds."""
        async with await self._db.session() as session:
            result = await session.execute(
                select(func.count()).select_from(MessageModel).where(MessageModel.chat_id == chat_id)
            )
            return int(result.scalar_one())

    async def delete_from(self, chat_id: str, message_id: str) -> None:
        """Delete a message and everything after it, so a turn can be redone."""
        async with await self._db.session() as session:
            anchor = await session.get(MessageModel, message_id)
            if anchor is None:
                return
            await session.execute(
                delete(MessageModel).where(
                    MessageModel.chat_id == chat_id,
                    MessageModel.created_at >= anchor.created_at,
                )
            )
            await _commit(session, "The messages could not be removed.")

    async def clear(self, chat_id: str) -> None:
        """Remove every message from a conversation."""
        async with await self._db.session() as session:
            await session.execute(delete(MessageModel).where(MessageModel.chat_id == chat_id))
            await _commit(session, "The conversation could not be cleared.")


class DocumentRepository:
    """Track the PDFs attached to each chat."""

    def __init__(self, db: Database = database) -> None:
        self._db = db

    async def add(self, document: DocumentRecord) -> DocumentRecord:
        """Register a document against a chat."""
        async with await self._db.session() as session:
            session.add(_document_to_model(document))
            await _commit(session, "The document could not be registered.")
        return document

    async def list(self, chat_id: str) -> list[DocumentRecord]:
        """Return every document belonging to one chat."""
        async with await self._db.session() as session:
            result = await session.execute(
                select(DocumentModel)
                .where(DocumentModel.chat_id == chat_id)
                .order_by(DocumentModel.created_at.asc())
            )
            return [_model_to_document(model) for model in result.scalars()]

    async def list_indexed(self, chat_id: str) -> list[DocumentRecord]:
        """Return only the documents that are ready to be queried."""
        documents = await self.list(chat_id)
        return [doc for doc in documents if doc.status is DocumentStatus.INDEXED]

    async def get(self, document_id: str) -> DocumentRecord | None:
        """Return one document by id."""
        async with await self._db.session() as session:
            model = await session.get(DocumentModel, document_id)
            return _model_to_document(model) if model else None

    async def find_by_hash(self, chat_id: str, file_hash: str) -> DocumentRecord | None:
        """Return an existing document with the same content in this chat."""
        async with await self._db.session() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    DocumentModel.chat_id == chat_id,
                    DocumentModel.file_hash == file_hash,
                )
            )
            model = result.scalars().first()
            return _model_to_document(model) if model else None

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        page_count: int | None = None,
        error_message: str = "",
        metadata: DocumentMetadata | None = None,
    ) -> None:
        """Record processing progress for a document."""
        values: dict[str, Any] = {"status": status.value, "error_message": error_message}
        if page_count is not None:
            values["page_count"] = page_count
        if metadata is not None:
            values["doc_metadata"] = metadata.model_dump(mode="json")

        async with await self._db.session() as session:
            await session.execute(
                update(DocumentModel).where(DocumentModel.id == document_id).values(**values)
            )
            await _commit(session, "The document status could not be updated.")

    async def delete(self, document_id: str) -> None:
        """Detach a document from its chat.

        The cached parse stays on disk so re-adding the same paper, here or in
        another chat, remains instant.
        """
        async with await self._db.session() as session:
            await session.execute(delete(DocumentModel).where(DocumentModel.id == document_id))
            await _commit(session, "The document could not be removed.")


class ChatIndexRepository:
    """Persist the PageIndex forest of each chat."""

    def __init__(self, db: Database = database) -> None:
        self._db = db

    async def load(self, chat_id: str) -> ChatIndex:
        """Return a chat's index, or an empty one when none exists yet."""
        async with await self._db.session() as session:
            model = await session.get(ChatIndexModel, chat_id)
        if model is None or not model.payload:
            return ChatIndex(chat_id=chat_id)
        try:
            return ChatIndex.model_validate(model.payload)
        except Exception as exc:  # noqa: BLE001 - a corrupt index must not block the chat
            log_event(logger, "index.load", "Stored index was unreadable and reset",
                      level=30, chat_id=chat_id, error=str(exc))
            return ChatIndex(chat_id=chat_id)

    async def save(self, index: ChatIndex) -> None:
        """Persist a chat's index."""
        index.updated_at = utc_now()
        payload = index.model_dump(mode="json")
        async with await self._db.session() as session:
            model = await session.get(ChatIndexModel, index.chat_id)
            if model is None:
                session.add(ChatIndexModel(chat_id=index.chat_id, payload=payload))
            else:
                model.payload = payload
                model.updated_at = index.updated_at
            await _commit(session, "The document index could not be saved.")

    async def delete(self, chat_id: str) -> None:
        """Drop a chat's index."""
        async with await self._db.session() as session:
            await session.execute(delete(ChatIndexModel).where(ChatIndexModel.chat_id == chat_id))
            await _commit(session, "The document index could not be removed.")


class SummaryRepository:
    """Persist the topic-level digests used for cross-chat recall."""

    def __init__(self, db: Database = database) -> None:
        self._db = db

    async def get(self, chat_id: str) -> ChatSummary | None:
        """Return one chat's digest."""
        async with await self._db.session() as session:
            model = await session.get(ChatSummaryModel, chat_id)
            return _model_to_summary(model) if model else None

    async def save(self, summary: ChatSummary) -> None:
        """Store or refresh a chat's digest."""
        async with await self._db.session() as session:
            model = await session.get(ChatSummaryModel, summary.chat_id)
            if model is None:
                model = ChatSummaryModel(chat_id=summary.chat_id)
                session.add(model)
            model.title = summary.title
            model.topics = list(summary.topics)
            model.summary = summary.summary
            model.message_count = summary.message_count
            model.is_stale = False
            model.updated_at = utc_now()
            await _commit(session, "The chat summary could not be saved.")

    async def others(self, chat_id: str, limit: int = 40) -> list[ChatSummary]:
        """Return digests of every other chat, newest first."""
        async with await self._db.session() as session:
            result = await session.execute(
                select(ChatSummaryModel)
                .where(ChatSummaryModel.chat_id != chat_id)
                .order_by(ChatSummaryModel.updated_at.desc())
                .limit(limit)
            )
            return [_model_to_summary(model) for model in result.scalars()]

    async def mark_stale(self, chat_id: str) -> None:
        """Flag a digest as out of date after new messages arrive."""
        async with await self._db.session() as session:
            await session.execute(
                update(ChatSummaryModel)
                .where(ChatSummaryModel.chat_id == chat_id)
                .values(is_stale=True)
            )
            await _commit(session, "The chat summary could not be updated.")


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

async def _commit(session, message: str) -> None:  # noqa: ANN001 - AsyncSession
    """Commit, converting any driver failure into a friendly error."""
    try:
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        await session.rollback()
        raise DatabaseError(message, details=str(exc)) from exc


def _chat_to_model(chat: Chat) -> ChatModel:
    return ChatModel(
        id=chat.id,
        title=chat.title,
        provider=chat.provider,
        model=chat.model,
        temperature=chat.temperature,
        max_tokens=chat.max_tokens,
        top_p=chat.top_p,
        system_prompt=chat.system_prompt,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )


def _model_to_chat(model: ChatModel) -> Chat:
    return Chat(
        id=model.id,
        title=model.title,
        provider=model.provider,
        model=model.model,
        temperature=model.temperature,
        max_tokens=model.max_tokens,
        top_p=model.top_p,
        system_prompt=model.system_prompt,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _model_to_message(model: MessageModel) -> ChatMessage:
    return ChatMessage(
        id=model.id,
        chat_id=model.chat_id,
        role=model.role,
        content=model.content,
        payload=model.payload or {},
        created_at=model.created_at,
    )


def _document_to_model(document: DocumentRecord) -> DocumentModel:
    return DocumentModel(
        id=document.id,
        chat_id=document.chat_id,
        filename=document.filename,
        file_path=document.file_path,
        file_hash=document.file_hash,
        source=document.source.value,
        status=document.status.value,
        page_count=document.page_count,
        error_message=document.error_message,
        doc_metadata=document.metadata.model_dump(mode="json"),
        created_at=document.created_at,
    )


def _model_to_document(model: DocumentModel) -> DocumentRecord:
    return DocumentRecord(
        id=model.id,
        chat_id=model.chat_id,
        filename=model.filename,
        file_path=model.file_path,
        file_hash=model.file_hash,
        source=DocumentSource(model.source),
        status=DocumentStatus(model.status),
        page_count=model.page_count,
        error_message=model.error_message,
        metadata=DocumentMetadata.model_validate(model.doc_metadata or {}),
        created_at=model.created_at,
    )


def _model_to_summary(model: ChatSummaryModel) -> ChatSummary:
    return ChatSummary(
        chat_id=model.chat_id,
        title=model.title,
        topics=list(model.topics or []),
        summary=model.summary,
        message_count=model.message_count,
        updated_at=model.updated_at,
    )


chat_repository = ChatRepository()
message_repository = MessageRepository()
document_repository = DocumentRepository()
chat_index_repository = ChatIndexRepository()
summary_repository = SummaryRepository()
