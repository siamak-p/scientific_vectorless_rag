"""Database engine, schema and automatic initialisation.

SQLite is used deliberately: the application is a local research tool, so the
whole store is a single file under the application home and needs no server.
The schema is created on first use, which is what makes the app usable
immediately after installation with no manual migration step.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    event,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from core.constants import DATABASE_FILE, DocumentStatus, ensure_directories
from core.exceptions import DatabaseError
from core.models import utc_now
from observability.logger import get_logger, log_event

logger = get_logger("memory.database")


class Base(DeclarativeBase):
    """Declarative base for every persisted table."""


class ChatModel(Base):
    """A conversation thread. Owns its messages, documents and index."""

    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(300), default="New chat")
    provider: Mapped[str] = mapped_column(String(40), default="")
    model: Mapped[str] = mapped_column(String(160), default="")
    temperature: Mapped[float] = mapped_column(Float, default=0.2)
    max_tokens: Mapped[int] = mapped_column(Integer, default=4096)
    top_p: Mapped[float] = mapped_column(Float, default=1.0)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    messages: Mapped[list["MessageModel"]] = relationship(
        back_populates="chat", cascade="all, delete-orphan", passive_deletes=True
    )
    documents: Mapped[list["DocumentModel"]] = relationship(
        back_populates="chat", cascade="all, delete-orphan", passive_deletes=True
    )


class MessageModel(Base):
    """One turn of a conversation."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    chat_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chats.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    chat: Mapped[ChatModel] = relationship(back_populates="messages")


class DocumentModel(Base):
    """A PDF attached to one chat.

    ``file_hash`` is what makes reuse possible: the parsed content and tree are
    cached by hash, while this row stays scoped to a single chat.
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    chat_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chats.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(400))
    file_path: Mapped[str] = mapped_column(Text, default="")
    file_hash: Mapped[str] = mapped_column(String(64), index=True, default="")
    source: Mapped[str] = mapped_column(String(40), default="uploaded")
    status: Mapped[str] = mapped_column(String(40), default=DocumentStatus.PENDING.value)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str] = mapped_column(Text, default="")
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    chat: Mapped[ChatModel] = relationship(back_populates="documents")


class ChatIndexModel(Base):
    """The serialised PageIndex forest belonging to one chat."""

    __tablename__ = "chat_indexes"

    chat_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chats.id", ondelete="CASCADE"), primary_key=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ChatSummaryModel(Base):
    """Topic-level digest of a chat.

    This is the only content that crosses chat boundaries: it holds themes,
    never specific statements, which keeps per-chat memory isolated while
    still letting the assistant recall what has been worked on in general.
    """

    __tablename__ = "chat_summaries"

    chat_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chats.id", ondelete="CASCADE"), primary_key=True
    )
    title: Mapped[str] = mapped_column(String(300), default="")
    topics: Mapped[list[str]] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    is_stale: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


Index("ix_messages_chat_created", MessageModel.chat_id, MessageModel.created_at)
Index("ix_documents_chat_status", DocumentModel.chat_id, DocumentModel.status)


class Database:
    """Owns the async engine and guarantees the schema exists before first use."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(DATABASE_FILE)
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._ready = False
        self._lock = asyncio.Lock()

    @property
    def url(self) -> str:
        """SQLAlchemy URL for the configured SQLite file."""
        return f"sqlite+aiosqlite:///{self._db_path}"

    def engine(self) -> AsyncEngine:
        """Return the shared engine, creating it on first access."""
        if self._engine is None:
            ensure_directories()
            self._engine = create_async_engine(self.url, echo=False, future=True)
            _register_pragmas(self._engine)
            self._session_factory = async_sessionmaker(
                self._engine, expire_on_commit=False, class_=AsyncSession
            )
        return self._engine

    async def initialise(self) -> None:
        """Create every table if it does not already exist.

        Safe to call on every application start; concurrent callers are
        serialised so Streamlit reruns cannot race each other.
        """
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            try:
                async with self.engine().begin() as connection:
                    await connection.run_sync(Base.metadata.create_all)
            except Exception as exc:  # noqa: BLE001 - surfaced as a friendly error
                raise DatabaseError(
                    "The local database could not be prepared.",
                    details=str(exc),
                ) from exc
            self._ready = True
            log_event(logger, "database.ready", "Database initialised", path=self._db_path)

    async def session(self) -> AsyncSession:
        """Return a new session, initialising the schema first."""
        await self.initialise()
        assert self._session_factory is not None  # set by engine()
        return self._session_factory()

    async def dispose(self) -> None:
        """Close all pooled connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._ready = False


def _register_pragmas(engine: AsyncEngine) -> None:
    """Enable foreign keys and WAL so cascades and concurrent reads work."""

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001 - SQLAlchemy signature
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


database = Database()


async def init_database() -> None:
    """Prepare the database. Called once at application start."""
    await database.initialise()
