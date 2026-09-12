"""Session state and application service helpers for the interface.

The interface never talks to a repository twice for the same thing in one
rerun, and never keeps conversation content in ``st.session_state``: messages
live in the database, so switching chats or reloading the browser cannot lose
them.
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_settings, save_settings
from core.constants import AnswerMode, DocumentSource
from core.exceptions import ScientificRAGError
from core.models import AppSettings, Chat, ChatMessage, DocumentRecord, GeneratedAnswer
from graph.context import RunContext
from graph.graph_builder import WorkflowRunner
from llm.client import LLMClient
from llm.factory import resolve_provider
from memory.chat_memory import memory_manager
from memory.database import init_database
from memory.storage import (
    chat_index_repository,
    chat_repository,
    document_repository,
    message_repository,
)
from documents.processor import DocumentProcessor
from ui.runtime import StreamBridge, run_async, submit

_ACTIVE_CHAT = "active_chat_id"
_PENDING = "pending_query"


def bootstrap() -> AppSettings:
    """Prepare the database and return the current settings.

    The schema is created on first launch, so the application is usable
    immediately after installation with no manual migration step.
    """
    if not st.session_state.get("_database_ready"):
        run_async(init_database())
        st.session_state["_database_ready"] = True
    return get_settings()


def save_research_options(settings: AppSettings) -> None:
    """Persist the sidebar's research toggles so they survive a fresh session.

    Best effort: a transient disk error here should not block answering, so
    failures are swallowed after being surfaced once as a warning.
    """
    try:
        save_settings(settings)
    except ScientificRAGError as exc:
        st.warning(exc.user_message)


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------

def list_chats(search: str = "") -> list[Chat]:
    """Return conversations, optionally filtered by a search term."""
    if search.strip():
        return run_async(chat_repository.search(search))
    return run_async(chat_repository.list())


def active_chat(settings: AppSettings) -> Chat:
    """Return the selected conversation, creating one when none exists."""
    chat_id = st.session_state.get(_ACTIVE_CHAT)

    if chat_id:
        try:
            return run_async(chat_repository.get(chat_id))
        except ScientificRAGError:
            st.session_state.pop(_ACTIVE_CHAT, None)

    existing = list_chats()
    chat = existing[0] if existing else create_chat(settings)
    st.session_state[_ACTIVE_CHAT] = chat.id
    return chat


def create_chat(settings: AppSettings) -> Chat:
    """Create a conversation that follows the current model configuration.

    Provider and model are intentionally left unset so the chat always uses
    whatever is chosen on the Settings page, even if that changes later.
    """
    chat = Chat(
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        top_p=settings.llm.top_p,
        system_prompt=settings.llm.system_prompt,
    )
    run_async(chat_repository.create(chat))
    st.session_state[_ACTIVE_CHAT] = chat.id
    return chat


def select_chat(chat_id: str) -> None:
    """Switch the active conversation."""
    st.session_state[_ACTIVE_CHAT] = chat_id


def rename_chat(chat_id: str, title: str) -> None:
    """Rename a conversation."""
    run_async(chat_repository.rename(chat_id, title))


def delete_chat(chat_id: str) -> None:
    """Delete a conversation and clear it from the session."""
    run_async(chat_repository.delete(chat_id))
    if st.session_state.get(_ACTIVE_CHAT) == chat_id:
        st.session_state.pop(_ACTIVE_CHAT, None)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def load_messages(chat_id: str) -> list[ChatMessage]:
    """Return the persisted messages of a conversation."""
    return run_async(message_repository.list(chat_id))


def save_message(chat_id: str, role: str, content: str, payload: dict | None = None) -> None:
    """Persist one message."""
    run_async(message_repository.add(chat_id, role, content, payload))


def clear_messages(chat_id: str) -> None:
    """Remove every message from a conversation."""
    run_async(message_repository.clear(chat_id))


def delete_messages_from(chat_id: str, message_id: str) -> None:
    """Delete one message and everything after it, so a turn can be redone."""
    run_async(message_repository.delete_from(chat_id, message_id))


def pending_query() -> str:
    """Return and clear the query queued by the previous rerun."""
    return st.session_state.pop(_PENDING, "")


def queue_query(query: str) -> None:
    """Queue a query to be answered on the next rerun."""
    st.session_state[_PENDING] = query


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def list_documents(chat_id: str) -> list[DocumentRecord]:
    """Return the documents attached to a conversation."""
    return run_async(document_repository.list(chat_id))


def upload_document(chat: Chat, filename: str, data: bytes) -> DocumentRecord:
    """Ingest an uploaded PDF into a conversation."""
    processor = DocumentProcessor(LLMClient.for_chat(chat))
    result = run_async(
        processor.process_upload(chat.id, filename, data, source=DocumentSource.UPLOADED)
    )
    return result.document


def upload_documents(
    chat: Chat, uploads: list[tuple[str, bytes]]
) -> list[tuple[str, DocumentRecord | None, str | None]]:
    """Ingest several uploaded PDFs into a conversation at once.

    Every paper's tree is built concurrently and the chat's forest is
    updated once for the whole batch, instead of once per file - the same
    optimisation used when auto-search downloads several papers together.

    Returns one ``(filename, document, error_message)`` tuple per upload, in
    the same order, so a failure in one file does not hide the others.
    """
    processor = DocumentProcessor(LLMClient.for_chat(chat))
    items = [(filename, data, DocumentSource.UPLOADED, None) for filename, data in uploads]
    results = run_async(processor.process_uploads(chat.id, items))
    return [
        (
            filename,
            None if result.error else result.document,
            result.error.user_message if result.error else None,
        )
        for (filename, _), result in zip(uploads, results)
    ]


def remove_document(chat: Chat, document_id: str) -> None:
    """Detach a document from a conversation."""
    processor = DocumentProcessor(LLMClient.for_chat(chat))
    run_async(processor.remove(chat.id, document_id))


def load_index(chat_id: str):
    """Return the PageIndex forest of a conversation."""
    return run_async(chat_index_repository.load(chat_id))


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

def start_turn(
    chat: Chat,
    settings: AppSettings,
    query: str,
    mode: AnswerMode,
    explainability: bool,
    auto_search: bool,
    deep_research: bool,
):
    """Submit a turn to the background loop without blocking the script.

    The workflow keeps running on the shared event loop regardless of what
    the script does next, so a run started here survives the user navigating
    to another page and back: :func:`consume_turn` can be called again later
    to pick up exactly where it left off.
    """
    bridge = StreamBridge()
    context = RunContext.for_chat(
        chat, settings, on_token=bridge.on_token, on_stage=bridge.on_stage
    )
    runner = WorkflowRunner(chat, settings, context=context)
    future = submit(
        runner.run(
            query=query,
            mode=mode,
            explainability_enabled=explainability,
            auto_search_enabled=auto_search,
            deep_research_enabled=deep_research,
        )
    )
    return future, bridge


def consume_turn(
    bridge: StreamBridge, future, answer_slot, status_slot, started_at: float | None = None
) -> GeneratedAnswer:
    """Render tokens as they arrive and return the finished answer."""
    return bridge.consume(future, answer_slot, status_slot, started_at=started_at)


def refresh_memory(chat: Chat) -> None:
    """Update the chat's topic digest used for cross-chat recall."""
    run_async(memory_manager.refresh_summary(chat.id, chat.title, LLMClient.for_chat(chat)))


def maybe_title_chat(chat: Chat, first_message: str) -> str:
    """Give an untitled conversation a name based on its first message."""
    # Compared case-insensitively: a fallback title that differs only in case
    # would otherwise switch auto-titling off for that chat forever.
    if chat.title.strip().lower() not in {"", "new chat"}:
        return chat.title

    client = LLMClient.for_chat(chat)
    try:
        title = run_async(client.complete("answer/chat_title", message=first_message))
    except ScientificRAGError:
        title = first_message
    cleaned = " ".join(title.split()).strip("\"'.")[:60] or "New Chat"
    rename_chat(chat.id, cleaned)
    return cleaned


def _provider_type(settings: AppSettings):
    """Resolve the configured provider, falling back to the library default."""
    return resolve_provider(settings.active_provider or None)


def provider_setup_hint(settings: AppSettings) -> str:
    """Return what still has to be configured before a question can be answered."""
    try:
        provider = _provider_type(settings)
    except ScientificRAGError:
        return "No provider is selected yet. Open **Settings** to choose one."

    missing = settings.provider(provider).missing_requirement()
    if not missing:
        return ""
    return (
        f"The **{provider.label}** provider still needs its {missing}. "
        "Open **Settings** to complete the configuration."
    )


def effective_provider_model(chat: Chat, settings: AppSettings) -> tuple[str, str]:
    """Return the provider and model that will actually answer this chat.

    Every chat always follows the current Settings selection - there is no
    UI to pin a chat to a different model - so this always matches what
    :class:`LLMClient` resolves at answer time, and immediately reflects any
    change made on the Settings page.
    """
    del chat
    try:
        provider = _provider_type(settings)
    except ScientificRAGError:
        return "not configured", "not selected"
    model = settings.provider(provider).selected_model
    return provider.value, model or "not selected"
