"""The conversation panel: documents, message history, streaming and edits."""

from __future__ import annotations

import time

import streamlit as st

from core.constants import DocumentStatus, ExportFormat
from core.exceptions import ScientificRAGError
from core.models import AppSettings, Chat, ChatMessage, GeneratedAnswer, TokenUsage
from export.exporter import export_chat
from ui import services
from ui.components.trace import render_details

_EMPTY_STATE = """
### Ask a question about your papers

Upload PDFs below, or enable **Search scientific databases** in the sidebar to let the
assistant find open-access papers for you. Every answer is grounded in the pages it
actually read, and every claim is checked against that evidence before you see it.
"""

_STATUS_ICONS = {
    DocumentStatus.INDEXED: "✓",
    DocumentStatus.PROCESSING: "…",
    DocumentStatus.FAILED: "!",
    DocumentStatus.PENDING: "·",
}

_INFLIGHT = "inflight_turns"


def render_chat(chat: Chat, settings: AppSettings, options: dict) -> None:
    """Render the conversation and handle the next turn."""
    _render_header(chat)
    _render_documents(chat)

    messages = services.load_messages(chat.id)
    if not messages:
        st.markdown(_EMPTY_STATE)

    running = _inflight().get(chat.id)
    last_user_id = _last_user_message_id(messages)
    editing_id = st.session_state.get("editing_message") if running is None else None
    last_message_id = messages[-1].id if messages else None

    total_usage = TokenUsage()
    for message in messages:
        with st.chat_message(message.role):
            if message.role == "user" and message.id == last_user_id and message.id == editing_id:
                _render_message_editor(chat, settings, options, message)
                continue

            st.markdown(message.content)
            answer = _restore_answer(message.payload)
            if message.role == "assistant" and answer is not None and answer.elapsed_seconds > 0:
                st.caption(f"⏱ {answer.elapsed_seconds:0.1f}s")
            if message.role == "user" and message.id == last_user_id and running is None:
                if st.button("✎ Edit", key=f"edit_{message.id}", help="Edit and resend this message"):
                    st.session_state["editing_message"] = message.id
                    st.rerun()

            if answer is not None:
                total_usage = total_usage.add(answer.token_usage)
                with st.expander("Inspect this answer", expanded=False):
                    is_last = message.id == last_message_id
                    render_details(
                        chat.id,
                        answer,
                        key=message.id,
                        total_usage=total_usage if is_last else None,
                    )

    if running is not None:
        with st.chat_message("assistant"):
            status_slot = st.empty()
            answer_slot = st.empty()
            _finish_turn(
                chat,
                running["prompt"],
                running["future"],
                running["bridge"],
                answer_slot,
                status_slot,
                running["started_at"],
            )
        return

    prompt = st.chat_input("Ask about the papers in this conversation")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        _start_turn(chat, settings, options, prompt.strip())


def _inflight() -> dict:
    """Turns submitted to the background loop, keyed by chat id.

    Kept in session state (not a local variable) so a run survives the user
    navigating to another page and back: the workflow keeps executing on the
    shared event loop regardless of what the script does next, and whichever
    rerun finds it here resumes streaming it instead of losing the result.
    """
    return st.session_state.setdefault(_INFLIGHT, {})


def _last_user_message_id(messages: list[ChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.role == "user":
            return message.id
    return None


def _start_turn(chat: Chat, settings: AppSettings, options: dict, prompt: str) -> None:
    """Persist the user's message and submit the turn without blocking."""
    if not prompt:
        return
    services.save_message(chat.id, "user", prompt)
    future, bridge = services.start_turn(
        chat,
        settings,
        prompt,
        mode=options["mode"],
        explainability=options["explainability"],
        auto_search=options["auto_search"],
        deep_research=options["deep_research"],
    )
    _inflight()[chat.id] = {
        "future": future,
        "bridge": bridge,
        "prompt": prompt,
        "started_at": time.monotonic(),
    }
    st.rerun()


def _finish_turn(chat: Chat, prompt: str, future, bridge, answer_slot, status_slot, started_at: float) -> None:
    """Consume a submitted turn - resuming it cleanly even if it was interrupted."""
    try:
        answer = services.consume_turn(bridge, future, answer_slot, status_slot, started_at)
    except ScientificRAGError as exc:
        status_slot.empty()
        answer_slot.error(exc.user_message)
        services.save_message(chat.id, "assistant", f"**Error.** {exc.user_message}")
        _inflight().pop(chat.id, None)
        st.rerun()
        return
    except Exception as exc:  # noqa: BLE001 - never leave the UI without a reply
        status_slot.empty()
        answer_slot.error("The request could not be completed.")
        services.save_message(chat.id, "assistant", f"**Error.** {exc}")
        _inflight().pop(chat.id, None)
        st.rerun()
        return

    answer_slot.markdown(answer.answer)
    services.save_message(chat.id, "assistant", answer.answer, _serialise(answer))
    services.maybe_title_chat(chat, prompt)
    services.refresh_memory(chat)
    _inflight().pop(chat.id, None)
    st.rerun()


def _render_message_editor(chat: Chat, settings: AppSettings, options: dict, message: ChatMessage) -> None:
    """Let the user rewrite their last message and resend it."""
    new_text = st.text_area(
        "Edit your message",
        value=message.content,
        key=f"editor_{message.id}",
        label_visibility="collapsed",
    )
    columns = st.columns([0.14, 0.14, 0.72])
    if columns[0].button("Send", key=f"resend_{message.id}", type="primary"):
        edited = new_text.strip()
        if edited:
            services.delete_messages_from(chat.id, message.id)
            st.session_state.pop("editing_message", None)
            _start_turn(chat, settings, options, edited)
    if columns[1].button("Cancel", key=f"cancel_edit_{message.id}"):
        st.session_state.pop("editing_message", None)
        st.rerun()


def _render_documents(chat: Chat) -> None:
    """Upload, list and remove the PDFs belonging to this conversation."""
    documents = services.list_documents(chat.id)
    label = f"📎 Documents in this chat ({len(documents)})"
    with st.expander(label, expanded=not documents):
        st.caption("Documents uploaded here belong to this conversation only.")

        uploads = st.file_uploader(
            "Add PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{chat.id}",
            label_visibility="collapsed",
        )

        if uploads:
            processed = st.session_state.setdefault("processed_uploads", set())
            new_uploads = [
                upload
                for upload in uploads
                if f"{chat.id}:{upload.name}:{upload.size}" not in processed
            ]
            if new_uploads:
                label = (
                    f"Indexing {new_uploads[0].name}"
                    if len(new_uploads) == 1
                    else f"Indexing {len(new_uploads)} documents"
                )
                with st.spinner(label):
                    outcomes = services.upload_documents(
                        chat, [(upload.name, upload.getvalue()) for upload in new_uploads]
                    )
                for upload, (_, _, error) in zip(new_uploads, outcomes):
                    if error:
                        st.error(error)
                    processed.add(f"{chat.id}:{upload.name}:{upload.size}")
                st.rerun()

        if not documents:
            st.caption("No documents yet.")
            return

        for document in documents:
            icon = _STATUS_ICONS.get(document.status, "·")
            columns = st.columns([0.85, 0.15])
            with columns[0]:
                st.markdown(f"{icon} **{document.label()[:60]}**")
                detail = f"{document.page_count} pages · {document.source.value}"
                st.caption(document.error_message or detail)
            with columns[1]:
                if st.button("🗑", key=f"doc_del_{document.id}", help="Remove from this chat"):
                    services.remove_document(chat, document.id)
                    st.rerun()


def _render_header(chat: Chat) -> None:
    """Render the title bar with the export controls."""
    columns = st.columns([0.62, 0.13, 0.13, 0.12])
    columns[0].markdown(f"### {chat.title}")

    if columns[1].button("Export MD", help="Export this conversation as Markdown"):
        _export(chat, ExportFormat.MARKDOWN)
    if columns[2].button("Export PDF", help="Export this conversation as PDF"):
        _export(chat, ExportFormat.PDF)
    if columns[3].button("Clear", help="Delete every message in this conversation"):
        services.clear_messages(chat.id)
        st.rerun()


def _export(chat: Chat, fmt: ExportFormat) -> None:
    """Write the conversation to disk and offer it for download."""
    try:
        path = export_chat(chat, services.load_messages(chat.id), fmt)
    except ScientificRAGError as exc:
        st.error(exc.user_message)
        return

    st.success(f"Exported to {path}")
    st.download_button(
        "Download export",
        data=path.read_bytes(),
        file_name=path.name,
        mime="application/pdf" if fmt is ExportFormat.PDF else "text/markdown",
    )


def _serialise(answer: GeneratedAnswer) -> dict:
    """Store the full trace alongside the message so it survives a reload."""
    return {"answer": answer.model_dump(mode="json")}


def _restore_answer(payload: dict) -> GeneratedAnswer | None:
    """Rebuild a stored answer, ignoring payloads written by older versions."""
    raw = (payload or {}).get("answer")
    if not raw:
        return None
    try:
        return GeneratedAnswer.model_validate(raw)
    except ValueError:
        return None
