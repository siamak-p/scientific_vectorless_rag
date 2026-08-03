"""Sidebar: conversations and research controls.

Document management lives in the chat panel itself (each conversation owns
its PDFs), so this module only handles the conversation list and the
research toggles.
"""

from __future__ import annotations

import streamlit as st

from core.constants import AnswerMode, SearchProviderType
from core.models import AppSettings, Chat
from ui import services

_MODE_LABELS = {
    AnswerMode.SHORT_ANSWER: "Short answer",
    AnswerMode.RESEARCH_REPORT: "Research report",
}


def render_sidebar(chat: Chat, settings: AppSettings) -> dict:
    """Render the sidebar and return the research options the user selected."""
    with st.sidebar:
        st.markdown("## Scientific RAG")
        st.caption("Vectorless retrieval over scientific papers")

        if st.button("New chat", width="stretch", type="primary"):
            services.create_chat(settings)
            st.rerun()

        _render_chat_list(chat)
        st.divider()
        options = _render_options(settings)
        st.divider()
        _render_footer(chat, settings)

    return options


def _render_chat_list(chat: Chat) -> None:
    """List conversations with search, inline rename and delete."""
    st.markdown("#### Conversations")
    search = st.text_input(
        "Search conversations",
        key="chat_search",
        placeholder="Search titles and messages",
        label_visibility="collapsed",
    )

    chats = services.list_chats(search)
    if not chats:
        st.caption("No conversations match this search.")
        return

    renaming = st.session_state.get("renaming_chat")

    for item in chats[:50]:
        is_active = item.id == chat.id
        columns = st.columns([0.78, 0.11, 0.11])

        if renaming == item.id:
            with columns[0]:
                new_title = st.text_input(
                    "New title",
                    value=item.title,
                    key=f"rename_input_{item.id}",
                    label_visibility="collapsed",
                )
            with columns[1]:
                if st.button("✔", key=f"rename_save_{item.id}", help="Save the new title"):
                    services.rename_chat(item.id, new_title.strip() or item.title)
                    st.session_state.pop("renaming_chat", None)
                    st.rerun()
            with columns[2]:
                if st.button("✕", key=f"rename_cancel_{item.id}", help="Cancel"):
                    st.session_state.pop("renaming_chat", None)
                    st.rerun()
            continue

        with columns[0]:
            if st.button(
                f"{'▸ ' if is_active else ''}{item.title}",
                key=f"chat_{item.id}",
                width="stretch",
                type="secondary",
                help=f"Last updated {item.updated_at:%Y-%m-%d %H:%M}",
            ):
                services.select_chat(item.id)
                st.rerun()

        with columns[1]:
            if st.button("✎", key=f"rename_{item.id}", help="Rename this conversation"):
                st.session_state["renaming_chat"] = item.id
                st.rerun()

        with columns[2]:
            if st.button("🗑", key=f"delete_{item.id}", help="Delete this conversation"):
                services.delete_chat(item.id)
                st.rerun()


def _render_options(settings: AppSettings) -> dict:
    """Render the three research toggles and the answer mode selector.

    Their values are persisted in `settings.research` (on disk), not just in
    `session_state`: whatever the user sets stays set across page
    navigation, new chats and even app restarts, until they change it
    themselves again. `session_state` keys give the widgets stable identity
    across reruns (and across the chat/Settings page swap, which unmounts
    them for a run); the disk save is what survives a fresh session.
    """
    st.markdown("#### Research options")
    research = settings.research

    st.session_state.setdefault("opt_explainability", research.show_retrieval_reasoning)
    st.session_state.setdefault("opt_auto_search", research.auto_search_default)
    st.session_state.setdefault("opt_deep_research", research.deep_research_default)
    st.session_state.setdefault(
        "opt_mode",
        research.default_answer_mode
        if research.default_answer_mode in _MODE_LABELS
        else AnswerMode.SHORT_ANSWER,
    )

    explainability = st.checkbox(
        "Show retrieval reasoning",
        key="opt_explainability",
        help=(
            "Displays the navigation tree: which section of which paper was selected "
            "or rejected, why, with relevance and confidence scores and the pages read."
        ),
    )
    auto_search = st.checkbox(
        "Search scientific databases",
        key="opt_auto_search",
        help=(
            "Lets the model search arXiv, Semantic Scholar, OpenAlex, Crossref and "
            "PubMed only when the papers already in this chat do not cover the query. "
            "The download setting is a maximum, not a required number per question."
        ),
    )
    deep_research = st.checkbox(
        "Deep research",
        key="opt_deep_research",
        help=(
            "Runs an iterative loop: analyse what is missing, search for it, read the "
            "new papers, and repeat until the evidence is sufficient. Slower and uses "
            "considerably more tokens."
        ),
    )

    if auto_search or deep_research:
        missing = [
            provider
            for provider in settings.retrieval.search_providers
            if provider is SearchProviderType.TAVILY and not settings.tavily_api_key
        ]
        if missing:
            st.warning(
                "Tavily is selected under Settings → Search but has no API key, so it "
                "will be skipped. Add a key there to use it."
            )

    mode = st.radio(
        "Answer style",
        options=list(_MODE_LABELS),
        format_func=lambda option: _MODE_LABELS[option],
        key="opt_mode",
        horizontal=True,
        disabled=deep_research,
        help="Deep research always produces a full report.",
    )

    if (
        explainability != research.show_retrieval_reasoning
        or auto_search != research.auto_search_default
        or deep_research != research.deep_research_default
        or mode != research.default_answer_mode
    ):
        research.show_retrieval_reasoning = explainability
        research.auto_search_default = auto_search
        research.deep_research_default = deep_research
        research.default_answer_mode = mode
        services.save_research_options(settings)

    return {
        "explainability": explainability,
        "auto_search": auto_search or deep_research,
        "deep_research": deep_research,
        "mode": AnswerMode.DEEP_RESEARCH if deep_research else mode,
    }


def _render_footer(chat: Chat, settings: AppSettings) -> None:
    """Show the model that will answer this chat and a link to settings."""
    provider, model = services.effective_provider_model(chat, settings)
    st.caption(f"Provider: {provider}")
    st.caption(f"Model: {model}")
    if st.button("Settings", width="stretch"):
        st.session_state["page"] = "settings"
        st.rerun()

