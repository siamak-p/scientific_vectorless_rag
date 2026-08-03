"""Settings page: providers, models, generation, retrieval and research."""

from __future__ import annotations

import streamlit as st

from config.settings import save_settings
from core.constants import (
    AnswerMode,
    CitationFormat,
    LLMProviderType,
    SearchProviderType,
)
from core.exceptions import ScientificRAGError
from core.models import AppSettings
from documents.cache import document_cache
from llm.factory import list_models
from ui.runtime import run_async


def render_settings(settings: AppSettings) -> None:
    """Render the settings page."""
    header = st.columns([0.8, 0.2])
    header[0].markdown("## Settings")
    if header[1].button("Back to chat", width="stretch"):
        st.session_state["page"] = "chat"
        st.rerun()

    tabs = st.tabs(["Model", "Retrieval", "Search", "Research", "Storage"])

    with tabs[0]:
        _render_model(settings)
    with tabs[1]:
        _render_retrieval(settings)
    with tabs[2]:
        _render_search(settings)
    with tabs[3]:
        _render_research(settings)
    with tabs[4]:
        _render_storage()


def _render_model(settings: AppSettings) -> None:
    """Configure the provider, credentials and generation parameters."""
    st.markdown("#### Provider")

    providers = list(LLMProviderType)
    current = settings.active_provider or providers[0].value
    index = next((i for i, p in enumerate(providers) if p.value == current), 0)

    provider = st.selectbox(
        "Provider",
        options=providers,
        index=index,
        format_func=lambda item: item.label,
    )
    config = settings.provider(provider)

    if provider.requires_api_key:
        config.api_key = st.text_input(
            "API key",
            value=config.api_key,
            type="password",
            help="Stored locally with restricted permissions. Environment variables take precedence.",
        )
    else:
        config.base_url = st.text_input(
            "Server URL",
            value=config.effective_base_url(),
            help="The address of your local model server.",
        )

    models = _discover_models(provider)
    if models:
        model_ids = [model.id for model in models]
        selected_index = model_ids.index(config.selected_model) if config.selected_model in model_ids else 0
        config.selected_model = st.selectbox(
            "Model",
            options=model_ids,
            index=selected_index,
            help="Loaded live from the provider whenever it is reachable.",
        )
    else:
        config.selected_model = st.text_input("Model", value=config.selected_model)

    if st.button("Refresh model list"):
        st.session_state.pop(f"models_{provider.value}", None)
        st.rerun()

    st.markdown("#### Generation")
    settings.llm.temperature = st.slider(
        "Temperature", 0.0, 2.0, float(settings.llm.temperature), 0.05,
        help="Lower values keep answers close to the evidence.",
    )
    settings.llm.max_tokens = st.number_input(
        "Maximum output tokens", 256, 32_768, int(settings.llm.max_tokens), 256
    )
    settings.llm.top_p = st.slider("Top-p", 0.0, 1.0, float(settings.llm.top_p), 0.05)
    settings.llm.streaming = st.checkbox("Stream responses", value=settings.llm.streaming)
    settings.llm.system_prompt = st.text_area(
        "Additional system instructions",
        value=settings.llm.system_prompt,
        help="Appended to the built-in research assistant instructions.",
    )

    settings.active_provider = provider.value
    _save_button(settings, "model")


def _render_retrieval(settings: AppSettings) -> None:
    """Configure the navigation and evidence budgets."""
    retrieval = settings.retrieval
    st.markdown("#### Navigation")
    retrieval.retrieval_depth = st.slider(
        "Tree depth", 1, 5, retrieval.retrieval_depth,
        help="How many levels of the document hierarchy the navigator may descend.",
    )
    retrieval.max_pages_per_query = st.slider(
        "Maximum pages read per question", 4, 60, retrieval.max_pages_per_query,
        help="The reading budget shared across every selected paper.",
    )
    retrieval.max_evidence_items = st.slider(
        "Maximum evidence items", 4, 60, retrieval.max_evidence_items
    )
    retrieval.toc_check_pages = st.slider(
        "Pages scanned for a table of contents", 2, 20, retrieval.toc_check_pages
    )
    _save_button(settings, "retrieval")


def _render_search(settings: AppSettings) -> None:
    """Choose the literature sources and their credentials."""
    retrieval = settings.retrieval
    st.markdown("#### Sources")

    scientific_databases = [p for p in SearchProviderType if p is not SearchProviderType.TAVILY]
    current_selection = [p for p in retrieval.search_providers if p in scientific_databases]
    retrieval.search_providers = st.multiselect(
        "Scientific databases",
        options=scientific_databases,
        default=current_selection,
        format_func=lambda item: item.label,
        help=(
            "Available to the model when \u201cSearch scientific databases\u201d is enabled and "
            "the papers already indexed in the chat do not cover the question. Tavily "
            "web search (below) is used alongside these when needed and a key is set."
        ),
    )

    retrieval.max_search_papers = st.slider(
        "Maximum papers to download per question", 1, 20, retrieval.max_search_papers,
        help=(
            "A strict upper bound, not a download target. The model chooses whether new "
            "papers are needed and requests the smallest adequate number up to this cap."
        ),
    )
    retrieval.download_pdfs = st.checkbox(
        "Download open-access PDFs",
        value=retrieval.download_pdfs,
        help="Only files the provider marks as openly licensed are downloaded.",
    )

    st.markdown("#### Credentials")
    settings.tavily_api_key = st.text_input(
        "Tavily API key", value=settings.tavily_api_key, type="password",
        help=(
            "Optional. Tavily is a general web search used as a bonus source - set a key "
            "here and it is used automatically, with no need to select it above."
        ),
    )
    settings.semantic_scholar_api_key = st.text_input(
        "Semantic Scholar API key",
        value=settings.semantic_scholar_api_key,
        type="password",
        help="Optional. Raises the rate limit.",
    )
    settings.contact_email = st.text_input(
        "Contact email",
        value=settings.contact_email,
        help="Sent to OpenAlex, Crossref and PubMed as a courtesy, which raises the rate limit.",
    )
    _save_button(settings, "search")


def _render_research(settings: AppSettings) -> None:
    """Configure citation style, answer mode and grounding strictness."""
    research = settings.research
    research.citation_format = st.selectbox(
        "Citation style",
        options=list(CitationFormat),
        index=list(CitationFormat).index(research.citation_format),
        format_func=lambda item: item.value.upper(),
    )
    research.default_answer_mode = st.selectbox(
        "Default answer style",
        options=[AnswerMode.SHORT_ANSWER, AnswerMode.RESEARCH_REPORT],
        index=0 if research.default_answer_mode is AnswerMode.SHORT_ANSWER else 1,
        format_func=lambda item: item.value.replace("_", " ").title(),
    )
    research.deep_research_iterations = st.slider(
        "Deep research iterations", 1, 3, research.deep_research_iterations
    )
    research.strict_grounding = st.checkbox(
        "Verify every claim against the evidence",
        value=research.strict_grounding,
        help="Removes statements the retrieved evidence does not support. Costs one extra call.",
    )

    settings.interface.show_token_usage = st.checkbox(
        "Show token usage", value=settings.interface.show_token_usage
    )
    _save_button(settings, "research")


def _render_storage() -> None:
    """Show cache usage and allow it to be cleared."""
    stats = document_cache.stats()
    columns = st.columns(2)
    columns[0].metric("Cached documents", stats["documents"])
    columns[1].metric("Cache size", f"{stats['bytes'] / (1024 * 1024):.1f} MB")

    st.caption(
        "The cache stores parsed pages, metadata and document trees keyed by file "
        "content, so the same paper is never processed twice."
    )
    if st.button("Clear document cache", type="secondary"):
        document_cache.purge()
        st.success("The cache was cleared.")


def _discover_models(provider: LLMProviderType):
    """Load the provider's model list, caching it for this session."""
    key = f"models_{provider.value}"
    if key in st.session_state:
        return st.session_state[key]

    try:
        models = run_async(list_models(provider))
    except ScientificRAGError as exc:
        st.warning(exc.user_message)
        models = []
    except Exception as exc:  # noqa: BLE001 - discovery is best effort
        st.warning(f"The model list could not be loaded: {exc}")
        models = []

    st.session_state[key] = models
    return models


def _save_button(settings: AppSettings, section: str) -> None:
    """Persist the settings."""
    if st.button("Save settings", type="primary", key=f"save_{section}"):
        try:
            save_settings(settings)
            st.success("Settings saved.")
        except ScientificRAGError as exc:
            st.error(exc.user_message)
