"""Settings page: providers, models, generation, retrieval and research."""

from __future__ import annotations

import hashlib

import streamlit as st

from config.catalog import pricing_refresh_days
from config.pricing import price_key, pricing_store
from config.settings import save_settings
from core.constants import (
    RESEARCH_DOMAINS,
    AnswerMode,
    CitationFormat,
    LLMProviderType,
    SearchProviderType,
)
from core.exceptions import ScientificRAGError
from core.models import AppSettings, CustomSearchSource, ModelPrice, ProviderSettings
from documents.cache import document_cache
from llm.base import ModelCatalog
from llm.factory import fetch_models
from scientific_search.custom_source import EXAMPLE_SOURCE
from ui.runtime import run_async, submit

# Bumped by "Refresh model list" so an unchanged key is still re-discovered.
_REFRESH_NONCE = "model_refresh_nonce"
_PRICE_UPDATE_STARTED = "pricing_update_started"
_CUSTOM_SOURCES_DRAFT = "custom_sources_draft"


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
        key="settings_provider",
    )
    config = settings.provider(provider)

    if provider.requires_base_url:
        config.base_url = st.text_input(
            "Server URL",
            value=config.effective_base_url(),
            key=f"base_url_{provider.value}",
            placeholder="https://openrouter.ai/api/v1",
            help=(
                "The address of your model server, including the API version path."
                if provider.is_local
                else "The OpenAI-compatible endpoint, including the /v1 path."
            ),
        ).strip()

    if provider.accepts_api_key:
        config.api_key = st.text_input(
            "API key",
            value=config.api_key,
            type="password",
            key=f"api_key_{provider.value}",
            help=(
                "Stored locally with restricted permissions. Environment variables "
                "take precedence."
                + ("" if provider.requires_api_key else " Leave empty if the server needs none.")
            ),
        ).strip()

    columns = st.columns([0.75, 0.25], vertical_alignment="bottom")
    if columns[1].button("Refresh model list", width="stretch"):
        st.session_state[_REFRESH_NONCE] = st.session_state.get(_REFRESH_NONCE, 0) + 1

    fingerprint = _credential_fingerprint(provider, config)
    catalog = _discover_models(provider, config, fingerprint)
    models = catalog.models
    if models:
        model_ids = [model.id for model in models]
        selected_index = model_ids.index(config.selected_model) if config.selected_model in model_ids else 0
        config.selected_model = columns[0].selectbox(
            "Model",
            options=model_ids,
            index=selected_index,
            key=f"model_{provider.value}_{fingerprint}",
            help="Loaded live from the provider whenever it is reachable.",
        )
    else:
        config.selected_model = columns[0].text_input(
            "Model", value=config.selected_model, key=f"model_text_{provider.value}"
        )

    if provider.requires_base_url and not config.base_url:
        st.caption("Enter the server URL above to load the model list.")
    elif provider.requires_api_key and not config.api_key:
        st.caption("Enter the API key above to load this provider's model list.")
    elif catalog.error:
        st.warning(catalog.error)
    elif catalog.live:
        st.caption(f"{len(models)} models loaded from {provider.label}.")

    _render_pricing(settings, provider, config.selected_model)

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

    scientific_databases = [p for p in SearchProviderType if p.is_builtin_database]
    current_selection = [p for p in retrieval.search_providers if p in scientific_databases]
    retrieval.search_providers = st.multiselect(
        "Scientific databases",
        options=scientific_databases,
        default=current_selection,
        format_func=lambda item: item.label,
        help=(
            "Available to the model when \u201cSearch scientific databases\u201d is enabled and "
            "the papers already indexed in the chat do not cover the question. Semantic "
            "Scholar, OpenAlex, Crossref and DOAJ index every field; PubMed and Europe PMC "
            "index biomedicine; arXiv indexes physics, mathematics, computer science, "
            "engineering and economics only. When none of the selected databases covers a "
            "question's field, the key-free ones that do are queried as well and the answer "
            "says so. Tavily web search (below) is used alongside these when needed and a "
            "key is set."
        ),
    )

    _render_custom_sources(settings)

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


def _render_custom_sources(settings: AppSettings) -> None:
    """Let the user wire in any JSON search API without code."""
    # `settings` is a fresh copy on every rerun, so unsaved rows added or
    # removed here live in session state until "Save settings" persists them.
    draft: list[CustomSearchSource] = st.session_state.setdefault(
        _CUSTOM_SOURCES_DRAFT, [s.model_copy(deep=True) for s in settings.custom_search_sources]
    )
    settings.custom_search_sources = draft

    st.markdown("#### Custom sources")
    st.caption(
        "Add a database that is not in the list above. It must answer a GET request "
        "with JSON: name the query parameter, the path to the list of records, and the "
        "path from one record to each field. Enabled sources are queried automatically "
        "alongside the databases selected above."
    )

    with st.expander("Example: how to describe an endpoint", expanded=not draft):
        example = EXAMPLE_SOURCE
        st.markdown(
            f"**{example.label}** — `GET {example.endpoint}?{example.query_param}=<query>"
            f"&{example.limit_param}=<n>`\n\n"
            "A record in the reply looks like "
            "`{\"hits\": {\"hits\": [{\"metadata\": {\"title\": ..., \"creators\": [{\"name\": ...}]}, "
            "\"doi\": ..., \"links\": {\"self_html\": ...}, \"files\": [{\"links\": {\"self\": ...}}]}]}}`, "
            "so the settings are:"
        )
        st.table(
            [
                {"Setting": "Results path", "Value": example.results_path,
                 "Meaning": "Where the list of records is inside the JSON reply"},
                *[
                    {"Setting": f"Field: {name}", "Value": path,
                     "Meaning": "Dotted path inside one record; numbers index into lists"}
                    for name, path in example.field_map.items()
                ],
                {"Setting": "Extra parameters", "Value": ", ".join(f"{k}={v}" for k, v in example.extra_params.items()),
                 "Meaning": "Sent with every request"},
            ]
        )
        if st.button("Add this example as a source", key="custom_source_example"):
            draft.append(example.model_copy(update={"id": CustomSearchSource().id}, deep=True))
            st.rerun()

    if st.button("Add empty source", key="custom_source_add"):
        draft.append(CustomSearchSource(label="New source"))
        st.rerun()

    for source in list(draft):
        _render_custom_source(draft, source)


def _render_custom_source(draft: list[CustomSearchSource], source: CustomSearchSource) -> None:
    """Edit one custom source in place; widget keys are per source id."""
    key = f"cs_{source.id}"
    status = "" if source.is_configured() else " — incomplete"
    with st.expander(f"{source.label or 'Unnamed source'}{status}", expanded=not source.is_configured()):
        head = st.columns([0.55, 0.2, 0.25])
        source.label = head[0].text_input("Name", value=source.label, key=f"{key}_label")
        source.enabled = head[1].checkbox("Enabled", value=source.enabled, key=f"{key}_enabled")
        if head[2].button("Remove", key=f"{key}_remove"):
            draft.remove(source)
            st.rerun()

        source.endpoint = st.text_input(
            "Endpoint URL (GET)", value=source.endpoint, key=f"{key}_endpoint",
            placeholder="https://example.org/api/search",
        )
        params = st.columns(3)
        source.query_param = params[0].text_input(
            "Query parameter", value=source.query_param, key=f"{key}_q",
            help="The parameter that carries the search text, e.g. q or query.",
        )
        source.limit_param = params[1].text_input(
            "Page-size parameter", value=source.limit_param, key=f"{key}_limit",
            help="Optional. Leave empty if the API has none.",
        )
        source.results_path = params[2].text_input(
            "Results path", value=source.results_path, key=f"{key}_results",
            help="Dotted path to the list of records, e.g. results or hits.hits.",
        )

        extra = st.text_input(
            "Extra parameters", key=f"{key}_extra",
            value=", ".join(f"{k}={v}" for k, v in source.extra_params.items()),
            help="Comma separated key=value pairs sent with every request.",
        )
        source.extra_params = _parse_pairs(extra)

        st.markdown("Field paths inside one record")
        grid = st.columns(4)
        for index, name in enumerate(("title", "abstract", "authors", "year", "doi", "url", "pdf_url")):
            source.field_map[name] = grid[index % 4].text_input(
                name, value=source.field_map.get(name, ""), key=f"{key}_f_{name}"
            )

        creds = st.columns(3)
        source.api_key = creds[0].text_input(
            "API key (optional)", value=source.api_key, type="password", key=f"{key}_key"
        )
        source.api_key_header = creds[1].text_input(
            "Key header", value=source.api_key_header, key=f"{key}_key_header",
            help="Header that carries the key, e.g. Authorization or x-api-key.",
        )
        source.api_key_param = creds[2].text_input(
            "Key query parameter", value=source.api_key_param, key=f"{key}_key_param",
            help="Used only when no header is given.",
        )

        options = ["all", *RESEARCH_DOMAINS]
        source.covers = st.multiselect(
            "Fields this source indexes",
            options=options,
            default=[c for c in source.covers if c in options] or ["all"],
            key=f"{key}_covers",
            help="Lets the app route a question here when the selected databases do not cover its field.",
        )


def _parse_pairs(text: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for chunk in text.split(","):
        if "=" in chunk:
            name, value = chunk.split("=", 1)
            if name.strip():
                pairs[name.strip()] = value.strip()
    return pairs


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


def _render_pricing(
    settings: AppSettings, provider: LLMProviderType, model: str
) -> None:
    """Show and manage the rate used to estimate what a turn costs."""
    pricing = settings.pricing
    if provider.is_local:
        st.caption("Local models run on your own hardware, so no cost is estimated.")
        return

    with st.expander("Cost estimation", expanded=False):
        st.caption(
            "Token counts come from the provider's own usage metadata. Prices are not "
            "exposed by any provider API, so they are downloaded from a public price "
            "catalogue and cached locally - nothing is fixed in the application."
        )

        pricing.auto_update = st.checkbox(
            "Keep prices up to date automatically",
            value=pricing.auto_update,
            key="pricing_auto_update",
            help=(
                f"Re-downloads the catalogue at most every {pricing_refresh_days()} days, "
                "while this page is open."
            ),
        )

        if pricing.auto_update and pricing_store.is_stale():
            _auto_update_prices()

        columns = st.columns([0.7, 0.3], vertical_alignment="bottom")
        updated = pricing_store.updated_at
        if updated is None:
            columns[0].caption("No price catalogue has been downloaded yet.")
        else:
            columns[0].caption(
                f"{pricing_store.count()} models priced · updated {updated:%Y-%m-%d}"
            )
        if columns[1].button("Update prices now", width="stretch"):
            _refresh_prices(quiet=False)

        if not model:
            return

        key = price_key(provider, model)
        resolved = pricing_store.resolve(provider, model, pricing.overrides)
        origin = {
            "override": "your manual override",
            "catalogue": "the downloaded catalogue",
            "offline": "the bundled offline table",
        }.get(resolved.source if resolved else "", "")

        st.markdown(f"**{model}** — USD per 1M tokens")
        rates = st.columns(2)
        input_price = rates[0].number_input(
            "Input", min_value=0.0, step=0.05, format="%.4f",
            value=float(resolved.input if resolved else 0.0),
            key=f"price_in_{key}",
        )
        output_price = rates[1].number_input(
            "Output", min_value=0.0, step=0.05, format="%.4f",
            value=float(resolved.output if resolved else 0.0),
            key=f"price_out_{key}",
        )

        if resolved is None:
            st.caption("This model has no published price. Enter one to estimate its cost.")
        elif origin:
            st.caption(f"Rate taken from {origin}. Change it to store your own, then save.")

        edited = resolved is None or (
            abs(input_price - resolved.input) > 1e-9 or abs(output_price - resolved.output) > 1e-9
        )
        if edited and (input_price or output_price):
            pricing.overrides[key] = ModelPrice(input=input_price, output=output_price)
        elif resolved is not None and resolved.source != "override":
            pricing.overrides.pop(key, None)


def _auto_update_prices() -> None:
    """Refresh a stale catalogue in the background, once per session.

    Deliberately not awaited: a slow download must never hold up the page,
    and the offline table already covers the interval until it lands.
    """
    if st.session_state.get(_PRICE_UPDATE_STARTED):
        return
    st.session_state[_PRICE_UPDATE_STARTED] = True
    submit(_swallow(pricing_store.refresh()))


async def _swallow(coroutine) -> None:
    try:
        await coroutine
    except Exception:  # noqa: BLE001 - a stale price never blocks the application
        pass


def _refresh_prices(quiet: bool) -> None:
    """Download the price catalogue, reporting the outcome unless quiet."""
    try:
        with st.spinner("Updating model prices…"):
            count = run_async(pricing_store.refresh())
    except ScientificRAGError as exc:
        if not quiet:
            st.warning(exc.user_message)
        return
    except Exception as exc:  # noqa: BLE001 - pricing must never break the page
        if not quiet:
            st.warning(f"The prices could not be updated: {exc}")
        return

    if quiet:
        return
    st.success(f"Prices updated for {count} models.")


def _credential_fingerprint(provider: LLMProviderType, config: ProviderSettings) -> str:
    """Identify the credentials a discovery result belongs to.

    The raw key never leaves this function: only its digest is used, as a
    cache key and as part of a widget key.
    """
    material = f"{provider.value}|{config.api_key}|{config.effective_base_url()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _discover_models(
    provider: LLMProviderType, config: ProviderSettings, fingerprint: str
) -> ModelCatalog:
    """Load the provider's model list for the credentials currently entered.

    The lookup must use ``config`` rather than the saved settings, otherwise a
    key that has just been typed is ignored until it is saved. Results are
    cached per credential fingerprint, so changing the key re-discovers
    automatically and an unchanged key is not re-fetched on every rerun.
    """
    nonce = st.session_state.get(_REFRESH_NONCE, 0)
    key = f"models_{provider.value}_{fingerprint}_{nonce}"
    cached = st.session_state.get(key)
    if cached is not None:
        return cached

    with st.spinner(f"Loading the {provider.label} model list…"):
        try:
            catalog = run_async(fetch_models(provider, config=config))
        except ScientificRAGError as exc:
            catalog = ModelCatalog(error=exc.user_message)
        except Exception as exc:  # noqa: BLE001 - discovery is best effort
            catalog = ModelCatalog(error=f"The model list could not be loaded: {exc}")

    st.session_state[key] = catalog
    return catalog


def _save_button(settings: AppSettings, section: str) -> None:
    """Persist the settings."""
    if st.button("Save settings", type="primary", key=f"save_{section}"):
        try:
            save_settings(settings)
            st.success("Settings saved.")
        except ScientificRAGError as exc:
            st.error(exc.user_message)
