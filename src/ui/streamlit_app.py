"""Streamlit entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.constants import APP_DESCRIPTION, APP_NAME  # noqa: E402
from core.exceptions import ScientificRAGError  # noqa: E402
from observability.logger import configure_logging  # noqa: E402
from ui import services  # noqa: E402
from ui.components.chat_panel import render_chat  # noqa: E402
from ui.components.sidebar import render_sidebar  # noqa: E402
from ui.views.settings import render_settings  # noqa: E402


def main() -> None:
    """Configure the page and route to the requested view."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="�",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"about": APP_DESCRIPTION},
    )

    configure_logging()

    try:
        settings = services.bootstrap()
    except ScientificRAGError as exc:
        st.error(exc.user_message)
        st.stop()
        return

    page = st.session_state.setdefault("page", "chat")

    if page == "settings":
        render_settings(settings)
        return

    chat = services.active_chat(settings)
    options = render_sidebar(chat, settings)

    _, effective_model = services.effective_provider_model(chat, settings)
    if effective_model == "not selected":
        st.info(
            "No model is selected yet. Open **Settings** in the sidebar to choose a "
            "provider and model before asking a question."
        )

    render_chat(chat, settings, options)


main()
