"""Bridging Streamlit's synchronous script model with the async pipeline.

Streamlit re-runs the script on every interaction, so a single long-lived
event loop is kept in a background thread. That keeps the SQLAlchemy engine,
the checkpointer and every open connection bound to one loop instead of being
rebuilt — and invalidated — on each rerun.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

import streamlit as st

T = TypeVar("T")

_TOKEN = "token"
_STAGE = "stage"


@st.cache_resource(show_spinner=False)
def get_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, starting it on first use."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="scientific-rag-loop", daemon=True)
    thread.start()
    return loop


def run_async(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on the shared loop and wait for its result."""
    return asyncio.run_coroutine_threadsafe(coroutine, get_loop()).result()


def submit(coroutine: Coroutine[Any, Any, T]) -> Future[T]:
    """Schedule a coroutine on the shared loop without waiting for it."""
    return asyncio.run_coroutine_threadsafe(coroutine, get_loop())


class StreamBridge:
    """Carries tokens and stage updates from the loop thread to the script."""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()

    def on_token(self, text: str) -> None:
        """Callback handed to the workflow for generated tokens."""
        self._queue.put((_TOKEN, text))

    def on_stage(self, label: str) -> None:
        """Callback handed to the workflow for progress updates."""
        self._queue.put((_STAGE, label))

    def consume(
        self,
        future: Future[T],
        answer_slot,
        status_slot,
        started_at: float | None = None,
    ) -> T:
        """Render tokens as they arrive and return the workflow result.

        ``status_slot`` is refreshed on every poll tick (not just when a new
        stage arrives), so the elapsed time ticks up live while a stage is
        in progress instead of jumping only between stages.
        """
        chunks: list[str] = []
        start = started_at if started_at is not None else time.monotonic()
        stage = "Starting"

        def render_status() -> None:
            if status_slot is not None:
                elapsed = time.monotonic() - start
                status_slot.caption(f"⏳ {elapsed:0.0f}s · {stage}…")

        render_status()
        while True:
            try:
                kind, payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                if future.done():
                    break
                render_status()
                continue

            if kind == _TOKEN:
                chunks.append(payload)
                answer_slot.markdown("".join(chunks) + " ▌")
            else:
                stage = payload
            render_status()

        while not self._queue.empty():
            kind, payload = self._queue.get()
            if kind == _TOKEN:
                chunks.append(payload)

        if chunks:
            answer_slot.markdown("".join(chunks))
        if status_slot is not None:
            status_slot.empty()

        result = future.result()
        elapsed = time.monotonic() - start
        if hasattr(result, "elapsed_seconds"):
            result.elapsed_seconds = elapsed
        return result
