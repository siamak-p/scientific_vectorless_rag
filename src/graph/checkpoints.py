"""LangGraph checkpointing.

Every chat is a separate thread, so resuming a conversation restores exactly
that conversation's workflow state and never another's. SQLite is used instead
of Postgres to keep the application a self-contained local tool.
"""

from __future__ import annotations

import asyncio
import enum
import inspect
from contextlib import AsyncExitStack

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from core import constants as core_constants
from core import models as core_models
from core.constants import CHECKPOINT_FILE, ensure_directories
from observability.logger import get_logger, log_event

logger = get_logger("graph.checkpoints")


def _checkpointed_types() -> tuple[type, ...]:
    """Return every domain type that may appear inside a checkpoint.

    Derived from the modules themselves rather than listed by hand, so a new
    state field cannot be forgotten here. LangGraph blocks the deserialisation
    of types it was not told about, and warns about it until then.
    """
    return tuple(
        obj
        for module in (core_models, core_constants)
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and obj.__module__ == module.__name__
        and issubclass(obj, (BaseModel, enum.Enum))
    )


def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_checkpointed_types())


class CheckpointManager:
    """Owns the process-wide checkpointer and its connection lifetime."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path or str(CHECKPOINT_FILE)
        self._saver = None
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    async def saver(self):
        """Return the shared checkpointer, opening it on first use.

        Falls back to an in-memory saver when the SQLite file cannot be opened,
        so a read-only or locked home directory degrades gracefully instead of
        breaking the application.
        """
        if self._saver is not None:
            return self._saver

        async with self._lock:
            if self._saver is not None:
                return self._saver

            ensure_directories()
            try:
                self._stack = AsyncExitStack()
                saver = await self._stack.enter_async_context(
                    AsyncSqliteSaver.from_conn_string(self._path)
                )
                saver.serde = _serializer()
                await saver.setup()
                self._saver = saver
                log_event(logger, "checkpoint.ready", "Checkpointer ready", path=self._path)
            except Exception as exc:  # noqa: BLE001 - degrade instead of failing
                log_event(logger, "checkpoint.fallback",
                          "Persistent checkpointing unavailable, using memory",
                          level=30, error=str(exc))
                self._stack = None
                self._saver = InMemorySaver(serde=_serializer())
        return self._saver

    async def close(self) -> None:
        """Release the checkpointer connection."""
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self._saver = None


def thread_config(chat_id: str) -> dict:
    """Return the LangGraph config that isolates one chat's checkpoints."""
    return {"configurable": {"thread_id": chat_id}}


checkpoint_manager = CheckpointManager()
