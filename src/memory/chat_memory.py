"""Conversation memory.

Two layers, deliberately asymmetric:

* **Chat memory** is complete but private. A chat sees its own recent turns
  verbatim and nothing from any other chat.
* **Global memory** is shared but abstract. It carries only the topics each
  other chat has covered, so the assistant can answer "what have we been
  working on?" without ever leaking a specific statement, document or result
  from another conversation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.constants import GLOBAL_MEMORY_MAX_TOPICS, MEMORY_RECENT_MESSAGE_WINDOW
from core.models import ChatMessage, ChatSummary
from llm.client import LLMClient
from memory.storage import (
    MessageRepository,
    SummaryRepository,
    message_repository,
    summary_repository,
)
from observability.logger import get_logger, log_event

logger = get_logger("memory.chat_memory")

_SUMMARY_REFRESH_EVERY = 6


class _SummarySchema(BaseModel):
    """Topic-level digest produced by the model."""

    topics: list[str] = Field(default_factory=list, description="Three to six short topics.")
    summary: str = Field(default="", description="Two sentences, themes only.")
    suggested_title: str = Field(default="", description="Short chat title.")


class ChatMemory:
    """Recent-turn memory for a single conversation."""

    def __init__(
        self,
        window: int = MEMORY_RECENT_MESSAGE_WINDOW,
        messages: MessageRepository = message_repository,
    ) -> None:
        self._window = window
        self._messages = messages

    async def history(self, chat_id: str) -> list[ChatMessage]:
        """Return the recent turns of one conversation."""
        return await self._messages.recent(chat_id, self._window)

    async def transcript(self, chat_id: str, limit: int | None = None) -> str:
        """Render recent turns as plain text for prompt injection."""
        history = await self._messages.recent(chat_id, limit or self._window)
        return render_transcript(history)


class GlobalMemory:
    """Topic-only recall across conversations."""

    def __init__(
        self,
        summaries: SummaryRepository = summary_repository,
        max_topics: int = GLOBAL_MEMORY_MAX_TOPICS,
    ) -> None:
        self._summaries = summaries
        self._max_topics = max_topics

    async def topics(self, chat_id: str) -> list[str]:
        """Return distinct topics drawn from every other conversation."""
        others = await self._summaries.others(chat_id)
        collected: list[str] = []
        seen: set[str] = set()

        for summary in others:
            for topic in summary.topics:
                key = topic.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                collected.append(topic.strip())
                if len(collected) >= self._max_topics:
                    return collected
        return collected

    async def describe(self, chat_id: str) -> str:
        """Render the shared topics as a prompt-ready block."""
        topics = await self.topics(chat_id)
        if not topics:
            return "No topics have been discussed in other conversations yet."
        return "\n".join(f"- {topic}" for topic in topics)


class MemoryManager:
    """Coordinates both memory layers and keeps the digests fresh."""

    def __init__(
        self,
        chat_memory: ChatMemory | None = None,
        global_memory: GlobalMemory | None = None,
        messages: MessageRepository = message_repository,
        summaries: SummaryRepository = summary_repository,
    ) -> None:
        self.chat = chat_memory or ChatMemory(messages=messages)
        self.shared = global_memory or GlobalMemory(summaries=summaries)
        self._messages = messages
        self._summaries = summaries

    async def context(self, chat_id: str) -> tuple[str, str]:
        """Return the recent transcript and the shared topic list."""
        return await self.chat.transcript(chat_id), await self.shared.describe(chat_id)

    async def refresh_summary(
        self, chat_id: str, chat_title: str, client: LLMClient, force: bool = False
    ) -> ChatSummary | None:
        """Rebuild a chat's topic digest when it has drifted out of date.

        Summarisation is throttled to every few turns so a normal conversation
        does not pay for an extra model call on each message.
        """
        message_count = await self._messages.count(chat_id)
        if message_count < 2:
            return None

        existing = await self._summaries.get(chat_id)
        if not force and existing and message_count - existing.message_count < _SUMMARY_REFRESH_EVERY:
            return existing

        history = await self._messages.recent(chat_id, MEMORY_RECENT_MESSAGE_WINDOW * 2)
        try:
            result = await client.structured(
                "memory/chat_summary",
                _SummarySchema,
                temperature=0.0,
                title=chat_title,
                messages=render_transcript(history),
            )
        except Exception as exc:  # noqa: BLE001 - memory must never break a chat
            log_event(logger, "memory.summary", "Summary refresh failed",
                      level=30, chat_id=chat_id, error=str(exc))
            return existing

        summary = ChatSummary(
            chat_id=chat_id,
            title=result.suggested_title.strip() or chat_title,
            topics=[topic.strip() for topic in result.topics if topic.strip()][:8],
            summary=result.summary.strip(),
            message_count=message_count,
        )
        await self._summaries.save(summary)
        log_event(logger, "memory.summary", "Summary refreshed",
                  chat_id=chat_id, topics=len(summary.topics))
        return summary

    async def mark_stale(self, chat_id: str) -> None:
        """Flag a chat's digest for regeneration."""
        await self._summaries.mark_stale(chat_id)


def render_transcript(messages: list[ChatMessage], max_chars: int = 6000) -> str:
    """Render messages as a compact ``Role: text`` transcript."""
    if not messages:
        return "This conversation has not started yet."

    lines: list[str] = []
    for message in messages:
        role = "User" if message.role == "user" else "Assistant"
        content = " ".join(message.content.split())
        lines.append(f"{role}: {content[:1200]}")

    transcript = "\n".join(lines)
    return transcript[-max_chars:] if len(transcript) > max_chars else transcript


memory_manager = MemoryManager()
