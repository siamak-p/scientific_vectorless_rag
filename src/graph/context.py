"""Per-run dependencies for the workflow.

LangGraph state must stay serialisable for checkpointing, so live objects —
the LLM client, repositories, the token stream callback — are held here and
bound to the nodes instead of being pushed through the state dictionary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from core.models import AppSettings, Chat, TokenUsage
from documents.processor import DocumentProcessor
from llm.client import LLMClient
from llm.usage import UsageTracker
from memory.chat_memory import MemoryManager, memory_manager
from memory.storage import (
    ChatIndexRepository,
    DocumentRepository,
    chat_index_repository,
    document_repository,
)


@dataclass
class RunContext:
    """Everything one workflow run needs that cannot live in the state."""

    chat: Chat
    settings: AppSettings
    client: LLMClient
    usage: UsageTracker
    memory: MemoryManager = field(default=memory_manager)
    documents: DocumentRepository = field(default=document_repository)
    indexes: ChatIndexRepository = field(default=chat_index_repository)
    on_token: Callable[[str], None] | None = None
    on_stage: Callable[[str], None] | None = None

    @classmethod
    def for_chat(
        cls,
        chat: Chat,
        settings: AppSettings,
        on_token: Callable[[str], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> "RunContext":
        """Build a context from a chat's stored model configuration."""
        client = LLMClient.for_chat(chat)
        return cls(
            chat=chat,
            settings=settings,
            client=client,
            usage=client.usage,
            on_token=on_token,
            on_stage=on_stage,
        )

    def processor(self) -> DocumentProcessor:
        """Return a document processor bound to this run's client."""
        return DocumentProcessor(self.client)

    def emit_token(self, text: str) -> None:
        """Forward a generated token to the user interface, if it is listening."""
        if self.on_token is not None and text:
            self.on_token(text)

    def emit_stage(self, label: str) -> None:
        """Report which stage of the pipeline is currently running."""
        if self.on_stage is not None:
            self.on_stage(label)

    def take_usage(self) -> TokenUsage:
        """Return the usage accumulated so far and reset the tracker."""
        snapshot = self.usage.snapshot()
        self.usage.reset()
        return snapshot
