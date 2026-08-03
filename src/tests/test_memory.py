"""Persistence, conversation isolation and shared topic memory."""

from __future__ import annotations

import pytest

from core.constants import DocumentSource, DocumentStatus
from core.models import Chat, ChatIndex, ChatSummary, DocumentRecord
from memory.chat_memory import GlobalMemory, render_transcript
from memory.database import Database
from memory.storage import (
    ChatIndexRepository,
    ChatRepository,
    DocumentRepository,
    MessageRepository,
    SummaryRepository,
)
from tests.conftest import make_tree


@pytest.fixture
async def db(tmp_path):
    """An isolated database for one test."""
    instance = Database(str(tmp_path / "test.sqlite3"))
    await instance.initialise()
    yield instance
    await instance.dispose()


@pytest.fixture
async def chats(db) -> tuple[ChatRepository, Chat, Chat]:
    """Two conversations sharing one database."""
    repository = ChatRepository(db)
    first = await repository.create(Chat(title="Transformers"))
    second = await repository.create(Chat(title="Protein folding"))
    return repository, first, second


async def test_schema_is_created_automatically(tmp_path) -> None:
    instance = Database(str(tmp_path / "auto.sqlite3"))
    await instance.initialise()

    repository = ChatRepository(instance)
    await repository.create(Chat(title="Ready to use"))

    assert len(await repository.list()) == 1
    await instance.dispose()


async def test_messages_are_isolated_between_chats(db, chats) -> None:
    _, first, second = chats
    messages = MessageRepository(db)

    await messages.add(first.id, "user", "Explain self-attention.")
    await messages.add(second.id, "user", "Explain AlphaFold.")

    first_messages = await messages.list(first.id)

    assert len(first_messages) == 1
    assert "self-attention" in first_messages[0].content
    assert len(await messages.list(second.id)) == 1


async def test_documents_are_isolated_between_chats(db, chats) -> None:
    _, first, second = chats
    documents = DocumentRepository(db)

    await documents.add(
        DocumentRecord(
            chat_id=first.id,
            filename="attention.pdf",
            file_hash="a" * 64,
            status=DocumentStatus.INDEXED,
            source=DocumentSource.UPLOADED,
        )
    )

    assert len(await documents.list(first.id)) == 1
    assert await documents.list(second.id) == []


async def test_the_same_pdf_is_recognised_by_content(db, chats) -> None:
    _, first, _ = chats
    documents = DocumentRepository(db)
    await documents.add(
        DocumentRecord(chat_id=first.id, filename="paper.pdf", file_hash="b" * 64)
    )

    assert await documents.find_by_hash(first.id, "b" * 64) is not None
    assert await documents.find_by_hash(first.id, "c" * 64) is None


async def test_indexes_are_isolated_between_chats(db, chats) -> None:
    _, first, second = chats
    indexes = ChatIndexRepository(db)

    index = ChatIndex(chat_id=first.id)
    index.attach(make_tree("doc-a", "Paper A", ["Introduction"]))
    await indexes.save(index)

    assert not (await indexes.load(first.id)).is_empty()
    assert (await indexes.load(second.id)).is_empty()


async def test_deleting_a_chat_removes_its_messages(db, chats) -> None:
    repository, first, _ = chats
    messages = MessageRepository(db)
    await messages.add(first.id, "user", "A question")

    await repository.delete(first.id)

    assert await messages.list(first.id) == []


async def test_search_matches_titles_and_message_content(db, chats) -> None:
    repository, first, _ = chats
    await MessageRepository(db).add(first.id, "user", "What is rotary positional encoding?")

    assert [chat.id for chat in await repository.search("rotary")] == [first.id]
    assert [chat.id for chat in await repository.search("Protein")] != [first.id]


async def test_global_memory_shares_topics_but_not_content(db, chats) -> None:
    _, first, second = chats
    summaries = SummaryRepository(db)
    await summaries.save(
        ChatSummary(
            chat_id=second.id,
            title="Protein folding",
            topics=["protein structure prediction", "AlphaFold"],
            summary="Themes only.",
            message_count=4,
        )
    )

    topics = await GlobalMemory(summaries).topics(first.id)

    assert "AlphaFold" in topics
    assert all("Explain" not in topic for topic in topics)


async def test_a_chat_never_sees_its_own_digest_as_shared(db, chats) -> None:
    _, first, _ = chats
    summaries = SummaryRepository(db)
    await summaries.save(
        ChatSummary(chat_id=first.id, title="Transformers", topics=["attention"], message_count=2)
    )

    assert await GlobalMemory(summaries).topics(first.id) == []


def test_transcript_rendering_is_compact() -> None:
    assert "not started" in render_transcript([])
