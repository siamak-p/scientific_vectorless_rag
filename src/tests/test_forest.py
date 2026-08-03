"""The chat forest: one root, one node per PDF, sections below."""

from __future__ import annotations

from core.models import ChatIndex
from retrieval.pageindex.forest import index_statistics, render_navigation_tree
from tests.conftest import make_tree


def test_each_pdf_becomes_a_sibling_under_one_root() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-a", "Attention Is All You Need", ["Introduction", "Method"]))
    index.attach(make_tree("doc-b", "Deep Residual Learning", ["Background", "Results"]))

    assert index.root.level == 0
    assert len(index.root.children) == 2
    assert [child.node_type for child in index.root.children] == ["document", "document"]
    assert all(child.level == 1 for child in index.root.children)
    assert all(
        section.level == 2
        for child in index.root.children
        for section in child.children
    )


def test_adding_a_paper_leaves_existing_subtrees_untouched() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-a", "Paper A", ["Introduction", "Method"]))
    before = index.document_node("doc-a").model_dump()

    index.attach(make_tree("doc-b", "Paper B", ["Background"]))

    assert index.document_node("doc-a").model_dump() == before
    assert len(index.root.children) == 2


def test_detaching_removes_only_that_paper() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-a", "Paper A", ["Introduction"]))
    index.attach(make_tree("doc-b", "Paper B", ["Background"]))

    index.detach("doc-a")

    assert index.document_ids == ["doc-b"]
    assert index.document_node("doc-a") is None
    assert index.document_node("doc-b") is not None


def test_reattaching_the_same_paper_replaces_it() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-a", "Paper A", ["Introduction"]))
    index.attach(make_tree("doc-a", "Paper A", ["Introduction", "Method"]))

    assert len(index.root.children) == 1
    assert len(index.document_node("doc-a").children) == 2


def test_empty_index_is_reported_as_empty() -> None:
    assert ChatIndex(chat_id="chat-1").is_empty()


def test_statistics_count_documents_and_sections() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-a", "Paper A", ["Introduction", "Method"]))
    index.attach(make_tree("doc-b", "Paper B", ["Background"]))

    stats = index_statistics(index)

    assert stats["documents"] == 2
    assert stats["sections"] == 3


def test_navigation_tree_without_decisions_is_explained() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-a", "Paper A", ["Introduction"]))

    assert "No navigation" in render_navigation_tree(index, [])
