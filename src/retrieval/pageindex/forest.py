"""Chat-level document forest.

Every chat owns one index: a single root node with one child per PDF, and the
section hierarchy of that PDF below it. Adding a paper appends a subtree and
never rebuilds the ones already present, which is what makes indexing
incremental::

    Chat Knowledge Base                (root, level 0)
     |
     +-- Attention Is All You Need     (document, level 1)
     |    +-- Introduction             (section,  level 2)
     |    +-- Model Architecture       (section,  level 2)
     |    |    +-- Encoder and Decoder (section,  level 3)
     |    +-- Results                  (section,  level 2)
     |
     +-- Deep Residual Learning        (document, level 1)
          +-- Introduction             (section,  level 2)
          +-- Method                   (section,  level 2)
"""

from __future__ import annotations

from core.constants import NodeDecision
from core.models import ChatIndex, DocumentTree, NavigationDecision

_BRANCH = "├── "
_LAST_BRANCH = "└── "
_VERTICAL = "│   "
_SPACE = "    "

_DECISION_LABELS = {
    NodeDecision.SELECTED: "Selected",
    NodeDecision.REJECTED: "Rejected",
    NodeDecision.PARTIALLY_SELECTED: "Partially selected",
}


def build_chat_index(chat_id: str, trees: list[DocumentTree]) -> ChatIndex:
    """Assemble a fresh chat index from a list of document trees."""
    index = ChatIndex(chat_id=chat_id)
    for tree in trees:
        index.attach(tree)
    return index


def add_document(index: ChatIndex, tree: DocumentTree) -> ChatIndex:
    """Append (or refresh) one document subtree, leaving the others untouched."""
    index.attach(tree)
    return index


def remove_document(index: ChatIndex, document_id: str) -> ChatIndex:
    """Remove one document subtree from the index."""
    index.detach(document_id)
    return index


def index_statistics(index: ChatIndex) -> dict[str, int]:
    """Return counts describing the size of the index."""
    nodes = list(index.root.iter_nodes())
    return {
        "documents": len(index.root.children),
        "sections": sum(1 for node in nodes if node.node_type == "section"),
        "max_depth": max((node.level for node in nodes), default=0),
    }


def render_navigation_tree(index: ChatIndex, decisions: list[NavigationDecision]) -> str:
    """Render the navigation path as an inspectable ASCII tree.

    Nodes that were never reached are omitted so the user sees exactly the
    branches the retrieval engine considered.
    """
    by_node: dict[str, NavigationDecision] = {}
    for decision in decisions:
        current = by_node.get(decision.node_id)
        if current is None or decision.relevance_score > current.relevance_score:
            by_node[decision.node_id] = decision

    if not by_node:
        return "No navigation was performed for this answer."

    lines: list[str] = ["Document Root"]

    visible_documents = [
        child for child in index.root.children if _has_visible(child, by_node)
    ]
    for position, document in enumerate(visible_documents):
        is_last = position == len(visible_documents) - 1
        _render_node(document, by_node, prefix="", is_last=is_last, lines=lines)

    return "\n".join(lines)


def _render_node(
    node,
    by_node: dict[str, NavigationDecision],
    prefix: str,
    is_last: bool,
    lines: list[str],
) -> None:
    connector = _LAST_BRANCH if is_last else _BRANCH
    lines.append(f"{prefix}{connector}{node.title}")

    child_prefix = prefix + (_SPACE if is_last else _VERTICAL)
    decision = by_node.get(node.node_id)

    visible_children = [child for child in node.children if _has_visible(child, by_node)]

    if decision is not None and not visible_children:
        label = _DECISION_LABELS.get(decision.decision, decision.decision.value)
        pages = _format_pages(decision.related_pages)
        lines.append(f"{child_prefix}{_LAST_BRANCH}{label} ({decision.relevance_score:.2f}){pages}")
        return

    if decision is not None:
        label = _DECISION_LABELS.get(decision.decision, decision.decision.value)
        lines.append(f"{child_prefix}{_BRANCH}{label} ({decision.relevance_score:.2f})")

    for position, child in enumerate(visible_children):
        _render_node(
            child,
            by_node,
            prefix=child_prefix,
            is_last=position == len(visible_children) - 1,
            lines=lines,
        )


def _has_visible(node, by_node: dict[str, NavigationDecision]) -> bool:
    """Whether a node or any descendant carries a recorded decision."""
    return any(candidate.node_id in by_node for candidate in node.iter_nodes())


def _format_pages(pages: list[int]) -> str:
    """Render a page list compactly, e.g. ``, pages 3-5``."""
    if not pages:
        return ""
    if len(pages) == 1:
        return f", page {pages[0]}"
    return f", pages {pages[0]}-{pages[-1]}"
