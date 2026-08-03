"""PageIndex tree construction.

The tree is built with the cheapest reliable signal available, in order:

1. the PDF's own embedded bookmark outline (exact, zero tokens);
2. rule based heading inference from the text (approximate, zero tokens);
3. an LLM pass over page-tagged text (used only when the first two fail).

Node summaries are taken verbatim from the opening of each section, which
gives the navigator real context without spending a token per node.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.constants import STRUCTURE_SCAN_CHARS
from core.models import DocumentTree, OutlineEntry, PageContent, TreeNode
from documents.parser import build_page_digest, infer_headings
from llm.client import LLMClient
from observability.logger import get_logger, log_event

logger = get_logger("retrieval.tree_builder")

_MIN_PAGES_FOR_TREE = 3
_MIN_INFERRED_HEADINGS = 4
_SUMMARY_CHARS = 320
_MAX_LEVEL = 3


class _StructureEntry(BaseModel):
    """One heading returned by the LLM structure pass."""

    title: str = Field(description="Section heading as printed, without numbering.")
    level: int = Field(description="1 for sections, 2 for subsections, 3 for deeper.")
    page_start: int = Field(description="Physical page number where the heading appears.")


class _StructureExtraction(BaseModel):
    """The full ordered heading list returned by the LLM."""

    entries: list[_StructureEntry] = Field(default_factory=list)


class TreeBuilder:
    """Builds the hierarchical PageIndex representation of one document."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    async def build(
        self,
        document_id: str,
        document_title: str,
        file_hash: str,
        pages: list[PageContent],
        embedded_outline: list[OutlineEntry] | None = None,
    ) -> DocumentTree:
        """Construct the PageIndex tree for one document."""
        total_pages = len(pages)
        if total_pages == 0:
            return self._flat_tree(document_id, document_title, file_hash, pages)

        outline, strategy = await self._resolve_outline(
            document_title, pages, embedded_outline or []
        )

        if not outline or total_pages < _MIN_PAGES_FOR_TREE:
            log_event(logger, "tree.build", "Using flat tree",
                      document_id=document_id, pages=total_pages, strategy=strategy)
            return self._flat_tree(document_id, document_title, file_hash, pages)

        root = TreeNode(
            title=document_title or "Document",
            level=0,
            node_type="document",
            document_id=document_id,
            page_start=1,
            page_end=total_pages,
            summary=_summarise(pages, 1, min(1, total_pages)),
        )
        self._assemble(root, outline, document_id, total_pages)
        _assign_page_ends(root, total_pages)
        _attach_summaries(root, pages)

        log_event(
            logger,
            "tree.build",
            "PageIndex tree built",
            document_id=document_id,
            strategy=strategy,
            sections=sum(1 for _ in root.iter_nodes()) - 1,
            pages=total_pages,
        )
        return DocumentTree(
            document_id=document_id,
            document_title=document_title,
            file_hash=file_hash,
            root=root,
            total_pages=total_pages,
        )

    # -- outline resolution ---------------------------------------------

    async def _resolve_outline(
        self,
        document_title: str,
        pages: list[PageContent],
        embedded: list[OutlineEntry],
    ) -> tuple[list[OutlineEntry], str]:
        """Return the best available outline and the strategy that produced it."""
        usable_embedded = _sanitise(embedded, len(pages))
        if len(usable_embedded) >= 2:
            return usable_embedded, "embedded_outline"

        inferred = _sanitise(infer_headings(pages), len(pages))
        if len(inferred) >= _MIN_INFERRED_HEADINGS:
            return inferred, "inferred_headings"

        if self._client is None:
            return inferred, "inferred_headings_no_llm"

        llm_outline = await self._llm_outline(document_title, pages)
        if len(llm_outline) >= 2:
            return llm_outline, "llm_structure"

        return inferred, "inferred_headings_fallback"

    async def _llm_outline(
        self, document_title: str, pages: list[PageContent]
    ) -> list[OutlineEntry]:
        digest = build_page_digest(pages, STRUCTURE_SCAN_CHARS)
        if not digest.strip():
            return []

        try:
            result = await self._client.structured(
                "documents/structure_extraction",
                _StructureExtraction,
                temperature=0.0,
                document_title=document_title or "Untitled document",
                total_pages=len(pages),
                text=digest,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to a flat tree
            log_event(logger, "tree.structure", "LLM structure extraction failed",
                      level=30, error=str(exc))
            return []

        entries = [
            OutlineEntry(title=e.title.strip(), level=e.level, page_start=e.page_start)
            for e in result.entries
            if e.title.strip()
        ]
        return _sanitise(entries, len(pages))

    # -- assembly --------------------------------------------------------

    @staticmethod
    def _assemble(
        root: TreeNode, outline: list[OutlineEntry], document_id: str, total_pages: int
    ) -> None:
        """Convert a flat outline into a hierarchy attached to ``root``."""
        stack: dict[int, TreeNode] = {0: root}

        for entry in outline:
            level = max(1, min(entry.level, _MAX_LEVEL))

            parent_level = level - 1
            while parent_level > 0 and parent_level not in stack:
                parent_level -= 1
            parent = stack.get(parent_level, root)

            node = TreeNode(
                title=entry.title,
                level=level,
                node_type="section",
                document_id=document_id,
                page_start=min(entry.page_start, total_pages),
                parent_id=parent.node_id,
            )
            parent.children.append(node)

            stack[level] = node
            for deeper in [lvl for lvl in stack if lvl > level]:
                del stack[deeper]

    @staticmethod
    def _flat_tree(
        document_id: str, document_title: str, file_hash: str, pages: list[PageContent]
    ) -> DocumentTree:
        """Return a single-node tree covering the whole document."""
        total = len(pages)
        root = TreeNode(
            title=document_title or "Document",
            level=0,
            node_type="document",
            document_id=document_id,
            page_start=1 if total else None,
            page_end=total or None,
            summary=_summarise(pages, 1, min(2, total)) if total else "",
        )
        return DocumentTree(
            document_id=document_id,
            document_title=document_title,
            file_hash=file_hash,
            root=root,
            total_pages=total,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise(entries: list[OutlineEntry], total_pages: int) -> list[OutlineEntry]:
    """Drop malformed entries and enforce monotonically increasing pages."""
    cleaned: list[OutlineEntry] = []
    seen: set[str] = set()
    last_page = 1

    for entry in entries:
        title = entry.title.strip()
        if not title or len(title) > 150:
            continue

        key = title.lower()
        if key in seen:
            continue

        page = entry.page_start
        if page < 1 or (total_pages and page > total_pages):
            continue
        if page < last_page:
            page = last_page

        seen.add(key)
        last_page = page
        cleaned.append(OutlineEntry(title=title, level=max(1, entry.level), page_start=page))

    return cleaned


def _assign_page_ends(node: TreeNode, total_pages: int) -> None:
    """Derive ``page_end`` for every node from the next sibling's start page."""
    if node.page_end is None:
        node.page_end = total_pages

    children = node.children
    for index, child in enumerate(children):
        if child.page_start is None:
            child.page_start = node.page_start or 1

        if index + 1 < len(children):
            next_start = children[index + 1].page_start or child.page_start
            child.page_end = max(child.page_start, next_start - 1)
        else:
            child.page_end = max(child.page_start, node.page_end or total_pages)

        _assign_page_ends(child, total_pages)


def _attach_summaries(node: TreeNode, pages: list[PageContent]) -> None:
    """Give every node a short verbatim excerpt so navigation has context."""
    for child in node.children:
        if child.page_start is not None and not child.summary:
            child.summary = _summarise(pages, child.page_start, child.page_start)
        _attach_summaries(child, pages)


def _summarise(pages: list[PageContent], start: int, end: int) -> str:
    """Return a short excerpt from a page range."""
    text = " ".join(
        page.text for page in pages if start <= page.page_number <= max(start, end)
    )
    collapsed = " ".join(text.split())
    return collapsed[:_SUMMARY_CHARS]
