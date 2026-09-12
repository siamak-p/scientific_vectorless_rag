"""PageIndex tree navigation — reasoning based, vectorless retrieval.

The navigator walks the chat's document forest the way a researcher scans a
table of contents: it reads titles, hierarchy and short excerpts, then decides
which branches to open and which to prune. Every decision is recorded so the
whole path can be replayed in Explainability Mode.
"""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel, Field

from core.constants import (
    MAX_PAGES_PER_QUERY,
    NodeDecision,
)
from core.models import ChatIndex, NavigationDecision, TreeNode
from llm.client import LLMClient
from observability.logger import get_logger, log_event

logger = get_logger("retrieval.navigator")

_FALLBACK_PAGES = 4
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CAPACITY_ERROR_MARKERS = (
    "rate limit",
    "rate_limit",
    "tokens per minute",
    "request too large",
    "error code: 413",
    "error code: 429",
)
_NAVIGATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "what", "when", "which", "why", "with",
}


class NavigationResult(BaseModel):
    """Outcome of navigating the forest for one question."""

    decisions: list[NavigationDecision] = Field(default_factory=list)
    pages_by_document: dict[str, list[int]] = Field(default_factory=dict)

    def total_pages(self) -> int:
        """Total number of pages selected across all documents."""
        return sum(len(pages) for pages in self.pages_by_document.values())

    def merge(self, other: "NavigationResult") -> "NavigationResult":
        """Merge another result into this one, de-duplicating pages."""
        self.decisions.extend(other.decisions)
        for document_id, pages in other.pages_by_document.items():
            merged = set(self.pages_by_document.get(document_id, [])) | set(pages)
            self.pages_by_document[document_id] = sorted(merged)
        return self


class _NodeEvaluation(BaseModel):
    """The model's judgement about a single node."""

    node_id: str = Field(description="Exact node id supplied in the request.")
    decision: NodeDecision = Field(
        description="selected, partially_selected, or rejected."
    )
    reason: str = Field(default="", description="One auditable sentence.")
    relevance_score: float = Field(default=0.0, description="0.0 to 1.0.")
    confidence_score: float = Field(default=0.0, description="0.0 to 1.0.")


class _BatchEvaluation(BaseModel):
    """Judgements for one sibling level, returned in a single response."""

    evaluations: list[_NodeEvaluation] = Field(default_factory=list)


class TreeNavigator:
    """Navigates a :class:`~core.models.ChatIndex` without embeddings."""

    def __init__(
        self,
        client: LLMClient,
        max_depth: int = 3,
        page_budget: int = MAX_PAGES_PER_QUERY,
    ) -> None:
        self._client = client
        self._max_depth = max(1, max_depth)
        self._page_budget = max(1, page_budget)
        self.rate_limit_encountered = False

    async def navigate(
        self,
        index: ChatIndex,
        question: str,
        allowed_document_ids: list[str] | None = None,
    ) -> NavigationResult:
        """Select the pages worth reading for one question.

        Args:
            index: The chat's unified document forest.
            question: The sub-question being answered.
            allowed_document_ids: Restrict navigation to these documents.

        Returns:
            The selected pages per document plus the full decision trace.
        """
        result = NavigationResult()

        candidates = [
            child
            for child in index.root.children
            if allowed_document_ids is None or child.document_id in allowed_document_ids
        ]
        if not candidates:
            return result

        # Give every selected paper a fair, proportional share of the page
        # budget up front. Without this, the traversal order alone decides
        # who gets read: the first document explored could recurse deep
        # enough to exhaust the whole budget before the others are ever
        # visited, leaving every citation pointing at just one paper. A
        # lone candidate still receives the entire budget, unchanged from
        # before.
        shares = {
            node.document_id: max(1, self._page_budget // len(candidates))
            for node in candidates
            if node.document_id
        }

        # Ranking has already selected these documents. When a complete paper
        # fits its fair share, reading it costs no more pages than navigating
        # its tree and avoids a needless provider call entirely.
        oversized: list[TreeNode] = []
        for node in candidates:
            pages = node.page_range()
            budget = shares.get(node.document_id or "", self._page_budget)
            if pages and len(pages) <= budget:
                self._collect(node, result, shares)
                result.decisions.append(
                    NavigationDecision(
                        node_id=node.node_id,
                        document_id=node.document_id,
                        document_title=node.title,
                        section_name=node.title,
                        parent_section="Corpus",
                        level=node.level,
                        decision=NodeDecision.SELECTED,
                        reason="The complete ranked paper fits within its page budget.",
                        relevance_score=1.0,
                        confidence_score=1.0,
                        related_pages=pages,
                        sub_question=question,
                    )
                )
            else:
                oversized.append(node)

        await self._walk(
            oversized,
            question,
            depth=1,
            result=result,
            parent_title="Corpus",
            document_title="",
            shares=shares,
        )

        # Ranking already decided these papers are worth reading. A document
        # whose sections were all rejected must still contribute something
        # citable, otherwise one paper that happened to be selected supplies
        # every reference in the answer while the rest are silently dropped.
        self._rescue_unread(candidates, result, question, shares)

        log_event(
            logger,
            "navigation.complete",
            "Tree navigation finished",
            question=question[:120],
            documents=len(result.pages_by_document),
            pages=result.total_pages(),
            decisions=len(result.decisions),
        )
        return result

    # -- traversal -------------------------------------------------------

    async def _walk(
        self,
        nodes: list[TreeNode],
        question: str,
        depth: int,
        result: NavigationResult,
        parent_title: str,
        document_title: str,
        shares: dict[str, int],
    ) -> None:
        """Evaluate a level of siblings, then recurse into the survivors."""
        if not nodes or depth > self._max_depth:
            return

        evaluations = await self._evaluate_many(
            nodes, question, parent_title, document_title
        )

        survivors: list[TreeNode] = []
        for node, evaluation in zip(nodes, evaluations):
            node_document_title = (
                node.title if node.node_type == "document" else document_title
            )
            result.decisions.append(
                NavigationDecision(
                    node_id=node.node_id,
                    document_id=node.document_id,
                    document_title=node_document_title,
                    section_name=node.title,
                    parent_section=parent_title,
                    level=node.level,
                    decision=evaluation.decision,
                    reason=evaluation.reason,
                    relevance_score=_clamp(evaluation.relevance_score),
                    confidence_score=_clamp(evaluation.confidence_score),
                    related_pages=node.page_range(),
                    sub_question=question,
                )
            )

            if evaluation.decision is NodeDecision.REJECTED:
                continue

            if node.children and depth < self._max_depth:
                survivors.append(node)
            else:
                self._collect(node, result, shares)

        for node in survivors:
            budget = shares.get(node.document_id or "", self._page_budget)
            if len(result.pages_by_document.get(node.document_id or "", [])) >= budget:
                continue
            await self._walk(
                node.children,
                question,
                depth + 1,
                result,
                parent_title=node.title,
                document_title=node.title if node.node_type == "document" else document_title,
                shares=shares,
            )

    async def _evaluate_many(
        self,
        nodes: list[TreeNode],
        question: str,
        parent_title: str,
        document_title: str,
    ) -> list[_NodeEvaluation]:
        """Evaluate sibling nodes together so one level costs one model call."""
        rendered: list[str] = []
        for node in nodes:
            children = ", ".join(child.title for child in node.children[:12]) or "none"
            pages = node.page_range()
            page_range = f"{pages[0]}-{pages[-1]}" if pages else "unknown"
            title = node.title if node.node_type == "document" else document_title
            rendered.append(
                f"NODE {node.node_id}\n"
                f"title: {node.title}\n"
                f"kind: {'whole document' if node.node_type == 'document' else 'section'}\n"
                f"level: {node.level}\n"
                f"parent: {parent_title}\n"
                f"document: {title or node.title}\n"
                f"pages: {page_range}\n"
                f"children: {children}\n"
                f"summary: {node.summary or 'not available'}"
            )

        fallback = [self._fallback(node, question) for node in nodes]
        try:
            result = await self._client.structured(
                "retrieval/tree_navigation_batch",
                _BatchEvaluation,
                temperature=0.0,
                question=question,
                nodes="\n\n".join(rendered),
                node_ids=[node.node_id for node in nodes],
            )
        except Exception as exc:  # noqa: BLE001 - conservative degradation
            if _is_rate_limit_error(exc):
                self.rate_limit_encountered = True
            if len(nodes) > 1 and not _is_capacity_error(exc):
                middle = len(nodes) // 2
                log_event(
                    logger,
                    "navigation.evaluate",
                    "Navigation batch failed; retrying smaller batches",
                    level=30,
                    nodes=len(nodes),
                    error=str(exc),
                )
                left, right = await asyncio.gather(
                    self._evaluate_many(
                        nodes[:middle], question, parent_title, document_title
                    ),
                    self._evaluate_many(
                        nodes[middle:], question, parent_title, document_title
                    ),
                )
                return left + right
            log_event(
                logger,
                "navigation.evaluate",
                "Navigation model unavailable; using query-aware lexical decisions",
                level=30,
                nodes=len(nodes),
                error=str(exc),
            )
            return fallback

        by_id = {item.node_id: item for item in result.evaluations}
        return [by_id.get(node.node_id, default) for node, default in zip(nodes, fallback)]

    @staticmethod
    def _fallback(node: TreeNode, question: str) -> _NodeEvaluation:
        query_terms = {
            term for term in _WORD_RE.findall(question.lower())
            if len(term) > 1 and term not in _NAVIGATION_STOPWORDS
        }
        if "ai" in query_terms:
            query_terms.update({"artificial", "intelligence"})
        searchable = " ".join(
            [node.title, node.summary, *(child.title for child in node.children[:12])]
        ).lower()
        searchable_terms = set(_WORD_RE.findall(searchable))
        overlap = len(query_terms & searchable_terms) / max(1, len(query_terms))
        decision = (
            NodeDecision.SELECTED if overlap >= 0.3 else NodeDecision.PARTIALLY_SELECTED
        )
        return _NodeEvaluation(
            node_id=node.node_id,
            decision=decision,
            reason="Selected from query overlap because model navigation was unavailable.",
            relevance_score=round(max(0.2, overlap), 3),
            confidence_score=0.25,
        )

    # -- page collection -------------------------------------------------

    def _collect(
        self, node: TreeNode, result: NavigationResult, shares: dict[str, int]
    ) -> None:
        """Add a node's pages to the result, respecting its document's fair share."""
        pages = node.page_range()
        if not pages or not node.document_id:
            return

        budget = shares.get(node.document_id, self._page_budget)
        existing = set(result.pages_by_document.get(node.document_id, []))
        remaining = budget - len(existing)
        if remaining <= 0:
            return

        additions = [page for page in pages if page not in existing][:remaining]
        if not additions:
            return

        existing.update(additions)
        result.pages_by_document[node.document_id] = sorted(existing)

    @staticmethod
    def _rescue_unread(
        candidates: list[TreeNode],
        result: NavigationResult,
        question: str,
        shares: dict[str, int],
    ) -> None:
        """Give every ranked document that selected no page a second chance.

        The pages are chosen by free lexical overlap with the question, so the
        rescue costs no provider call and still lands on the part of the paper
        most likely to answer it.
        """
        for node in candidates:
            document_id = node.document_id
            if not document_id or result.pages_by_document.get(document_id):
                continue

            budget = max(1, min(_FALLBACK_PAGES, shares.get(document_id, _FALLBACK_PAGES)))
            section, pages = _closest_section(node, question, budget)
            if not pages:
                continue

            result.pages_by_document[document_id] = pages
            result.decisions.append(
                NavigationDecision(
                    node_id=section.node_id,
                    document_id=document_id,
                    document_title=node.title,
                    section_name=section.title,
                    parent_section=node.title,
                    level=section.level,
                    decision=NodeDecision.PARTIALLY_SELECTED,
                    reason=(
                        "No section was judged relevant, so the closest matching "
                        "pages were read instead."
                    ),
                    relevance_score=0.2,
                    confidence_score=0.2,
                    related_pages=pages,
                    sub_question=question,
                )
            )


def _closest_section(
    document_node: TreeNode, question: str, budget: int
) -> tuple[TreeNode, list[int]]:
    """Return the section of a document that best matches the question."""
    query_terms = {
        term for term in _WORD_RE.findall(question.lower())
        if len(term) > 1 and term not in _NAVIGATION_STOPWORDS
    }

    best_node = document_node
    best_score = -1.0
    for node in _descendants(document_node):
        if not node.page_range():
            continue
        terms = set(_WORD_RE.findall(f"{node.title} {node.summary}".lower()))
        score = len(query_terms & terms) / max(1, len(query_terms))
        if score > best_score:
            best_node, best_score = node, score

    pages = best_node.page_range() or document_node.page_range()
    return best_node, pages[:budget]


def _descendants(node: TreeNode) -> list[TreeNode]:
    """Return every node below (and including) this one."""
    collected: list[TreeNode] = [node]
    for child in node.children:
        collected.extend(_descendants(child))
    return collected


def _clamp(value: float) -> float:
    """Constrain a score to the closed unit interval."""
    return round(max(0.0, min(1.0, float(value))), 3)


def _is_capacity_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CAPACITY_ERROR_MARKERS)


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in (
        "rate limit", "rate_limit", "quota", "tokens per minute", "tpm", "429"
    ))
