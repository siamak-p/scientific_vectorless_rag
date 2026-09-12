"""Pydantic data models used throughout the Scientific RAG application.

These models define the shape of every piece of data flowing through the
system: settings, documents, PageIndex trees, evidence, citations and the
final generated answer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from core.constants import (
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_MAX_SEARCH_PAPERS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_RETRIEVAL_DEPTH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    TOC_CHECK_PAGES,
    AnswerMode,
    CitationFormat,
    ClaimVerdict,
    DocumentSource,
    DocumentStatus,
    LLMProviderType,
    NodeDecision,
    SearchProviderType,
    UNAVAILABLE_METADATA,
)


def new_id() -> str:
    """Return a new random identifier."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# ===========================================================================
# Settings
# ===========================================================================

class ProviderSettings(BaseModel):
    """Credentials and defaults for a single LLM provider."""

    provider_type: LLMProviderType
    api_key: str = ""
    base_url: str = ""
    enabled: bool = True
    selected_model: str = ""

    def effective_base_url(self) -> str:
        """Return the base URL to use, falling back to local defaults."""
        if self.base_url:
            return self.base_url
        if self.provider_type is LLMProviderType.OLLAMA:
            return DEFAULT_OLLAMA_URL
        if self.provider_type is LLMProviderType.LMSTUDIO:
            return DEFAULT_LMSTUDIO_URL
        return ""

    def missing_requirement(self) -> str:
        """Name what still has to be configured before this provider can answer."""
        if self.provider_type.requires_base_url and not self.effective_base_url().strip():
            return "server URL"
        if self.provider_type.requires_api_key and not self.api_key.strip():
            return "API key"
        if not self.selected_model.strip():
            return "model"
        return ""


class LLMSettings(BaseModel):
    """Model level generation parameters."""

    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    top_p: float = DEFAULT_TOP_P
    streaming: bool = True
    system_prompt: str = ""


class RetrievalSettings(BaseModel):
    """Retrieval pipeline tunables."""

    max_search_papers: int = DEFAULT_MAX_SEARCH_PAPERS
    search_providers: list[SearchProviderType] = Field(
        default_factory=lambda: [
            SearchProviderType.ARXIV,
            SearchProviderType.SEMANTIC_SCHOLAR,
            SearchProviderType.OPENALEX,
        ]
    )
    retrieval_depth: int = DEFAULT_RETRIEVAL_DEPTH
    toc_check_pages: int = TOC_CHECK_PAGES
    max_pages_per_query: int = 24
    max_evidence_items: int = 20
    download_pdfs: bool = True


class InterfaceSettings(BaseModel):
    """User interface preferences."""

    theme: str = "dark"
    streaming_enabled: bool = True
    show_token_usage: bool = True


class ResearchSettings(BaseModel):
    """Defaults for the research oriented features."""

    citation_format: CitationFormat = CitationFormat.APA
    default_answer_mode: AnswerMode = AnswerMode.SHORT_ANSWER
    deep_research_iterations: int = 3
    strict_grounding: bool = True

    # The sidebar's "Research options" toggles. Persisted so a choice the
    # user makes sticks across chats, page navigation and app restarts,
    # until they change it again themselves.
    show_retrieval_reasoning: bool = True
    auto_search_default: bool = False
    deep_research_default: bool = False


class ModelPrice(BaseModel):
    """What one model charges, in USD per one million tokens."""

    input: float = 0.0
    output: float = 0.0
    source: str = ""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return the USD cost of a call with these token counts."""
        spend = (input_tokens / 1_000_000) * self.input
        spend += (output_tokens / 1_000_000) * self.output
        return round(spend, 6)


class PricingSettings(BaseModel):
    """How model prices are obtained."""

    # Prices are not exposed by any provider API, so they are downloaded from
    # the catalogue configured in providers.yaml rather than written in code.
    auto_update: bool = True
    # Keyed by "<provider>/<model>"; wins over every downloaded price.
    overrides: dict[str, ModelPrice] = Field(default_factory=dict)


class CustomSearchSource(BaseModel):
    """A user-defined scientific database reached through a JSON GET endpoint.

    Most academic APIs share one shape - a GET request with a query parameter
    that returns a JSON list of records - and differ only in parameter and
    field names. ``field_map`` holds dotted paths (``metadata.title``,
    ``links.0.url``) from one record to the fields the pipeline needs.
    """

    id: str = Field(default_factory=new_id)
    label: str = ""
    enabled: bool = True
    endpoint: str = ""
    query_param: str = "q"
    limit_param: str = ""
    extra_params: dict[str, str] = Field(default_factory=dict)
    api_key: str = ""
    # The key is sent in this header when set, else as this query parameter.
    api_key_header: str = ""
    api_key_param: str = ""
    results_path: str = "results"
    field_map: dict[str, str] = Field(
        default_factory=lambda: {
            "title": "title",
            "abstract": "abstract",
            "authors": "authors",
            "year": "year",
            "doi": "doi",
            "url": "url",
            "pdf_url": "pdf_url",
        }
    )
    covers: list[str] = Field(default_factory=lambda: ["all"])

    def is_configured(self) -> bool:
        """Whether the source has what a request needs: a name and an http(s) URL."""
        parsed = urlparse(self.endpoint.strip())
        return bool(self.label.strip()) and parsed.scheme in ("http", "https") and bool(parsed.netloc)

    def covers_domain(self, domain: str) -> bool:
        """Whether the source declares coverage of a research field."""
        fields = {field.lower() for field in self.covers}
        return not domain or "all" in fields or domain.lower() in fields


class AppSettings(BaseModel):
    """Top level application settings, persisted as JSON in the app home."""

    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    active_provider: str = ""
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    interface: InterfaceSettings = Field(default_factory=InterfaceSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    pricing: PricingSettings = Field(default_factory=PricingSettings)
    custom_search_sources: list[CustomSearchSource] = Field(default_factory=list)
    tavily_api_key: str = ""
    semantic_scholar_api_key: str = ""
    contact_email: str = ""

    def provider(self, provider_type: LLMProviderType) -> ProviderSettings:
        """Return the configuration for a provider, creating a default if absent."""
        existing = self.providers.get(provider_type.value)
        if existing is not None:
            return existing
        created = ProviderSettings(provider_type=provider_type)
        self.providers[provider_type.value] = created
        return created


# ===========================================================================
# Chats and messages
# ===========================================================================

class Chat(BaseModel):
    """A single conversation thread with isolated memory and documents."""

    id: str = Field(default_factory=new_id)
    title: str = "New Chat"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    provider: str = ""
    model: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    top_p: float = DEFAULT_TOP_P
    system_prompt: str = ""


class ChatMessage(BaseModel):
    """One persisted message belonging to a chat."""

    id: str = Field(default_factory=new_id)
    chat_id: str
    role: str
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ChatSummary(BaseModel):
    """A high level, topic-only summary of a conversation.

    Used to give the assistant a *general* awareness of what has been
    discussed in other chats without leaking their specific content.
    """

    chat_id: str
    title: str = ""
    topics: list[str] = Field(default_factory=list)
    summary: str = ""
    message_count: int = 0
    updated_at: datetime = Field(default_factory=utc_now)


# ===========================================================================
# Documents
# ===========================================================================

class DocumentMetadata(BaseModel):
    """Rich bibliographic metadata for a scientific paper."""

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    doi: str = ""
    url: str = ""
    keywords: list[str] = Field(default_factory=list)
    publication_year: Optional[int] = None
    journal: str = ""
    conference: str = ""
    publisher: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    citation_count: int = 0
    is_peer_reviewed: bool = False
    venue_rank: float = 0.0
    references: list[str] = Field(default_factory=list)
    provider: Optional[SearchProviderType] = None

    def display_title(self, fallback: str = "") -> str:
        """Return the best available title for display purposes."""
        return self.title or fallback or UNAVAILABLE_METADATA

    def author_string(self) -> str:
        """Return the author list formatted for compact display."""
        if not self.authors:
            return UNAVAILABLE_METADATA
        if len(self.authors) > 3:
            return f"{self.authors[0]} et al."
        return ", ".join(self.authors)


class FigureRef(BaseModel):
    """A figure detected in the document."""

    label: str
    caption: str = ""
    page_number: int


class TableRef(BaseModel):
    """A table detected in the document."""

    label: str
    caption: str = ""
    page_number: int


class PageContent(BaseModel):
    """Extracted text for a single PDF page."""

    page_number: int
    text: str
    char_count: int = 0


class ParsedDocument(BaseModel):
    """Everything extracted from a PDF before the tree is built."""

    document_id: str
    file_hash: str
    page_count: int
    pages: list[PageContent] = Field(default_factory=list)
    outline: list["OutlineEntry"] = Field(default_factory=list)
    figures: list[FigureRef] = Field(default_factory=list)
    tables: list[TableRef] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)

    def page_text(self, page_number: int) -> str:
        """Return the text of a single page, or an empty string."""
        for page in self.pages:
            if page.page_number == page_number:
                return page.text
        return ""

    def pages_in_range(self, start: int, end: int) -> list[PageContent]:
        """Return every page whose number falls inside ``[start, end]``."""
        return [p for p in self.pages if start <= p.page_number <= end]


class OutlineEntry(BaseModel):
    """A structural heading discovered in the PDF (embedded outline or inferred)."""

    title: str
    level: int
    page_start: int


ParsedDocument.model_rebuild()


class DocumentRecord(BaseModel):
    """A PDF document tracked inside a single chat."""

    id: str = Field(default_factory=new_id)
    chat_id: str
    filename: str
    file_path: str = ""
    file_hash: str = ""
    source: DocumentSource = DocumentSource.UPLOADED
    status: DocumentStatus = DocumentStatus.PENDING
    page_count: int = 0
    error_message: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def has_pdf(self) -> bool:
        """Whether a local PDF file is available for this record."""
        return bool(self.file_path)

    def label(self) -> str:
        """Return the best short label for this document."""
        return self.metadata.title or self.filename


# ===========================================================================
# PageIndex tree
# ===========================================================================

class TreeNode(BaseModel):
    """A node in the hierarchical PageIndex structure.

    The tree is a *forest under a single root*: level 0 is the chat knowledge
    base, level 1 is one node per document, and deeper levels mirror the
    section hierarchy of that document.
    """

    node_id: str = Field(default_factory=new_id)
    title: str
    level: int = 0
    node_type: str = "section"  # "root" | "document" | "section"
    document_id: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    summary: str = ""
    parent_id: Optional[str] = None
    children: list["TreeNode"] = Field(default_factory=list)

    def page_range(self) -> list[int]:
        """Return every page number covered by this node."""
        if self.page_start is None:
            return []
        end = self.page_end if self.page_end is not None else self.page_start
        if end < self.page_start:
            end = self.page_start
        return list(range(self.page_start, end + 1))

    def iter_nodes(self):
        """Yield this node and every descendant, depth first."""
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def find(self, node_id: str) -> Optional["TreeNode"]:
        """Return the descendant with the given id, if present."""
        for node in self.iter_nodes():
            if node.node_id == node_id:
                return node
        return None


TreeNode.model_rebuild()


class DocumentTree(BaseModel):
    """PageIndex tree for one document."""

    document_id: str
    document_title: str = ""
    file_hash: str = ""
    root: TreeNode
    total_pages: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class ChatIndex(BaseModel):
    """The unified multi-document tree for one chat.

    A single root node holds one child per document. Adding a document only
    appends a new subtree; existing subtrees are never rebuilt.
    """

    chat_id: str
    root: TreeNode = Field(
        default_factory=lambda: TreeNode(title="Chat Knowledge Base", level=0, node_type="root")
    )
    document_ids: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)

    def document_node(self, document_id: str) -> Optional[TreeNode]:
        """Return the top level node representing a document."""
        for child in self.root.children:
            if child.document_id == document_id:
                return child
        return None

    def attach(self, tree: DocumentTree) -> None:
        """Attach (or replace) a document subtree under the chat root."""
        self.detach(tree.document_id)

        node = TreeNode(
            title=tree.document_title or tree.root.title,
            level=1,
            node_type="document",
            document_id=tree.document_id,
            page_start=1,
            page_end=tree.total_pages,
            summary=tree.root.summary,
            parent_id=self.root.node_id,
        )
        for child in tree.root.children:
            node.children.append(_rebase(child, parent_id=node.node_id, level=2,
                                         document_id=tree.document_id))

        self.root.children.append(node)
        self.document_ids.append(tree.document_id)
        self.updated_at = utc_now()

    def detach(self, document_id: str) -> None:
        """Remove a document subtree from the chat index."""
        self.root.children = [c for c in self.root.children if c.document_id != document_id]
        self.document_ids = [d for d in self.document_ids if d != document_id]
        self.updated_at = utc_now()

    def is_empty(self) -> bool:
        """Whether the index currently holds no documents."""
        return not self.root.children


def _rebase(node: TreeNode, parent_id: str, level: int, document_id: str) -> TreeNode:
    """Return a copy of ``node`` re-levelled under a new parent."""
    rebased = TreeNode(
        node_id=node.node_id,
        title=node.title,
        level=level,
        node_type="section",
        document_id=document_id,
        page_start=node.page_start,
        page_end=node.page_end,
        summary=node.summary,
        parent_id=parent_id,
    )
    rebased.children = [
        _rebase(child, parent_id=rebased.node_id, level=level + 1, document_id=document_id)
        for child in node.children
    ]
    return rebased


# ===========================================================================
# Retrieval, evidence and citations
# ===========================================================================

class NavigationDecision(BaseModel):
    """Record of the navigator's decision about one tree node."""

    node_id: str
    document_id: str = ""
    document_title: str = ""
    section_name: str
    parent_section: str = ""
    level: int = 1
    decision: NodeDecision
    reason: str = ""
    relevance_score: float = 0.0
    confidence_score: float = 0.0
    related_pages: list[int] = Field(default_factory=list)
    sub_question: str = ""


class EvidenceItem(BaseModel):
    """A single validated piece of retrieved evidence."""

    id: str = Field(default_factory=new_id)
    document_id: str
    document_title: str = ""
    page_number: int
    section: str = ""
    subsection: str = ""
    content: str
    quote: str = ""
    retrieval_reason: str = ""
    confidence_score: float = 0.0
    sub_question: str = ""
    figure_number: str = ""
    table_number: str = ""

    def marker(self, index: int) -> str:
        """Return the inline marker used by the answer generator."""
        return f"[E{index}]"


class Citation(BaseModel):
    """A fully traceable citation attached to a claim in the answer."""

    evidence_id: str = ""
    document_id: str = ""
    paper_title: str = ""
    authors: list[str] = Field(default_factory=list)
    doi: str = ""
    url: str = ""
    journal: str = ""
    conference: str = ""
    publisher: str = ""
    publication_year: Optional[int] = None
    volume: str = ""
    issue: str = ""
    pages: str = ""
    section: str = ""
    subsection: str = ""
    pdf_page_number: Optional[int] = None
    supporting_quote: str = ""
    figure_number: str = ""
    table_number: str = ""
    confidence_score: float = 0.0

    def venue(self) -> str:
        """Return the publication venue, or a clear unavailable marker."""
        return self.journal or self.conference or self.publisher or UNAVAILABLE_METADATA


class ClaimValidation(BaseModel):
    """Result of checking one claim from the answer against the evidence."""

    claim: str
    verdict: ClaimVerdict
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = ""


class ValidationReport(BaseModel):
    """Aggregate hallucination-control report for one generated answer."""

    claims: list[ClaimValidation] = Field(default_factory=list)
    removed_statements: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    grounding_score: float = 1.0
    needs_more_information: bool = False

    @property
    def unsupported_count(self) -> int:
        """Number of claims that could not be grounded in evidence."""
        return sum(1 for c in self.claims if c.verdict is ClaimVerdict.UNSUPPORTED)


# ===========================================================================
# Query planning
# ===========================================================================

class SubQuestion(BaseModel):
    """One decomposed sub-question produced by query planning."""

    id: str = Field(default_factory=new_id)
    question: str
    purpose: str = ""
    answered: bool = False
    answer: str = ""


class QueryPlan(BaseModel):
    """A structured plan for answering a user query."""

    original_query: str = ""
    intent: str = ""
    search_queries: list[str] = Field(default_factory=list)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    requires_multi_hop: bool = False
    is_conversational: bool = False
    language: str = Field(
        default="",
        description="English name of the language the user wrote in, e.g. 'Persian'.",
    )
    domain: str = Field(
        default="",
        description=(
            "Research field of the question, used to pick databases that index it; "
            "one of core.constants.RESEARCH_DOMAINS."
        ),
    )
    needs_clarification: bool = Field(
        default=False,
        description="True when the request cannot be researched until the user clarifies it.",
    )
    clarification_question: str = Field(
        default="",
        description="The question to ask the user, written in the user's language.",
    )
    correction_note: str = Field(
        default="",
        description=(
            "Short note in the user's language explaining a corrected typo or "
            "non-standard term that was interpreted; empty when nothing was corrected."
        ),
    )

    def questions(self) -> list[str]:
        """Return the sub-questions, falling back to the original query."""
        if self.sub_questions:
            return [sq.question for sq in self.sub_questions]
        return [self.original_query] if self.original_query else []


class SearchAssessment(BaseModel):
    """Whether the current chat corpus needs new papers for this query."""

    search_needed: bool = Field(
        default=False,
        description="True only when the existing papers cannot adequately cover the query.",
    )
    corpus_relevant: bool = Field(
        default=True,
        description=(
            "False when no indexed paper is about the query's topic at all. The "
            "assistant answers only from indexed papers, so this always requires a search."
        ),
    )
    suggested_papers: int = Field(
        default=0,
        ge=0,
        description="Number of new papers needed; zero when search is unnecessary.",
    )
    reasoning: str = Field(default="", description="One concise coverage-based reason.")


# ===========================================================================
# Ranking
# ===========================================================================

class PaperRankingEntry(BaseModel):
    """Ranking scores for one candidate paper."""

    document_id: str
    title: str = ""
    source: DocumentSource = DocumentSource.UPLOADED
    relevance_score: float = 0.0
    recency_score: float = 0.0
    citation_count: int = 0
    quality_score: float = 0.0
    coverage_score: float = 0.0
    overall_score: float = 0.0
    reason: str = ""


# ===========================================================================
# Token usage and cost
# ===========================================================================

class TokenUsage(BaseModel):
    """Cumulative token usage and estimated spend."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    calls: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """Return the sum of this usage and ``other``."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=round(self.estimated_cost_usd + other.estimated_cost_usd, 6),
            calls=self.calls + other.calls,
        )


# ===========================================================================
# Deep research
# ===========================================================================

class ResearchIteration(BaseModel):
    """One cycle of the deep research loop."""

    index: int
    focus: str
    search_queries: list[str] = Field(default_factory=list)
    new_document_ids: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    knowledge_gaps: list[str] = Field(default_factory=list)
    is_sufficient: bool = False


class ResearchTrace(BaseModel):
    """The complete audit trail of a deep research run."""

    iterations: list[ResearchIteration] = Field(default_factory=list)
    total_papers_examined: int = 0
    total_evidence_items: int = 0


# ===========================================================================
# Final answer
# ===========================================================================

class GeneratedAnswer(BaseModel):
    """The complete system response with answer, citations and full trace."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    navigation_trace: list[NavigationDecision] = Field(default_factory=list)
    query_plan: Optional[QueryPlan] = None
    paper_rankings: list[PaperRankingEntry] = Field(default_factory=list)
    validation: Optional[ValidationReport] = None
    research_trace: Optional[ResearchTrace] = None
    mode: AnswerMode = AnswerMode.SHORT_ANSWER
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)
