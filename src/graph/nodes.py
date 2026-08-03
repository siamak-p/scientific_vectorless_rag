"""Workflow nodes.

Each node is a small, independently testable step. Nodes return only the state
keys they change, never mutate the state in place, and never raise: a failure
is recorded in ``error`` so the graph can route to a graceful response.
"""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel, Field

from core.constants import (
    AnswerMode,
    DocumentSource,
    DOWNLOAD_CONCURRENCY,
    MAX_EVIDENCE_ITEMS,
    SearchProviderType,
)
from core.exceptions import ScientificRAGError
from core.models import (
    DocumentRecord,
    EvidenceItem,
    PaperRankingEntry,
    QueryPlan,
    SearchAssessment,
    SubQuestion,
)
from documents.cache import document_cache
from documents.pdf_loader import PDFLoader
from graph.context import RunContext
from graph.state import GraphState, documents_by_id
from observability.logger import get_logger, log_event
from retrieval.citations import build_citations, format_bibliography
from retrieval.evidence import (
    EvidenceExtractor,
    deduplicate,
    format_evidence_block,
    select_top,
)
from retrieval.pageindex.navigator import TreeNavigator
from retrieval.ranking import PaperRanker
from retrieval.validation import AnswerValidator
from scientific_search.aggregator import (
    SearchAggregator,
    available_providers,
    effective_search_providers,
)
from scientific_search.base import SearchResult
from scientific_search.downloader import PaperDownloader

logger = get_logger("graph.nodes")

_STYLE_INSTRUCTIONS = {
    AnswerMode.SHORT_ANSWER: (
        "Answer directly in at most four paragraphs. Lead with the conclusion."
    ),
    AnswerMode.RESEARCH_REPORT: (
        "Write a structured research report using the required section headings."
    ),
    AnswerMode.DEEP_RESEARCH: (
        "Write an exhaustive review that compares sources and states open questions."
    ),
}

_ANSWER_PROMPTS = {
    AnswerMode.SHORT_ANSWER: "answer/short_answer",
    AnswerMode.RESEARCH_REPORT: "answer/research_report",
    AnswerMode.DEEP_RESEARCH: "answer/deep_research_report",
}

_CONVERSATIONAL_PATTERN = re.compile(
    r"^(hi|hello|hey|hiya|yo|howdy|good\s?(morning|afternoon|evening|night)|"
    r"thanks?( you)?|thank you|cheers|bye|goodbye|see you|ok|okay|"
    r"how are you|what'?s up|salaam|سلام|درود|خوبی|چطوری|چطورید|ممنون|متشکرم|مرسی|خداحافظ)"
    r"[\s!.,?\u06F0-\u06F9]*$",
    re.IGNORECASE,
)

_SEARCH_ASSESSMENT_ABSTRACT_CHARS = 1_200


def _is_trivially_conversational(query: str) -> bool:
    """Recognise greetings and pleasantries without spending an LLM call on them.

    Anything longer or more ambiguous still goes through the full classifier;
    this only short-circuits the small-talk that would otherwise cost a full
    extra round trip before the conversational reply itself is generated.
    """
    stripped = query.strip()
    if not stripped or len(stripped) > 40:
        return False
    return bool(_CONVERSATIONAL_PATTERN.match(stripped))


class _PlanSchema(BaseModel):
    """Query understanding output."""

    intent: str = Field(default="", description="One sentence describing the goal.")
    is_conversational: bool = Field(default=False)
    requires_multi_hop: bool = Field(default=False)
    sub_questions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


class WorkflowNodes:
    """The node implementations, bound to one run's dependencies."""

    def __init__(self, context: RunContext) -> None:
        self.context = context

    # -- memory ----------------------------------------------------------

    async def load_memory(self, state: GraphState) -> GraphState:
        """Load the chat's own recent turns and the shared topic list."""
        self.context.emit_stage("Recalling the conversation")
        history, topics = await self.context.memory.context(state["chat_id"])
        return {"history": history, "global_topics": topics}

    # -- planning --------------------------------------------------------

    async def query_understanding(self, state: GraphState) -> GraphState:
        """Classify the request and decompose it into answerable sub-questions."""
        self.context.emit_stage("Understanding the question")
        query = state["query"]

        if _is_trivially_conversational(query):
            plan = QueryPlan(
                original_query=query,
                intent=query,
                is_conversational=True,
                search_queries=[query],
            )
            return {"query_plan": plan, "is_conversational": True}

        try:
            result = await self.context.client.structured(
                "retrieval/query_understanding",
                _PlanSchema,
                temperature=0.0,
                history=state.get("history", ""),
                query=query,
            )
        except ScientificRAGError as exc:
            return {"error": exc.user_message}
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "node.planning", "Query understanding failed",
                      level=30, error=str(exc))
            result = _PlanSchema(intent=query, search_queries=[query])

        plan = QueryPlan(
            original_query=query,
            intent=result.intent.strip() or query,
            is_conversational=result.is_conversational,
            requires_multi_hop=result.requires_multi_hop,
            search_queries=[q.strip() for q in result.search_queries if q.strip()] or [query],
            sub_questions=[
                SubQuestion(question=question.strip())
                for question in result.sub_questions
                if question.strip()
            ],
        )

        return {
            "query_plan": plan,
            "is_conversational": plan.is_conversational,
            "token_usage": self.context.take_usage(),
        }

    # -- corpus ----------------------------------------------------------

    async def load_documents(self, state: GraphState) -> GraphState:
        """Load the documents attached to this chat, and only this chat."""
        documents = await self.context.documents.list_indexed(state["chat_id"])
        return {"documents": documents}

    async def assess_search_need(self, state: GraphState) -> GraphState:
        """Decide whether the current corpus needs additional papers.

        Enabling automatic search grants permission to search when needed; it
        does not make every turn perform a search. The configured paper count
        is an upper bound, while the model chooses the smaller amount actually
        needed for this query.
        """
        if not state.get("auto_search_enabled"):
            return {"search_assessment": SearchAssessment()}

        documents = state.get("documents", [])
        plan: QueryPlan | None = state.get("query_plan")
        cap = max(1, self.context.settings.retrieval.max_search_papers)
        self.context.emit_stage("Checking whether new papers are needed")
        rate_limited = False

        try:
            assessment = await self.context.client.structured(
                "reasoning/search_need_assessment",
                SearchAssessment,
                temperature=0.0,
                question=state["query"],
                intent=plan.intent if plan and plan.intent else state["query"],
                existing_corpus=_render_search_corpus(documents),
                existing_count=len(documents),
                max_new_papers=cap,
                retrieval_outcome=(
                    "No pages have been read yet; assess likely coverage from metadata."
                ),
            )
            assessment = _normalise_search_assessment(
                assessment, cap=cap, corpus_empty=not documents
            )
        except Exception as exc:  # noqa: BLE001 - avoid blind network access on failure
            rate_limited = _is_rate_limit_text(str(exc))
            # If papers already exist, failure to assess coverage must not turn
            # into an automatic network request. An empty corpus cannot answer
            # a research query, so the smallest useful search is the safe
            # fallback in that one case.
            assessment = SearchAssessment(
                search_needed=not documents,
                suggested_papers=1 if not documents else 0,
                reasoning=(
                    "The corpus is empty, so one source is required."
                    if not documents
                    else "Coverage could not be assessed; existing papers will be used."
                ),
            )
            log_event(
                logger,
                "node.search_assessment",
                "Corpus coverage assessment failed",
                level=30,
                existing_papers=len(documents),
                error=str(exc),
            )

        log_event(
            logger,
            "node.search_assessment",
            "Search need assessed",
            search_needed=assessment.search_needed,
            suggested_papers=assessment.suggested_papers,
            existing_papers=len(documents),
            reason=assessment.reasoning,
        )
        update: GraphState = {
            "search_assessment": assessment,
            "token_usage": self.context.take_usage(),
        }
        if rate_limited:
            update["warnings"] = _with_rate_limit_warning(state)
        return update

    async def prepare_evidence_gap_search(self, state: GraphState) -> GraphState:
        """Require one search after the existing corpus yields zero evidence.

        Metadata can look relevant while the actual pages contain no citable
        support. That observed retrieval outcome overrides the earlier
        metadata-only decision. The model still chooses the smallest useful
        paper count, bounded by settings; failure falls back to one paper.
        """
        documents = state.get("documents", [])
        plan: QueryPlan | None = state.get("query_plan")
        cap = max(1, self.context.settings.retrieval.max_search_papers)
        self.context.emit_stage("Expanding the search after finding no evidence")
        rate_limited = False

        try:
            assessment = await self.context.client.structured(
                "reasoning/search_need_assessment",
                SearchAssessment,
                temperature=0.0,
                question=state["query"],
                intent=plan.intent if plan and plan.intent else state["query"],
                existing_corpus=_render_search_corpus(documents),
                existing_count=len(documents),
                max_new_papers=cap,
                retrieval_outcome=(
                    "The existing corpus was navigated and its selected pages were read, "
                    "but zero citable evidence items supported this query. A search is now "
                    "required; choose the smallest adequate number of new papers."
                ),
            )
            assessment = _normalise_search_assessment(
                assessment,
                cap=cap,
                corpus_empty=not documents,
                force_search=True,
            )
        except Exception as exc:  # noqa: BLE001 - one-paper recovery remains possible
            rate_limited = _is_rate_limit_text(str(exc))
            assessment = SearchAssessment(
                search_needed=True,
                suggested_papers=1,
                reasoning="Existing papers yielded no evidence, so one new source is required.",
            )
            log_event(
                logger,
                "node.evidence_gap_search",
                "Evidence-gap paper count assessment failed",
                level=30,
                error=str(exc),
            )

        update: GraphState = {
            "search_assessment": assessment,
            "token_usage": self.context.take_usage(),
        }
        if rate_limited:
            update["warnings"] = _with_rate_limit_warning(state)
        return update

    async def auto_search(self, state: GraphState) -> GraphState:
        """Find, download and index open-access papers for the question."""
        if not state.get("auto_search_enabled"):
            return {}

        assessment = state.get("search_assessment") or SearchAssessment()
        if not assessment.search_needed:
            return {}

        plan: QueryPlan | None = state.get("query_plan")
        queries = plan.search_queries if plan else [state["query"]]
        self.context.emit_stage("Searching the scientific literature")

        cap = max(1, self.context.settings.retrieval.max_search_papers)
        limit = min(cap, max(1, assessment.suggested_papers))

        added, warnings = await self.search_and_index(
            state["chat_id"],
            queries,
            limit=limit,
            existing=state.get("documents", []),
        )

        documents = list(state.get("documents", [])) + added
        return {
            "documents": documents,
            "discovered_document_ids": [doc.id for doc in added],
            "search_attempted": True,
            "warnings": list(state.get("warnings", [])) + warnings,
            "token_usage": self.context.take_usage(),
        }

    async def rank_papers(self, state: GraphState) -> GraphState:
        """Score every candidate paper so the reading budget goes to the best."""
        documents = state.get("documents", [])
        if not documents:
            return {"paper_rankings": [], "selected_document_ids": []}

        self.context.emit_stage("Ranking the available papers")
        ranker = PaperRanker(self.context.client)

        # Respect the user's configured reading budget exactly - a setting of
        # 1 means read 1 paper, not "at least 3" - and let it size the LLM
        # judgement shortlist too, so more papers requested means a wider
        # shortlist rather than a fixed cutoff.
        keep = max(1, self.context.settings.retrieval.max_search_papers)
        rankings = await ranker.rank(state["query"], documents, top_k=keep)
        selected = _select_documents_to_read(
            rankings,
            keep,
            preferred_ids=set(state.get("discovered_document_ids", [])),
        )

        update: GraphState = {
            "paper_rankings": rankings,
            "selected_document_ids": selected,
            "token_usage": self.context.take_usage(),
        }
        if ranker.rate_limit_encountered:
            update["warnings"] = _with_rate_limit_warning(state)
        return update

    # -- retrieval -------------------------------------------------------

    async def navigate(self, state: GraphState) -> GraphState:
        """Walk the document forest and decide which pages to read."""
        if not state.get("documents"):
            return {"navigation_trace": [], "pages_by_document": {}}

        self.context.emit_stage("Navigating the document structure")
        index = await self.context.indexes.load(state["chat_id"])
        if index.is_empty():
            return {
                "navigation_trace": [],
                "pages_by_document": {},
                "warnings": list(state.get("warnings", []))
                + ["The document index was empty, so no pages could be selected."],
            }

        settings = self.context.settings.retrieval
        navigator = TreeNavigator(
            self.context.client,
            max_depth=settings.retrieval_depth,
            page_budget=settings.max_pages_per_query,
        )

        plan: QueryPlan | None = state.get("query_plan")
        allowed = state.get("selected_document_ids") or None

        # A full tree walk is not free - only pay for one per sub-question
        # when the plan actually needs multi-hop reasoning across several of
        # them. Otherwise navigate once with the plan's (possibly focused,
        # e.g. from deep research) query instead of repeating the same walk
        # for near-duplicate sub-questions.
        if plan and plan.requires_multi_hop and plan.sub_questions:
            questions = plan.questions()[:3]
        else:
            questions = [plan.original_query] if plan and plan.original_query else [state["query"]]

        combined = None
        for question in questions:
            result = await navigator.navigate(index, question, allowed_document_ids=allowed)
            combined = result if combined is None else combined.merge(result)

        assert combined is not None  # questions is never empty
        update: GraphState = {
            "navigation_trace": combined.decisions,
            "pages_by_document": combined.pages_by_document,
            "token_usage": self.context.take_usage(),
        }
        if navigator.rate_limit_encountered:
            update["warnings"] = _with_rate_limit_warning(state)
        return update

    async def extract_evidence(self, state: GraphState) -> GraphState:
        """Read the selected pages and extract citable statements."""
        pages_by_document = state.get("pages_by_document", {})
        if not pages_by_document:
            return {"evidence": []}

        self.context.emit_stage("Reading the selected pages")
        extractor = EvidenceExtractor(self.context.client)
        catalogue = documents_by_id(state)
        plan: QueryPlan | None = state.get("query_plan")
        question = plan.intent if plan and plan.intent else state["query"]

        async def extract_document(
            document_id: str, page_numbers: list[int]
        ) -> list[EvidenceItem]:
            document = catalogue.get(document_id)
            if document is None:
                return []
            pages, outline = await self._load_pages(document, page_numbers)
            if not pages:
                return []
            return await extractor.extract(document, pages, question, outline)

        outcomes = await asyncio.gather(
            *(
                extract_document(document_id, page_numbers)
                for document_id, page_numbers in pages_by_document.items()
            ),
            return_exceptions=True,
        )
        collected: list[EvidenceItem] = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                log_event(
                    logger,
                    "node.evidence",
                    "Document evidence extraction failed",
                    level=30,
                    error=str(outcome),
                )
                continue
            collected.extend(outcome)

        update: GraphState = {
            "evidence": collected,
            "token_usage": self.context.take_usage(),
        }
        if extractor.rate_limit_encountered:
            update["warnings"] = _with_rate_limit_warning(state)
        return update

    async def rank_evidence(self, state: GraphState) -> GraphState:
        """De-duplicate the evidence and keep the strongest, most diverse items."""
        evidence = state.get("evidence", [])
        if not evidence:
            return {}

        limit = self.context.settings.retrieval.max_evidence_items or MAX_EVIDENCE_ITEMS
        return {"evidence": select_top(deduplicate(evidence), limit)}

    # -- answering -------------------------------------------------------

    async def conversational_answer(self, state: GraphState) -> GraphState:
        """Answer a chat-level question without touching the documents.

        Cross-chat recall is deliberately limited to topics, so the assistant
        can say what has been worked on without revealing another chat's
        content.
        """
        self.context.emit_stage("Answering")
        chunks: list[str] = []
        try:
            async for token in self.context.client.stream(
                "system/conversational",
                recent_messages=state.get("history", ""),
                global_topics=state.get("global_topics", ""),
                query=state["query"],
            ):
                chunks.append(token)
                self.context.emit_token(token)
        except ScientificRAGError as exc:
            return {"error": exc.user_message}

        return {
            "answer": "".join(chunks).strip(),
            "citations": [],
            "token_usage": self.context.take_usage(),
        }

    async def generate_answer(self, state: GraphState) -> GraphState:
        """Generate the grounded answer, streaming it to the interface."""
        evidence = state.get("evidence", [])
        if not evidence:
            return {"answer": _no_evidence_message(state)}

        mode: AnswerMode = state.get("mode", AnswerMode.SHORT_ANSWER)
        self.context.emit_stage("Writing the answer")

        values = {
            "question": state["query"],
            "evidence": format_evidence_block(evidence),
        }
        if mode is AnswerMode.SHORT_ANSWER:
            values["style_instruction"] = _STYLE_INSTRUCTIONS[mode]
        if mode is AnswerMode.DEEP_RESEARCH:
            values["research_trace"] = _render_trace(state)

        chunks: list[str] = []
        try:
            async for token in self.context.client.stream(_ANSWER_PROMPTS[mode], **values):
                chunks.append(token)
                self.context.emit_token(token)
        except ScientificRAGError as exc:
            return {"error": exc.user_message}

        return {"answer": "".join(chunks).strip(), "token_usage": self.context.take_usage()}

    async def validate_answer(self, state: GraphState) -> GraphState:
        """Audit the answer against its evidence and resolve the citations."""
        answer = state.get("answer", "")
        evidence = state.get("evidence", [])
        if not answer or not evidence:
            return {"citations": []}

        self.context.emit_stage("Verifying every claim")
        validator = AnswerValidator(
            self.context.client, strict=self.context.settings.research.strict_grounding
        )
        corrected, report = await validator.validate(state["query"], answer, evidence)
        final, citations = build_citations(corrected, evidence, documents_by_id(state))

        return {
            "answer": final,
            "citations": citations,
            "validation": report,
            "token_usage": self.context.take_usage(),
        }

    async def format_output(self, state: GraphState) -> GraphState:
        """Assemble the payload the interface renders and stores."""
        citations = state.get("citations", [])
        style = self.context.settings.research.citation_format

        payload = {
            "references": format_bibliography(citations, style) if citations else "",
            "citation_format": style.value,
            "evidence_count": len(state.get("evidence", [])),
            "documents_consulted": sorted(
                {item.document_title for item in state.get("evidence", [])}
            ),
        }
        return {"formatted_answer": payload}

    async def handle_error(self, state: GraphState) -> GraphState:
        """Turn a recorded failure into a readable reply."""
        message = state.get("error") or "The request could not be completed."
        if _is_rate_limit_text(message):
            return {
                "answer": (
                    "**The configured LLM provider has reached its rate limit.**\n\n"
                    "Please retry after the provider quota resets."
                )
            }
        return {
            "answer": (
                "**The request could not be completed automatically.**\n\n"
                "Please retry."
            )
        }

    # -- shared helpers --------------------------------------------------

    async def search_and_index(
        self,
        chat_id: str,
        queries: list[str],
        limit: int,
        existing: list[DocumentRecord],
    ) -> tuple[list[DocumentRecord], list[str]]:
        """Search, download and index new open-access papers for a chat.

        Every selected paper is downloaded concurrently, then indexed in one
        batch: their PageIndex trees are built together and the chat's
        forest is updated once for the whole set, not once per paper -
        regardless of how many papers `limit` asks for.
        """
        settings = self.context.settings
        configured_cap = max(1, settings.retrieval.max_search_papers)
        limit = min(configured_cap, max(0, limit))
        if limit == 0:
            return [], []

        aggregator = SearchAggregator(settings)
        warnings: list[str] = []

        unconfigured = [
            provider
            for provider in settings.retrieval.search_providers
            if provider is not SearchProviderType.TAVILY
            and provider not in available_providers(settings)
        ]
        warnings.extend(
            f"{provider.label} was skipped because it needs an API key. "
            "Add it under Settings \u2192 Search."
            for provider in unconfigured
        )

        try:
            results = await aggregator.search(
                queries,
                limit_per_query=max(3, limit),
                providers=effective_search_providers(settings),
            )
        except ScientificRAGError as exc:
            details = f"{exc.user_message} {exc.details or ''}"
            if _is_rate_limit_text(details):
                return [], [
                    "The configured scientific search providers reached their rate limits."
                ]
            log_event(
                logger,
                "node.auto_search",
                "Scientific search failed internally",
                level=30,
                error=str(exc),
            )
            return [], []

        warnings.extend(
            f"Scientific search provider '{name}' reached its rate limit; "
            "results from the other configured providers were used."
            for name, reason in aggregator.provider_errors.items()
            if _is_rate_limit_text(reason)
        )

        known_titles = {doc.metadata.title.lower() for doc in existing if doc.metadata.title}
        downloader = PaperDownloader()
        processor = self.context.processor()

        candidates: list[SearchResult] = []
        for result in results:
            if len(candidates) >= limit:
                break
            if result.metadata.title.lower() in known_titles:
                continue
            if not (result.is_open_access and settings.retrieval.download_pdfs):
                continue
            candidates.append(result)
            known_titles.add(result.metadata.title.lower())

        added: list[DocumentRecord] = []
        if candidates:
            semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

            async def fetch(
                result: SearchResult,
            ) -> tuple[SearchResult, bytes | None, ScientificRAGError | None]:
                async with semaphore:
                    try:
                        _, data = await downloader.download(result)
                        return result, data, None
                    except ScientificRAGError as exc:
                        return result, None, exc

            fetched = await asyncio.gather(*(fetch(result) for result in candidates))

            downloaded: list[tuple[SearchResult, bytes]] = []
            for result, data, error in fetched:
                if error is not None:
                    log_event(
                        logger,
                        "node.auto_search",
                        "Paper download failed internally",
                        level=30,
                        paper=result.metadata.title,
                        error=error.user_message,
                    )
                    continue
                downloaded.append((result, data))  # type: ignore[arg-type]

            if downloaded:
                uploads = [
                    (
                        downloader.filename_for(result),
                        data,
                        DocumentSource.AUTO_SEARCHED,
                        result.metadata,
                    )
                    for result, data in downloaded
                ]
                outcomes = await processor.process_uploads(chat_id, uploads)
                for (result, _), outcome in zip(downloaded, outcomes):
                    if outcome.error is not None:
                        if _is_rate_limit_text(outcome.error.user_message):
                            warning = (
                                "The configured LLM provider reached its rate limit while "
                                "indexing searched papers."
                            )
                            if warning not in warnings:
                                warnings.append(warning)
                        log_event(
                            logger,
                            "node.auto_search",
                            "Paper indexing failed internally",
                            level=30,
                            paper=result.metadata.title,
                            error=outcome.error.user_message,
                        )
                        continue
                    if outcome.succeeded and not outcome.reused:
                        added.append(outcome.document)

        log_event(logger, "node.auto_search", "Auto search finished",
                  chat_id=chat_id, candidates=len(results), indexed=len(added))
        return added, warnings

    async def _load_pages(self, document: DocumentRecord, page_numbers: list[int]):
        """Load only the pages the navigator selected, plus the outline."""
        parsed = document_cache.get_parsed(document.file_hash)
        if parsed is not None:
            wanted = set(page_numbers)
            return [page for page in parsed.pages if page.page_number in wanted], parsed.outline

        def read():
            loader = PDFLoader(document.file_path)
            pages = loader.load_all_pages()
            wanted = set(page_numbers)
            return [page for page in pages if page.page_number in wanted], loader.embedded_outline()

        try:
            return await asyncio.to_thread(read)
        except ScientificRAGError as exc:
            log_event(logger, "node.pages", "Pages could not be read",
                      level=30, document=document.filename, error=exc.user_message)
            return [], []


def _is_rate_limit_text(message: str) -> bool:
    text = message.lower()
    return any(marker in text for marker in (
        "rate limit", "rate_limit", "429", "quota", "tokens per minute", "tpm"
    ))


def _with_rate_limit_warning(state: GraphState) -> list[str]:
    """Append one explicit provider-limit warning to the current turn."""
    provider = (state.get("provider") or "LLM").replace("_", " ").title()
    warning = (
        f"The {provider} provider reached its rate limit during this request. "
        "A local fallback was used where available."
    )
    warnings = list(state.get("warnings", []))
    if warning not in warnings:
        warnings.append(warning)
    return warnings


def _no_evidence_message(state: GraphState) -> str:
    """Explain why no grounded answer could be produced."""
    if state.get("search_attempted"):
        if not state.get("documents"):
            return (
                "**Automatic scientific search did not find an indexable open-access "
                "paper for this question.**\n\n"
                "Try broadening the question, selecting additional scientific databases, "
                "or uploading a relevant PDF."
            )
        return (
            "**No citable evidence was found after automatic scientific search.**\n\n"
            "The existing and newly discovered papers were indexed and inspected, but their "
            "selected pages did not support this question. Try broadening the query or "
            "uploading a more directly relevant paper."
        )
    if not state.get("documents"):
        return (
            "**No documents are available in this chat yet.**\n\n"
            "Upload a PDF from the sidebar, or enable automatic scientific search "
            "so relevant open-access papers can be retrieved for you."
        )
    return (
        "**The available documents do not contain evidence for this question.**\n\n"
        "The document structure was searched but no supporting passage was found. "
        "Try rephrasing the question, adding another paper, or enabling automatic "
        "scientific search."
    )


def _normalise_search_assessment(
    assessment: SearchAssessment,
    cap: int,
    corpus_empty: bool,
    force_search: bool = False,
) -> SearchAssessment:
    """Enforce the configured cap and the minimum viable empty-corpus behavior."""
    limit = max(1, cap)
    search_needed = assessment.search_needed or corpus_empty or force_search
    suggested = min(limit, max(1, assessment.suggested_papers)) if search_needed else 0
    return assessment.model_copy(
        update={"search_needed": search_needed, "suggested_papers": suggested}
    )


def _render_search_corpus(documents: list[DocumentRecord]) -> str:
    """Render enough metadata for a coverage decision without reading or searching."""
    if not documents:
        return "No papers are currently indexed in this chat."

    entries: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        entries.append(
            f"{index}. {document.label()}\n"
            f"   Authors: {metadata.author_string()}\n"
            f"   Year: {metadata.publication_year or 'unknown'}\n"
            f"   Keywords: {', '.join(metadata.keywords) or 'not available'}\n"
            f"   Abstract: "
            f"{(metadata.abstract or 'not available')[:_SEARCH_ASSESSMENT_ABSTRACT_CHARS]}"
        )
    return "\n\n".join(entries)


def _select_documents_to_read(
    rankings: list[PaperRankingEntry], keep: int, preferred_ids: set[str]
) -> list[str]:
    """Read newly searched papers first, then fill from the global ranking."""
    limit = max(1, keep)
    selected = [
        entry.document_id for entry in rankings if entry.document_id in preferred_ids
    ][:limit]
    selected_set = set(selected)
    for entry in rankings:
        if len(selected) >= limit:
            break
        if entry.document_id in selected_set:
            continue
        selected.append(entry.document_id)
        selected_set.add(entry.document_id)
    return selected


def _render_trace(state: GraphState) -> str:
    """Render the deep research trace for the final report prompt."""
    trace = state.get("research_trace")
    if trace is None or not trace.iterations:
        return "No iterative research was performed."

    lines = [
        f"Papers examined: {trace.total_papers_examined}",
        f"Evidence items collected: {trace.total_evidence_items}",
        "",
    ]
    for iteration in trace.iterations:
        lines.append(f"Iteration {iteration.index}: {iteration.focus}")
        if iteration.search_queries:
            lines.append(f"  Queries: {', '.join(iteration.search_queries)}")
        lines.append(f"  New papers: {len(iteration.new_document_ids)}")
        lines.append(f"  Evidence found: {iteration.evidence_count}")
        if iteration.knowledge_gaps:
            lines.append("  Remaining gaps: " + "; ".join(iteration.knowledge_gaps))
    return "\n".join(lines)
