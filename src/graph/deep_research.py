"""Deep research.

An iterative loop that repeats until the evidence is sufficient or the
iteration budget is exhausted. Each cycle analyses what is still missing,
turns those gaps into new searches, indexes the papers it finds, reads them,
and re-assesses. Every cycle is recorded so the user can audit how the final
report was assembled.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.constants import DEEP_RESEARCH_MAX_ITERATIONS
from core.exceptions import ScientificRAGError
from core.models import QueryPlan, ResearchIteration, ResearchTrace
from graph.context import RunContext
from graph.nodes import WorkflowNodes
from graph.state import GraphState
from observability.logger import get_logger, log_event
from retrieval.evidence import deduplicate, format_evidence_block, select_top

logger = get_logger("graph.deep_research")


class _GapAnalysis(BaseModel):
    """What the current evidence still fails to cover."""

    is_sufficient: bool = Field(default=False)
    knowledge_gaps: list[str] = Field(default_factory=list)
    next_focus: str = Field(default="")
    next_search_queries: list[str] = Field(default_factory=list)
    suggested_papers: int = Field(
        default=0,
        ge=0,
        description="Smallest number of new papers needed for the identified gaps.",
    )


class DeepResearchLoop:
    """Runs the gap-driven research cycle."""

    def __init__(self, context: RunContext, nodes: WorkflowNodes) -> None:
        self.context = context
        self.nodes = nodes
        self.max_iterations = max(
            1,
            min(
                context.settings.research.deep_research_iterations,
                DEEP_RESEARCH_MAX_ITERATIONS,
            ),
        )

    async def run(self, state: GraphState) -> GraphState:
        """Iterate until the evidence is sufficient or the budget runs out."""
        trace = ResearchTrace()
        documents = list(state.get("documents", []))
        evidence = list(state.get("evidence", []))
        warnings = list(state.get("warnings", []))
        pages_by_document = dict(state.get("pages_by_document", {}))
        navigation_trace = list(state.get("navigation_trace", []))

        plan: QueryPlan | None = state.get("query_plan")
        question = state["query"]

        for index in range(1, self.max_iterations + 1):
            self.context.emit_stage(f"Deep research: assessing coverage ({index})")
            analysis = await self._analyse(question, plan, evidence, documents, index)

            iteration = ResearchIteration(
                index=index,
                focus=analysis.next_focus or question,
                knowledge_gaps=analysis.knowledge_gaps,
                search_queries=analysis.next_search_queries,
                is_sufficient=analysis.is_sufficient,
                evidence_count=len(evidence),
            )

            if analysis.is_sufficient or not analysis.next_search_queries:
                trace.iterations.append(iteration)
                break

            self.context.emit_stage(f"Deep research: gathering new sources ({index})")
            paper_limit = _deep_search_limit(
                analysis, self.context.settings.retrieval.max_search_papers
            )
            added, new_warnings = await self.nodes.search_and_index(
                state["chat_id"],
                analysis.next_search_queries,
                limit=paper_limit,
                existing=documents,
            )
            warnings.extend(new_warnings)
            iteration.new_document_ids = [document.id for document in added]

            if not added:
                trace.iterations.append(iteration)
                break

            documents.extend(added)
            cycle_state: GraphState = {
                **state,
                "documents": documents,
                "selected_document_ids": [document.id for document in added],
                "query_plan": _focus_plan(plan, analysis, question),
            }

            navigation = await self.nodes.navigate(cycle_state)
            cycle_state = {**cycle_state, **navigation}
            extracted = await self.nodes.extract_evidence(cycle_state)

            new_evidence = extracted.get("evidence", [])
            evidence = deduplicate(evidence + new_evidence)
            navigation_trace.extend(navigation.get("navigation_trace", []))
            for document_id, pages in navigation.get("pages_by_document", {}).items():
                merged = set(pages_by_document.get(document_id, [])) | set(pages)
                pages_by_document[document_id] = sorted(merged)

            iteration.evidence_count = len(new_evidence)
            trace.iterations.append(iteration)

            log_event(logger, "deep_research.iteration", "Iteration complete",
                      iteration=index, new_papers=len(added), new_evidence=len(new_evidence))

        trace.total_papers_examined = len(documents)
        limit = self.context.settings.retrieval.max_evidence_items
        final_evidence = select_top(evidence, limit)
        trace.total_evidence_items = len(final_evidence)

        return {
            "documents": documents,
            "evidence": final_evidence,
            "navigation_trace": navigation_trace,
            "pages_by_document": pages_by_document,
            "research_trace": trace,
            "warnings": warnings,
            "token_usage": self.context.take_usage(),
        }

    async def _analyse(
        self,
        question: str,
        plan: QueryPlan | None,
        evidence: list,
        documents: list,
        iteration: int,
    ) -> _GapAnalysis:
        """Ask the model what the current evidence still fails to answer."""
        sub_questions = "\n".join(f"- {item}" for item in (plan.questions() if plan else []))
        sources = "\n".join(f"- {document.label()}" for document in documents) or "None yet."

        try:
            return await self.context.client.structured(
                "reasoning/gap_analysis",
                _GapAnalysis,
                temperature=0.0,
                question=question,
                iteration=iteration,
                max_iterations=self.max_iterations,
                max_new_papers=max(1, self.context.settings.retrieval.max_search_papers),
                sub_questions=sub_questions or f"- {question}",
                sources=sources,
                evidence=format_evidence_block(evidence[:30]),
            )
        except ScientificRAGError as exc:
            log_event(logger, "deep_research.analyse", "Gap analysis failed",
                      level=30, error=exc.user_message)
            return _GapAnalysis(is_sufficient=True)
        except Exception as exc:  # noqa: BLE001 - stop rather than loop blindly
            log_event(logger, "deep_research.analyse", "Gap analysis failed",
                      level=30, error=str(exc))
            return _GapAnalysis(is_sufficient=True)


def _focus_plan(plan: QueryPlan | None, analysis: _GapAnalysis, question: str) -> QueryPlan:
    """Build a narrowed plan aimed at the gaps found in this iteration."""
    focus = analysis.next_focus or question
    return QueryPlan(
        original_query=focus,
        intent=focus,
        search_queries=analysis.next_search_queries,
        sub_questions=(plan.sub_questions if plan else []),
        requires_multi_hop=True,
    )


def _deep_search_limit(analysis: _GapAnalysis, configured_cap: int) -> int:
    """Use the model's smallest adequate count without exceeding user settings."""
    return min(max(1, configured_cap), max(1, analysis.suggested_papers))
