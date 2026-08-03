"""Workflow assembly.

The graph routes a turn down one of three paths:

* a conversational shortcut that never touches the documents;
* the standard retrieval path: plan, rank, navigate, read, answer, verify;
* the deep research path, which loops over gap analysis before answering.

Any node that records an error short-circuits to a graceful reply.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from core.constants import AnswerMode
from core.models import AppSettings, Chat, GeneratedAnswer, TokenUsage
from graph.checkpoints import checkpoint_manager, thread_config
from graph.context import RunContext
from graph.deep_research import DeepResearchLoop
from graph.nodes import WorkflowNodes
from graph.state import GraphState, initial_state
from observability.logger import get_logger, log_event

logger = get_logger("graph.builder")


def build_graph(context: RunContext, checkpointer=None):
    """Compile the workflow for one run context."""
    nodes = WorkflowNodes(context)
    loop = DeepResearchLoop(context, nodes)

    builder = StateGraph(GraphState)

    builder.add_node("load_memory", nodes.load_memory)
    builder.add_node("query_understanding", nodes.query_understanding)
    builder.add_node("conversational_answer", nodes.conversational_answer)
    builder.add_node("load_documents", nodes.load_documents)
    builder.add_node("assess_search_need", nodes.assess_search_need)
    builder.add_node("prepare_evidence_gap_search", nodes.prepare_evidence_gap_search)
    builder.add_node("auto_search", nodes.auto_search)
    builder.add_node("rank_papers", nodes.rank_papers)
    builder.add_node("navigate", nodes.navigate)
    builder.add_node("extract_evidence", nodes.extract_evidence)
    builder.add_node("rank_evidence", nodes.rank_evidence)
    builder.add_node("deep_research", loop.run)
    builder.add_node("generate_answer", nodes.generate_answer)
    builder.add_node("validate_answer", nodes.validate_answer)
    builder.add_node("format_output", nodes.format_output)
    builder.add_node("handle_error", nodes.handle_error)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "query_understanding")

    builder.add_conditional_edges(
        "query_understanding",
        _route_after_planning,
        {
            "error": "handle_error",
            "conversational": "conversational_answer",
            "research": "load_documents",
        },
    )

    builder.add_edge("conversational_answer", "format_output")
    builder.add_edge("load_documents", "assess_search_need")
    builder.add_conditional_edges(
        "assess_search_need",
        _route_after_search_assessment,
        {"search": "auto_search", "skip": "rank_papers"},
    )
    builder.add_edge("auto_search", "rank_papers")
    builder.add_edge("rank_papers", "navigate")
    builder.add_edge("navigate", "extract_evidence")
    builder.add_edge("extract_evidence", "rank_evidence")

    builder.add_conditional_edges(
        "rank_evidence",
        _route_after_evidence,
        {
            "deep_research": "deep_research",
            "recover_search": "prepare_evidence_gap_search",
            "answer": "generate_answer",
        },
    )
    builder.add_edge("prepare_evidence_gap_search", "auto_search")

    builder.add_edge("deep_research", "generate_answer")

    builder.add_conditional_edges(
        "generate_answer",
        _route_after_answer,
        {"error": "handle_error", "validate": "validate_answer", "format": "format_output"},
    )

    builder.add_edge("validate_answer", "format_output")
    builder.add_edge("format_output", END)
    builder.add_edge("handle_error", END)

    return builder.compile(checkpointer=checkpointer)


def _route_after_planning(state: GraphState) -> str:
    """Send chat-level questions down the shortcut and failures to recovery."""
    if state.get("error"):
        return "error"
    return "conversational" if state.get("is_conversational") else "research"


def _route_after_search_assessment(state: GraphState) -> str:
    """Search only when permission is enabled and the corpus has a real gap."""
    if not state.get("auto_search_enabled"):
        return "skip"
    assessment = state.get("search_assessment")
    return "search" if assessment and assessment.search_needed else "skip"


def _route_after_evidence(state: GraphState) -> str:
    """Research deeply, answer, or recover once when retrieval found nothing."""
    if state.get("deep_research_enabled") or state.get("mode") is AnswerMode.DEEP_RESEARCH:
        return "deep_research"
    if (
        not state.get("evidence")
        and state.get("auto_search_enabled")
        and not state.get("search_attempted")
    ):
        return "recover_search"
    return "answer"


def _route_after_answer(state: GraphState) -> str:
    """Verify grounded answers; skip verification when there is no evidence."""
    if state.get("error"):
        return "error"
    return "validate" if state.get("evidence") else "format"


class WorkflowRunner:
    """Executes one user turn end to end."""

    def __init__(self, chat: Chat, settings: AppSettings, context: RunContext | None = None) -> None:
        self.context = context or RunContext.for_chat(chat, settings)
        self.chat = chat
        self.settings = settings

    async def run(
        self,
        query: str,
        mode: AnswerMode,
        explainability_enabled: bool = True,
        auto_search_enabled: bool = False,
        deep_research_enabled: bool = False,
    ) -> GeneratedAnswer:
        """Run the workflow and return the complete answer with its trace."""
        if deep_research_enabled:
            mode = AnswerMode.DEEP_RESEARCH

        checkpointer = await checkpoint_manager.saver()
        graph = build_graph(self.context, checkpointer)

        state = initial_state(
            chat_id=self.chat.id,
            query=query,
            mode=mode,
            provider=self.context.client.provider.value,
            model=self.context.client.model,
            explainability_enabled=explainability_enabled,
            auto_search_enabled=auto_search_enabled,
            deep_research_enabled=deep_research_enabled,
        )

        final: GraphState = await graph.ainvoke(state, config=thread_config(self.chat.id))

        log_event(
            logger,
            "workflow.complete",
            "Turn completed",
            chat_id=self.chat.id,
            mode=mode.value,
            evidence=len(final.get("evidence", [])),
            citations=len(final.get("citations", [])),
        )
        return to_answer(final)


def to_answer(state: GraphState) -> GeneratedAnswer:
    """Convert the terminal graph state into the public answer object."""
    payload = state.get("formatted_answer") or {}
    answer = state.get("answer", "")

    warnings = state.get("warnings", [])
    if warnings:
        notes = "\n".join(f"- {warning}" for warning in warnings)
        answer = f"{answer}\n\n---\n**Notes**\n{notes}"

    references = payload.get("references")
    if references:
        answer = f"{answer}\n\n## References\n\n{references}"

    return GeneratedAnswer(
        answer=answer.strip(),
        citations=state.get("citations", []),
        evidence_items=state.get("evidence", []),
        navigation_trace=state.get("navigation_trace", []),
        query_plan=state.get("query_plan"),
        paper_rankings=state.get("paper_rankings", []),
        validation=state.get("validation"),
        research_trace=state.get("research_trace"),
        mode=state.get("mode", AnswerMode.SHORT_ANSWER),
        token_usage=state.get("token_usage") or TokenUsage(),
    )
