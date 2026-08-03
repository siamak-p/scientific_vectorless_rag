"""Inspection panels: citations, evidence, retrieval reasoning and audit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.constants import CitationFormat, NodeDecision
from core.models import Citation, GeneratedAnswer, TokenUsage
from retrieval.citations import format_citation
from retrieval.pageindex.forest import render_navigation_tree
from ui import services

_DECISION_ICONS = {
    NodeDecision.SELECTED: "✓",
    NodeDecision.PARTIALLY_SELECTED: "~",
    NodeDecision.REJECTED: "✗",
}


def render_details(
    chat_id: str, answer: GeneratedAnswer, key: str, total_usage: TokenUsage | None = None
) -> None:
    """Render every inspection panel for one answer.

    ``total_usage`` is the running total across every turn in the chat so far;
    pass it only for the most recent message so its usage tab shows both its
    own cost and the chat's cumulative cost.
    """
    if not answer.evidence_items and not answer.navigation_trace:
        _render_usage(answer, total_usage)
        return

    tabs = st.tabs(
        ["Citations", "Evidence", "Retrieval reasoning", "Paper ranking", "Verification", "Usage"]
    )

    with tabs[0]:
        _render_citations(answer)
    with tabs[1]:
        _render_evidence(answer)
    with tabs[2]:
        _render_navigation(chat_id, answer, key)
    with tabs[3]:
        _render_rankings(answer)
    with tabs[4]:
        _render_verification(answer)
    with tabs[5]:
        _render_usage(answer, total_usage)
        _render_research_trace(answer)


def _render_citations(answer: GeneratedAnswer) -> None:
    """List distinct papers once, with every exact cited location beneath each."""
    if not answer.citations:
        st.caption("This answer produced no citations.")
        return

    style = st.selectbox(
        "Citation style",
        options=["apa", "ieee", "bibtex"],
        format_func=str.upper,
        key=f"style_{id(answer)}",
    )
    fmt = CitationFormat(style)
    grouped: dict[str, list[Citation]] = {}
    for citation in answer.citations:
        key = citation.document_id or citation.paper_title
        grouped.setdefault(key, []).append(citation)

    st.caption(
        f"{len(grouped)} distinct paper{'s' if len(grouped) != 1 else ''} cited."
    )
    for index, citations in enumerate(grouped.values(), start=1):
        representative = citations[0]
        rendered = format_citation(representative, fmt, index)
        st.markdown(rendered if fmt is CitationFormat.IEEE else f"**[{index}]** {rendered}")
        st.caption("Cited locations")
        for citation in citations:
            location = f"page {citation.pdf_page_number}"
            if citation.section:
                location += f" · {citation.section}"
            if citation.figure_number:
                location += f" · Figure {citation.figure_number}"
            if citation.table_number:
                location += f" · Table {citation.table_number}"
            st.markdown(f"- **{location}**")
            if citation.supporting_quote:
                st.markdown(f"> {citation.supporting_quote}")
        st.divider()


def _render_evidence(answer: GeneratedAnswer) -> None:
    """Show each extracted passage with the reason it was retrieved."""
    if not answer.evidence_items:
        st.caption("No evidence was extracted.")
        return

    for index, item in enumerate(answer.evidence_items, start=1):
        header = f"E{index} · {item.document_title} · page {item.page_number}"
        with st.expander(header, expanded=False):
            if item.section:
                st.caption(f"Section: {item.section}")
            st.markdown(item.content)
            if item.quote:
                st.markdown(f"> {item.quote}")
            st.caption(
                f"Confidence {item.confidence_score:.2f}"
                + (f" · {item.retrieval_reason}" if item.retrieval_reason else "")
            )


def _render_navigation(chat_id: str, answer: GeneratedAnswer, key: str) -> None:
    """Show the navigation tree and the per-node decision table."""
    if not answer.navigation_trace:
        st.caption("No navigation was recorded for this answer.")
        return

    index = services.load_index(chat_id)
    st.code(render_navigation_tree(index, answer.navigation_trace), language="text")

    rows = [
        {
            "": _DECISION_ICONS.get(decision.decision, "·"),
            "Document": decision.document_title or "—",
            "Section": decision.section_name,
            "Decision": decision.decision.value.replace("_", " "),
            "Relevance": round(decision.relevance_score, 2),
            "Confidence": round(decision.confidence_score, 2),
            "Pages": _page_summary(decision.related_pages),
            "Reason": decision.reason,
        }
        for decision in answer.navigation_trace
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, key=f"nav_{key}")


def _render_rankings(answer: GeneratedAnswer) -> None:
    """Show how each candidate paper was scored."""
    if not answer.paper_rankings:
        st.caption("No papers were ranked for this answer.")
        return

    rows = [
        {
            "Paper": entry.title,
            "Overall": round(entry.overall_score, 2),
            "Relevance": round(entry.relevance_score, 2),
            "Coverage": round(entry.coverage_score, 2),
            "Recency": round(entry.recency_score, 2),
            "Quality": round(entry.quality_score, 2),
            "Citations": entry.citation_count,
            "Source": entry.source.value,
            "Reason": entry.reason,
        }
        for entry in answer.paper_rankings
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_verification(answer: GeneratedAnswer) -> None:
    """Show the grounding audit for this answer."""
    report = answer.validation
    if report is None:
        st.caption("This answer was not verified because it used no evidence.")
        return

    columns = st.columns(3)
    columns[0].metric("Grounding score", f"{report.grounding_score:.2f}")
    columns[1].metric("Unsupported claims", report.unsupported_count)
    columns[2].metric("Statements removed", len(report.removed_statements))

    if report.needs_more_information:
        st.warning("The evidence was insufficient. Add papers or enable automatic search.")

    for note in report.uncertainty_notes:
        st.info(note)

    if report.claims:
        rows = [
            {
                "Claim": claim.claim,
                "Verdict": claim.verdict.value.replace("_", " "),
                "Evidence": ", ".join(claim.supporting_evidence_ids) or "—",
                "Explanation": claim.explanation,
            }
            for claim in report.claims
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_research_trace(answer: GeneratedAnswer) -> None:
    """Show what each deep research iteration contributed."""
    trace = answer.research_trace
    if trace is None or not trace.iterations:
        return

    st.markdown("##### Deep research iterations")
    for iteration in trace.iterations:
        with st.expander(f"Iteration {iteration.index} — {iteration.focus}", expanded=False):
            st.caption(f"New papers: {len(iteration.new_document_ids)}")
            st.caption(f"Evidence found: {iteration.evidence_count}")
            if iteration.search_queries:
                st.markdown("**Queries**")
                for query in iteration.search_queries:
                    st.markdown(f"- {query}")
            if iteration.knowledge_gaps:
                st.markdown("**Remaining gaps**")
                for gap in iteration.knowledge_gaps:
                    st.markdown(f"- {gap}")


def _render_usage(answer: GeneratedAnswer, total_usage: TokenUsage | None = None) -> None:
    """Show token consumption and estimated spend for this turn.

    When ``total_usage`` is given (only on the last message), also show the
    running total for the whole chat underneath.
    """
    if total_usage is not None:
        st.caption("This message")
    usage = answer.token_usage
    columns = st.columns(4)
    columns[0].metric("Input tokens", f"{usage.input_tokens:,}")
    columns[1].metric("Output tokens", f"{usage.output_tokens:,}")
    columns[2].metric("Model calls", usage.calls)
    columns[3].metric("Estimated cost", f"${usage.estimated_cost_usd:.4f}")

    if total_usage is not None:
        st.caption("Whole chat")
        totals = st.columns(4)
        totals[0].metric("Input tokens", f"{total_usage.input_tokens:,}")
        totals[1].metric("Output tokens", f"{total_usage.output_tokens:,}")
        totals[2].metric("Model calls", total_usage.calls)
        totals[3].metric("Estimated cost", f"${total_usage.estimated_cost_usd:.4f}")


def _page_summary(pages: list[int]) -> str:
    """Render a page list compactly."""
    if not pages:
        return "—"
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}–{pages[-1]} ({len(pages)})"
