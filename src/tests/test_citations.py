"""Citation resolution and bibliographic formatting."""

from __future__ import annotations

from core.constants import CitationFormat
from retrieval.citations import build_citations, format_bibliography, format_citation


def test_markers_are_renumbered_in_order_of_appearance(evidence, document) -> None:
    answer = "Self-attention scales well [E3]. The model reaches 28.4 BLEU [E2]."

    rewritten, citations = build_citations(answer, evidence, {document.id: document})

    assert "[1, p. 2]" in rewritten and "[2, p. 6]" in rewritten
    assert "[E3]" not in rewritten
    assert citations[0].evidence_id == "ev-3"
    assert citations[1].evidence_id == "ev-2"


def test_repeated_markers_reuse_the_same_number(evidence, document) -> None:
    answer = "First [E1]. Second [E1]. Third [E2]."

    rewritten, citations = build_citations(answer, evidence, {document.id: document})

    assert rewritten.count("[1, p. 3]") == 2
    assert "[1, p. 6]" in rewritten
    assert len(citations) == 2


def test_markers_without_evidence_are_removed(evidence, document) -> None:
    answer = "A grounded claim [E1]. A fabricated one [E9]."

    rewritten, citations = build_citations(answer, evidence, {document.id: document})

    assert "[E9]" not in rewritten and "p. 9" not in rewritten
    assert len(citations) == 1


def test_citation_carries_the_exact_source_location(evidence, document) -> None:
    _, citations = build_citations("Claim [E1].", evidence, {document.id: document})
    citation = citations[0]

    assert citation.pdf_page_number == 3
    assert citation.section == "Model Architecture"
    assert citation.paper_title == "Attention Is All You Need"
    assert citation.publication_year == 2017


def test_apa_format(evidence, document) -> None:
    _, citations = build_citations("Claim [E1].", evidence, {document.id: document})
    rendered = format_citation(citations[0], CitationFormat.APA)

    assert "Vaswani, A." in rendered
    assert "(2017)" in rendered
    assert "10.5555" in rendered


def test_ieee_format(evidence, document) -> None:
    _, citations = build_citations("Claim [E1].", evidence, {document.id: document})
    rendered = format_citation(citations[0], CitationFormat.IEEE, index=1)

    assert rendered.startswith("[1] A. Vaswani")
    assert "NeurIPS" in rendered


def test_bibtex_format(evidence, document) -> None:
    _, citations = build_citations("Claim [E1].", evidence, {document.id: document})
    rendered = format_citation(citations[0], CitationFormat.BIBTEX)

    assert rendered.startswith("@inproceedings{vaswani2017")
    assert "author = {Ashish Vaswani and Noam Shazeer and Niki Parmar}" in rendered


def test_missing_metadata_is_reported_not_invented(evidence) -> None:
    _, citations = build_citations("Claim [E3].", evidence, {})
    rendered = format_citation(citations[0], CitationFormat.APA)

    assert "Metadata unavailable" in rendered


def test_bibliography_is_deduplicated_by_document(evidence, document) -> None:
    _, citations = build_citations("A [E1]. B [E2].", evidence, {document.id: document})

    bibliography = format_bibliography(citations, CitationFormat.APA)

    assert bibliography.count("Attention Is All You Need") == 1


def test_distinct_papers_get_distinct_reference_numbers(evidence, document) -> None:
    rewritten, citations = build_citations(
        "Attention result [E1]. Residual result [E3]. Another attention result [E2].",
        evidence,
        {document.id: document},
    )

    assert "[1, p. 3]" in rewritten
    assert "[2, p. 2]" in rewritten
    assert "[1, p. 6]" in rewritten
    bibliography = format_bibliography(citations, CitationFormat.APA)
    assert bibliography.count("Attention Is All You Need") == 1
    assert bibliography.count("Deep Residual Learning") == 1
