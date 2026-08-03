"""Zero-token structural parsing of PDF text."""

from __future__ import annotations

from documents.parser import (
    build_page_digest,
    extract_figures,
    extract_references,
    extract_tables,
    infer_headings,
    section_hint,
)


def test_headings_are_inferred_from_numbering(pages) -> None:
    headings = infer_headings(pages)
    titles = [entry.title for entry in headings]

    assert "Introduction" in " ".join(titles)
    assert "Model Architecture" in " ".join(titles)
    assert all(entry.page_start >= 1 for entry in headings)


def test_figures_are_detected_with_their_page(pages) -> None:
    figures = extract_figures(pages)

    assert figures
    assert figures[0].page_number == 3
    assert "Transformer" in figures[0].caption


def test_tables_are_detected_with_their_page(pages) -> None:
    tables = extract_tables(pages)

    assert tables
    assert tables[0].page_number == 4


def test_references_are_extracted(pages) -> None:
    references = extract_references(pages)

    assert len(references) >= 2
    assert any("Bahdanau" in reference for reference in references)


def test_section_hint_maps_a_page_to_its_heading(outline) -> None:
    assert section_hint(outline, 3) == "Model Architecture"
    assert section_hint(outline, 4) == "Results"


def test_section_hint_before_the_first_heading_is_empty(outline) -> None:
    assert section_hint(outline, 1) == ""


def test_page_digest_marks_page_boundaries(pages) -> None:
    digest = build_page_digest(pages, char_budget=2000)

    assert "Page 1" in digest
    assert "Page 3" in digest
    assert len(digest) <= 2200
