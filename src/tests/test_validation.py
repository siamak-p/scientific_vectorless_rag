"""Hallucination control."""

from __future__ import annotations

from retrieval.validation import (
    remove_statements,
    strip_invalid_markers,
    uncited_sentences,
)


def test_markers_beyond_the_evidence_list_are_stripped() -> None:
    cleaned, removed = strip_invalid_markers("Real [E2]. Invented [E7].", evidence_count=3)

    assert "[E2]" in cleaned
    assert "[E7]" not in cleaned
    assert removed == 1


def test_valid_markers_survive() -> None:
    cleaned, removed = strip_invalid_markers("Grounded [E1].", evidence_count=1)

    assert cleaned == "Grounded [E1]."
    assert removed == 0


def test_sentences_without_citations_are_flagged() -> None:
    answer = (
        "The Transformer removes recurrence entirely from the architecture [E1]. "
        "It was also the first model ever to win a Nobel Prize in physics."
    )

    uncited = uncited_sentences(answer)

    assert len(uncited) == 1
    assert "Nobel Prize" in uncited[0]


def test_headings_and_short_fragments_are_not_flagged() -> None:
    answer = "## Key Findings\n- Short bullet\nThe model reaches 28.4 BLEU on WMT 2014 [E2]."

    assert uncited_sentences(answer) == []


def test_unsupported_statements_are_deleted() -> None:
    answer = "Grounded claim [E1]. This statement is entirely fabricated and unsupported."

    corrected = remove_statements(
        answer, ["This statement is entirely fabricated and unsupported."]
    )

    assert "fabricated" not in corrected
    assert "Grounded claim [E1]." in corrected


def test_removal_ignores_fragments_that_are_too_short() -> None:
    answer = "A grounded claim [E1]."

    assert remove_statements(answer, ["A"]) == answer
