"""Paper ranking and evidence selection."""

from __future__ import annotations

from datetime import datetime, timezone

from core.constants import DocumentSource, DocumentStatus
from core.models import DocumentMetadata, DocumentRecord, EvidenceItem, PageContent
from retrieval.evidence import (
    EvidenceExtractor,
    _page_batches,
    deduplicate,
    format_evidence_block,
    select_top,
)
from retrieval.ranking import (
    PaperRanker,
    citation_score,
    lexical_overlap,
    recency_score,
    venue_quality,
)


def test_recent_papers_score_higher() -> None:
    current = datetime.now(timezone.utc).year

    assert recency_score(current) > recency_score(current - 5) > recency_score(current - 20)


def test_unknown_year_gets_a_neutral_score() -> None:
    assert 0.0 < recency_score(None) < 1.0


def test_citation_score_saturates() -> None:
    assert citation_score(0) == 0.0
    assert citation_score(10) < citation_score(1000)
    assert citation_score(10_000_000) <= 1.0


def test_peer_reviewed_venues_score_higher(document: DocumentRecord) -> None:
    preprint = document.model_copy(deep=True)
    preprint.metadata = DocumentMetadata(title="Preprint")

    assert venue_quality(document) > venue_quality(preprint)


def test_lexical_overlap_rewards_matching_terms() -> None:
    terms = {"transformer", "attention", "translation"}

    assert lexical_overlap("How does attention work in transformers?", terms) > lexical_overlap(
        "What is photosynthesis in plants?", terms
    )


class _BatchExtractionClient:
    def __init__(self) -> None:
        self.calls = 0

    async def structured(self, prompt_key, model_cls, temperature=0.0, **kwargs):
        self.calls += 1
        assert "=== PAGE 3 ===" in kwargs["pages"]
        return model_cls(
            items=[
                {
                    "page_number": 3,
                    "content": "The Transformer uses self-attention.",
                    "quote": "uses self-attention",
                    "confidence_score": 0.9,
                }
            ]
        )


async def test_evidence_pages_are_batched_with_exact_provenance(
    document: DocumentRecord,
) -> None:
    client = _BatchExtractionClient()
    pages = [
        PageContent(page_number=number, text=f"Scientific content on page {number}.")
        for number in range(1, 6)
    ]

    evidence = await EvidenceExtractor(client).extract(document, pages, "How does it work?")

    assert client.calls == 1
    assert len(evidence) == 1
    assert evidence[0].page_number == 3


def test_evidence_batches_use_a_conservative_token_budget() -> None:
    pages = [
        PageContent(page_number=number, text=("machine learning evidence " * 700))
        for number in range(1, 5)
    ]

    batches = _page_batches(pages)

    assert len(batches) > 1
    assert [page.page_number for batch in batches for page in batch] == [1, 2, 3, 4]


class _SplitRecoveryClient:
    def __init__(self) -> None:
        self.calls = 0

    async def structured(self, prompt_key, model_cls, temperature=0.0, **kwargs):
        self.calls += 1
        if kwargs["pages"].count("=== PAGE") > 1:
            raise ValueError("Malformed structured response")
        page_number = int(kwargs["pages"].split("=== PAGE ", 1)[1].split(" ", 1)[0])
        return model_cls(items=[{
            "page_number": page_number,
            "content": f"Evidence from page {page_number}.",
            "confidence_score": 0.9,
        }])


async def test_failed_evidence_batch_is_retried_as_single_pages(
    document: DocumentRecord,
) -> None:
    client = _SplitRecoveryClient()
    pages = [
        PageContent(page_number=number, text=f"Machine learning evidence on page {number}.")
        for number in range(1, 3)
    ]

    evidence = await EvidenceExtractor(client).extract(document, pages, "machine learning")

    assert client.calls == 3
    assert {item.page_number for item in evidence} == {1, 2}


class _RateLimitedExtractionClient:
    async def structured(self, prompt_key, model_cls, temperature=0.0, **kwargs):
        raise RuntimeError("Rate limit exceeded: error code: 429")


async def test_rate_limit_uses_exact_page_backed_evidence(
    document: DocumentRecord,
) -> None:
    pages = [PageContent(
        page_number=7,
        text=(
            "Machine learning is a branch of artificial intelligence that learns "
            "patterns from data instead of relying only on explicit rules."
        ),
    )]

    extractor = EvidenceExtractor(_RateLimitedExtractionClient())
    evidence = await extractor.extract(
        document, pages, "What is machine learning and artificial intelligence?"
    )

    assert len(evidence) == 1
    assert evidence[0].page_number == 7
    assert evidence[0].content.startswith("Machine learning is")
    assert extractor.rate_limit_encountered is True


async def test_ranking_without_a_model_falls_back_to_lexical_scoring(
    document: DocumentRecord,
) -> None:
    other = DocumentRecord(
        id="doc-b",
        chat_id="chat-1",
        filename="photosynthesis.pdf",
        status=DocumentStatus.INDEXED,
        source=DocumentSource.AUTO_SEARCHED,
        metadata=DocumentMetadata(title="Photosynthesis in C4 plants", publication_year=1998),
    )

    rankings = await PaperRanker(client=None).rank(
        "How does attention work in the Transformer?", [other, document]
    )

    assert rankings[0].document_id == document.id
    assert rankings[0].overall_score > rankings[1].overall_score


def test_duplicate_evidence_is_collapsed(evidence: list[EvidenceItem]) -> None:
    duplicate = evidence[0].model_copy(update={"id": "ev-dup", "confidence_score": 0.5})

    unique = deduplicate(evidence + [duplicate])

    assert len(unique) == 3
    assert unique[0].id == "ev-1"


def test_selection_keeps_documents_diverse(evidence: list[EvidenceItem]) -> None:
    selected = select_top(evidence, limit=2)

    assert {item.document_id for item in selected} == {"doc-a", "doc-b"}


def test_evidence_block_is_numbered_for_citation(evidence: list[EvidenceItem]) -> None:
    block = format_evidence_block(evidence)

    assert block.startswith("E1 |")
    assert "E3 |" in block
    assert "page 3" in block


def test_empty_evidence_block_is_explicit() -> None:
    assert "No evidence" in format_evidence_block([])
