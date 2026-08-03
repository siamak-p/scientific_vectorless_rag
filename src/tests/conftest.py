"""Shared test fixtures.

The application home is redirected to a temporary directory before any
application module is imported, so tests never touch the developer's real
settings, cache or database.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TEMP_HOME = Path(tempfile.mkdtemp(prefix="scientific-rag-tests-"))
os.environ["SCIENTIFIC_RAG_HOME"] = str(_TEMP_HOME)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from core.constants import DocumentSource, DocumentStatus  # noqa: E402
from core.models import (  # noqa: E402
    DocumentMetadata,
    DocumentRecord,
    DocumentTree,
    EvidenceItem,
    OutlineEntry,
    PageContent,
    TreeNode,
    new_id,
)


@pytest.fixture
def app_home() -> Path:
    """The temporary application home used by the whole test session."""
    return _TEMP_HOME


@pytest.fixture
def metadata() -> DocumentMetadata:
    """A fully populated metadata record."""
    return DocumentMetadata(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        abstract="We propose the Transformer, based solely on attention mechanisms.",
        doi="10.5555/3295222.3295349",
        url="https://arxiv.org/abs/1706.03762",
        keywords=["attention", "transformer", "sequence modelling"],
        publication_year=2017,
        conference="NeurIPS",
        citation_count=90000,
        is_peer_reviewed=True,
    )


@pytest.fixture
def document(metadata: DocumentMetadata) -> DocumentRecord:
    """An indexed document belonging to a chat."""
    return DocumentRecord(
        id="doc-a",
        chat_id="chat-1",
        filename="attention.pdf",
        file_hash="a" * 64,
        source=DocumentSource.UPLOADED,
        status=DocumentStatus.INDEXED,
        page_count=11,
        metadata=metadata,
    )


@pytest.fixture
def evidence(document: DocumentRecord) -> list[EvidenceItem]:
    """Three evidence items across two documents."""
    return [
        EvidenceItem(
            id="ev-1",
            document_id=document.id,
            document_title=document.label(),
            page_number=3,
            section="Model Architecture",
            content="The Transformer uses stacked self-attention layers.",
            quote="stacked self-attention",
            confidence_score=0.9,
        ),
        EvidenceItem(
            id="ev-2",
            document_id=document.id,
            document_title=document.label(),
            page_number=6,
            section="Results",
            content="The model reaches 28.4 BLEU on WMT 2014 English-to-German.",
            confidence_score=0.8,
        ),
        EvidenceItem(
            id="ev-3",
            document_id="doc-b",
            document_title="Deep Residual Learning",
            page_number=2,
            section="Introduction",
            content="Residual connections ease the training of deep networks.",
            confidence_score=0.7,
        ),
    ]


@pytest.fixture
def pages() -> list[PageContent]:
    """A small synthetic paper."""
    texts = [
        "Attention Is All You Need\nAbstract\nWe propose the Transformer.",
        "1 Introduction\nRecurrent models preclude parallelisation.",
        "2 Model Architecture\nThe Transformer follows an encoder-decoder structure.\n"
        "Figure 1: The Transformer architecture.",
        "3 Results\nTable 2: BLEU scores on WMT 2014.\nOur model reaches 28.4 BLEU.",
        "References\n[1] Bahdanau et al. Neural machine translation. 2015.\n"
        "[2] He et al. Deep residual learning. 2016.",
    ]
    return [
        PageContent(page_number=index, text=text, char_count=len(text))
        for index, text in enumerate(texts, start=1)
    ]


def make_tree(document_id: str, title: str, sections: list[str]) -> DocumentTree:
    """Build a simple two-level document tree for tests."""
    children = [
        TreeNode(
            node_id=new_id(),
            title=section,
            level=1,
            node_type="section",
            document_id=document_id,
            page_start=index * 2 + 1,
            page_end=index * 2 + 2,
        )
        for index, section in enumerate(sections)
    ]
    root = TreeNode(
        node_id=new_id(),
        title=title,
        level=0,
        node_type="document",
        document_id=document_id,
        page_start=1,
        page_end=len(sections) * 2 + 2,
        children=children,
    )
    return DocumentTree(
        document_id=document_id,
        document_title=title,
        file_hash=document_id * 8,
        root=root,
        total_pages=len(sections) * 2 + 2,
    )


@pytest.fixture
def outline() -> list[OutlineEntry]:
    """An embedded outline for the synthetic paper."""
    return [
        OutlineEntry(title="Introduction", level=1, page_start=2),
        OutlineEntry(title="Model Architecture", level=1, page_start=3),
        OutlineEntry(title="Results", level=1, page_start=4),
    ]
