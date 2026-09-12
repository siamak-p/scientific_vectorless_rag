"""Enumerations, constants and cross-platform path resolution.

This module only depends on the standard library so it can be imported from
anywhere in the application without creating import cycles.
"""

from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

APP_NAME: str = "Scientific RAG"
APP_SLUG: str = "scientific_rag"
APP_VERSION: str = "1.0.0"
APP_DESCRIPTION: str = "Vectorless Reasoning-based RAG for Scientific Papers"


# ---------------------------------------------------------------------------
# Cross-platform application directories (Windows + Linux + macOS)
# ---------------------------------------------------------------------------

def _resolve_app_home() -> Path:
    """Return the per-user application directory for the current OS.

    The location can always be overridden with the ``SCIENTIFIC_RAG_HOME``
    environment variable, which makes the application portable and keeps
    tests isolated from a developer's real configuration.
    """
    override = os.environ.get("SCIENTIFIC_RAG_HOME")
    if override:
        return Path(override).expanduser().resolve()

    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "ScientificRAG"
        return Path.home() / "AppData" / "Local" / "ScientificRAG"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_SLUG

    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / APP_SLUG
    return Path.home() / f".{APP_SLUG}"


APP_HOME: Path = _resolve_app_home()

SETTINGS_FILE: Path = APP_HOME / "settings.json"
DATA_DIR: Path = APP_HOME / "data"
CACHE_DIR: Path = APP_HOME / "cache"
UPLOAD_DIR: Path = APP_HOME / "uploads"
DOWNLOAD_DIR: Path = APP_HOME / "downloads"
LOG_DIR: Path = APP_HOME / "logs"
EXPORT_DIR: Path = APP_HOME / "exports"

DATABASE_FILE: Path = DATA_DIR / "scientific_rag.db"
CHECKPOINT_FILE: Path = DATA_DIR / "checkpoints.db"

_ALL_DIRS = (APP_HOME, DATA_DIR, CACHE_DIR, UPLOAD_DIR, DOWNLOAD_DIR, LOG_DIR, EXPORT_DIR)


def ensure_directories() -> None:
    """Create every application directory if it does not already exist."""
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

class LLMProviderType(str, Enum):
    """Supported LLM provider backends."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    CUSTOM = "custom"

    @property
    def label(self) -> str:
        """Human readable provider name for the UI."""
        return _PROVIDER_LABELS[self]

    @property
    def is_local(self) -> bool:
        """Whether the provider runs on the user's own machine."""
        return self in (LLMProviderType.OLLAMA, LLMProviderType.LMSTUDIO)

    @property
    def requires_base_url(self) -> bool:
        """Whether the endpoint is supplied by the user rather than fixed."""
        return self.is_local or self is LLMProviderType.CUSTOM

    @property
    def accepts_api_key(self) -> bool:
        """Whether a credential can be sent to this provider at all."""
        return not self.is_local

    @property
    def requires_api_key(self) -> bool:
        """Whether a credential must be configured before the provider is usable."""
        # A custom endpoint may be an unauthenticated server on the user's own
        # network, so its key is offered but never demanded.
        return self.accepts_api_key and self is not LLMProviderType.CUSTOM


_PROVIDER_LABELS: dict[LLMProviderType, str] = {
    LLMProviderType.OPENAI: "OpenAI",
    LLMProviderType.ANTHROPIC: "Anthropic Claude",
    LLMProviderType.GOOGLE: "Google Gemini",
    LLMProviderType.GROQ: "Groq",
    LLMProviderType.OLLAMA: "Ollama (local)",
    LLMProviderType.LMSTUDIO: "LM Studio (local)",
    LLMProviderType.CUSTOM: "Custom (OpenAI compatible)",
}

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
DEFAULT_LMSTUDIO_URL: str = "http://localhost:1234/v1"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentStatus(str, Enum):
    """Lifecycle states of a document inside a chat."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentSource(str, Enum):
    """How a document entered the chat."""

    UPLOADED = "uploaded"
    AUTO_SEARCHED = "auto_searched"


# ---------------------------------------------------------------------------
# Scientific search
# ---------------------------------------------------------------------------

class SearchProviderType(str, Enum):
    """Academic / scientific search backends."""

    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    EUROPE_PMC = "europe_pmc"
    DOAJ = "doaj"
    TAVILY = "tavily"
    # User-defined JSON endpoints (Settings -> Search -> Custom sources). One
    # enum member stands for all of them; each source carries its own label.
    CUSTOM = "custom"

    @property
    def label(self) -> str:
        """Human readable provider name for the UI."""
        return _SEARCH_LABELS[self]

    @property
    def requires_api_key(self) -> bool:
        """Whether the provider needs a credential to work at all."""
        return self is SearchProviderType.TAVILY

    @property
    def is_builtin_database(self) -> bool:
        """Whether this is a fixed scientific database the user can tick."""
        return self not in (SearchProviderType.TAVILY, SearchProviderType.CUSTOM)


_SEARCH_LABELS: dict[SearchProviderType, str] = {
    SearchProviderType.SEMANTIC_SCHOLAR: "Semantic Scholar",
    SearchProviderType.OPENALEX: "OpenAlex",
    SearchProviderType.CROSSREF: "Crossref",
    SearchProviderType.ARXIV: "arXiv",
    SearchProviderType.PUBMED: "PubMed",
    SearchProviderType.EUROPE_PMC: "Europe PMC",
    SearchProviderType.DOAJ: "DOAJ",
    SearchProviderType.TAVILY: "Tavily (web)",
    SearchProviderType.CUSTOM: "Custom sources",
}

# Research fields the query planner assigns and search databases declare
# coverage for (providers.yaml `covers`, custom sources' `covers`).
RESEARCH_DOMAINS: tuple[str, ...] = (
    "biomedicine",
    "physics",
    "mathematics",
    "computer_science",
    "engineering",
    "chemistry",
    "earth_and_environment",
    "social_sciences",
    "humanities",
    "economics",
    "general",
)


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

class CitationFormat(str, Enum):
    """Supported citation export styles."""

    APA = "apa"
    IEEE = "ieee"
    BIBTEX = "bibtex"


class AnswerMode(str, Enum):
    """How the system should format its response."""

    SHORT_ANSWER = "short_answer"
    RESEARCH_REPORT = "research_report"
    DEEP_RESEARCH = "deep_research"


class NodeDecision(str, Enum):
    """Whether a tree node was selected or rejected during navigation."""

    SELECTED = "selected"
    REJECTED = "rejected"
    PARTIALLY_SELECTED = "partially_selected"


class ClaimVerdict(str, Enum):
    """Outcome of validating a single claim against the evidence set."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class ExportFormat(str, Enum):
    """Supported export targets."""

    MARKDOWN = "markdown"
    PDF = "pdf"
    BIBTEX = "bibtex"


# ---------------------------------------------------------------------------
# Numeric defaults and hard limits
# ---------------------------------------------------------------------------

DEFAULT_MAX_SEARCH_PAPERS: int = 5
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_TOP_P: float = 1.0
DEFAULT_RETRIEVAL_DEPTH: int = 3

MAX_PAGES_PER_NODE: int = 10
MAX_PAGES_PER_QUERY: int = 24
MAX_EVIDENCE_ITEMS: int = 20
TOC_CHECK_PAGES: int = 20
STRUCTURE_SCAN_CHARS: int = 30_000
EVIDENCE_CONFIDENCE_THRESHOLD: float = 0.45

# A search provider always returns its best matches, however weak. Below this
# topical fit a result is treated as "nothing was found" rather than indexed.
MIN_SEARCH_RELEVANCE: float = 0.15

DEEP_RESEARCH_MAX_ITERATIONS: int = 3

MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB
PDF_MAGIC_BYTES: bytes = b"%PDF-"

EXTRACTION_CONCURRENCY: int = 4
SEARCH_CONCURRENCY: int = 4
DOWNLOAD_CONCURRENCY: int = 4
DOCUMENT_BUILD_CONCURRENCY: int = 4

HTTP_TIMEOUT_SECONDS: float = 30.0
HTTP_USER_AGENT: str = f"{APP_SLUG}/{APP_VERSION} (research assistant)"

MEMORY_RECENT_MESSAGE_WINDOW: int = 12
GLOBAL_MEMORY_MAX_TOPICS: int = 40

UNAVAILABLE_METADATA: str = "Metadata unavailable"
