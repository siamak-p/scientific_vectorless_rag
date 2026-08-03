"""Custom exception hierarchy for Scientific RAG.

Every application specific exception inherits from :class:`ScientificRAGError`
and carries a ``user_message`` that is safe to render directly in the UI.
"""

from __future__ import annotations


class ScientificRAGError(Exception):
    """Root exception for the Scientific RAG application."""

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        details: str | None = None,
    ) -> None:
        self.user_message = message
        self.details = details
        super().__init__(message)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigurationError(ScientificRAGError):
    """The application configuration is invalid or incomplete."""


class SettingsPersistenceError(ConfigurationError):
    """Settings could not be read from or written to disk."""

    def __init__(self, path: str, reason: str = "") -> None:
        super().__init__(
            message=f"Could not access the settings file at '{path}'. {reason}".strip(),
            details=path,
        )


# ---------------------------------------------------------------------------
# LLM / provider errors
# ---------------------------------------------------------------------------

class LLMError(ScientificRAGError):
    """Base class for LLM related failures."""


class APIKeyMissingError(LLMError):
    """An API key is required but was not configured."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            message=f"API key for '{provider}' is not configured. Add it on the Settings page.",
            details=provider,
        )


class ProviderUnavailableError(LLMError):
    """The requested LLM provider cannot be reached or is not configured."""

    def __init__(self, provider: str, reason: str = "") -> None:
        super().__init__(
            message=f"Provider '{provider}' is unavailable. {reason}".strip(),
            details=provider,
        )


class LocalProviderUnavailableError(ProviderUnavailableError):
    """A local inference server (Ollama / LM Studio) is not running."""

    def __init__(self, provider: str, base_url: str) -> None:
        super().__init__(
            provider,
            f"Could not reach the local server at {base_url}. Make sure it is running.",
        )


class ModelNotFoundError(LLMError):
    """The requested model is not available on the provider."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(
            message=f"Model '{model}' is not available on provider '{provider}'.",
            details=f"{provider}/{model}",
        )


class LLMRateLimitError(LLMError):
    """Rate limit exceeded on the provider API."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            message=f"Rate limit exceeded for provider '{provider}'. Please retry shortly.",
            details=provider,
        )


class LLMTimeoutError(LLMError):
    """The LLM request timed out."""

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        super().__init__(
            message=f"The request to '{provider}' timed out after {timeout_seconds:.0f}s.",
            details=provider,
        )


class LLMInvalidCredentialsError(LLMError):
    """The configured credential was rejected by the provider."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            message=f"The API key for '{provider}' was rejected. Please verify it in Settings.",
            details=provider,
        )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentError(ScientificRAGError):
    """Base class for document processing failures."""


class InvalidPDFError(DocumentError):
    """The file is not a valid PDF or is corrupted."""

    def __init__(self, filename: str, reason: str = "") -> None:
        super().__init__(
            message=f"Invalid or corrupted PDF: '{filename}'. {reason}".strip(),
            details=filename,
        )


class UnsupportedFileError(DocumentError):
    """The uploaded file type is not supported."""

    def __init__(self, filename: str) -> None:
        super().__init__(
            message=f"'{filename}' is not a supported file type. Only PDF files are accepted.",
            details=filename,
        )


class FileTooLargeError(DocumentError):
    """The uploaded file exceeds the configured size limit."""

    def __init__(self, filename: str, size_bytes: int, limit_bytes: int) -> None:
        super().__init__(
            message=(
                f"'{filename}' is {size_bytes / 1_048_576:.1f} MB which exceeds the "
                f"{limit_bytes / 1_048_576:.0f} MB limit."
            ),
            details=filename,
        )


class PDFExtractionError(DocumentError):
    """Text extraction from a PDF failed."""

    def __init__(self, filename: str, page: int | None = None) -> None:
        page_info = f" (page {page})" if page is not None else ""
        super().__init__(
            message=f"Failed to extract text from '{filename}'{page_info}.",
            details=filename,
        )


class DocumentNotFoundError(DocumentError):
    """A referenced document does not exist in the chat."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document '{document_id}' was not found in this chat.",
            details=document_id,
        )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievalError(ScientificRAGError):
    """Base class for retrieval / PageIndex failures."""


class EmptyIndexError(RetrievalError):
    """No document index is available for retrieval."""

    def __init__(self) -> None:
        super().__init__(
            message=(
                "No documents are indexed for this chat. Upload a PDF or enable "
                "automatic scientific search."
            )
        )


class NavigationError(RetrievalError):
    """The tree navigator could not traverse the index."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(message=f"Document tree navigation failed. {reason}".strip())


class EvidenceNotFoundError(RetrievalError):
    """No supporting evidence was found for the query."""

    def __init__(self) -> None:
        super().__init__(
            message="No supporting evidence was found in the available documents."
        )


# ---------------------------------------------------------------------------
# Scientific search
# ---------------------------------------------------------------------------

class SearchError(ScientificRAGError):
    """Base class for scientific search failures."""


class SearchProviderError(SearchError):
    """A scientific search provider returned an error."""

    def __init__(self, provider: str, reason: str = "") -> None:
        super().__init__(
            message=f"Scientific search via '{provider}' failed. {reason}".strip(),
            details=provider,
        )


class PaperDownloadError(SearchError):
    """A discovered paper could not be downloaded."""


# ---------------------------------------------------------------------------
# Persistence / session
# ---------------------------------------------------------------------------

class StorageError(ScientificRAGError):
    """Base class for persistence failures."""


class DatabaseError(StorageError):
    """The local database could not be reached or updated."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(
            message=f"Local database operation failed. {reason}".strip(),
        )


class MemoryError_(StorageError):
    """Conversation memory could not be loaded or persisted."""

    def __init__(self, chat_id: str, reason: str = "") -> None:
        super().__init__(
            message=f"Conversation memory for chat '{chat_id}' is unavailable. {reason}".strip(),
            details=chat_id,
        )


class ChatNotFoundError(StorageError):
    """The requested chat does not exist."""

    def __init__(self, chat_id: str) -> None:
        super().__init__(
            message="The selected conversation no longer exists.",
            details=chat_id,
        )


# ---------------------------------------------------------------------------
# Prompts / export
# ---------------------------------------------------------------------------

class PromptNotFoundError(ScientificRAGError):
    """A prompt template file is missing from the prompt library."""

    def __init__(self, name: str) -> None:
        super().__init__(
            message=f"Prompt template '{name}' is missing from the prompt library.",
            details=name,
        )


class ExportError(ScientificRAGError):
    """Exporting a chat or report failed."""

    def __init__(self, target: str, reason: str = "") -> None:
        super().__init__(
            message=f"Export to {target} failed. {reason}".strip(),
            details=target,
        )
