"""Application settings loading, validation and persistence.

Settings live in a JSON file inside the per-user application home directory
(see :mod:`core.constants`). Secrets are **never** stored in source control:
API keys are entered through the Settings page, or supplied through
environment variables / a ``.env`` file, which always take precedence.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from core.constants import (
    APP_HOME,
    SETTINGS_FILE,
    LLMProviderType,
    ensure_directories,
)
from core.exceptions import SettingsPersistenceError
from core.models import AppSettings, ProviderSettings
from observability.logger import get_logger, log_event

logger = get_logger("config.settings")

# Environment variable used to inject each provider credential without the UI.
_ENV_KEYS: dict[LLMProviderType, str] = {
    LLMProviderType.OPENAI: "OPENAI_API_KEY",
    LLMProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProviderType.GOOGLE: "GOOGLE_API_KEY",
    LLMProviderType.GROQ: "GROQ_API_KEY",
}

_ENV_BASE_URLS: dict[LLMProviderType, str] = {
    LLMProviderType.OLLAMA: "OLLAMA_BASE_URL",
    LLMProviderType.LMSTUDIO: "LMSTUDIO_BASE_URL",
}


def _load_dotenv() -> None:
    """Load a ``.env`` file from the project root or the app home, if present."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        return

    for candidate in (
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
        APP_HOME / ".env",
    ):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


class SettingsManager:
    """Thread safe loader/saver for :class:`~core.models.AppSettings`."""

    def __init__(self, settings_file: Path | None = None) -> None:
        self._settings_file = settings_file or SETTINGS_FILE
        self._lock = threading.Lock()
        self._cache: AppSettings | None = None
        ensure_directories()
        _load_dotenv()

    # -- public API ------------------------------------------------------

    def load(self, refresh: bool = False) -> AppSettings:
        """Return the current settings, reading from disk when needed."""
        with self._lock:
            if self._cache is not None and not refresh:
                return self._cache.model_copy(deep=True)

            settings = self._read_from_disk()
            self._apply_defaults(settings)
            self._apply_environment(settings)
            self._cache = settings
            return settings.model_copy(deep=True)

    def save(self, settings: AppSettings) -> None:
        """Persist settings atomically, then refresh the in-memory cache."""
        with self._lock:
            self._apply_defaults(settings)
            payload = settings.model_dump(mode="json")
            temp_path = self._settings_file.with_suffix(".tmp")
            try:
                self._settings_file.parent.mkdir(parents=True, exist_ok=True)
                temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(temp_path, self._settings_file)
                self._restrict_permissions()
            except OSError as exc:
                raise SettingsPersistenceError(str(self._settings_file), str(exc)) from exc

            self._cache = settings.model_copy(deep=True)

        log_event(logger, "settings.save", "Settings persisted",
                  active_provider=settings.active_provider)

    def invalidate(self) -> None:
        """Drop the in-memory cache so the next load re-reads the file."""
        with self._lock:
            self._cache = None

    # -- internals -------------------------------------------------------

    def _read_from_disk(self) -> AppSettings:
        if not self._settings_file.is_file():
            return AppSettings()
        try:
            raw = json.loads(self._settings_file.read_text(encoding="utf-8"))
            return AppSettings.model_validate(raw)
        except (OSError, ValueError) as exc:
            backup = self._settings_file.with_suffix(".invalid")
            log_event(
                logger,
                "settings.load",
                f"Settings file was unreadable and has been moved to {backup.name}",
                level=30,
                error=str(exc),
            )
            try:
                os.replace(self._settings_file, backup)
            except OSError:
                pass
            return AppSettings()

    @staticmethod
    def _apply_defaults(settings: AppSettings) -> None:
        """Guarantee every provider has an entry so the UI can bind to it."""
        for provider_type in LLMProviderType:
            settings.provider(provider_type)

    @staticmethod
    def _apply_environment(settings: AppSettings) -> None:
        """Overlay credentials coming from the environment (never persisted)."""
        for provider_type, env_name in _ENV_KEYS.items():
            value = os.environ.get(env_name, "").strip()
            if value:
                settings.provider(provider_type).api_key = value

        for provider_type, env_name in _ENV_BASE_URLS.items():
            value = os.environ.get(env_name, "").strip()
            if value:
                settings.provider(provider_type).base_url = value

        tavily = os.environ.get("TAVILY_API_KEY", "").strip()
        if tavily:
            settings.tavily_api_key = tavily

        semantic = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if semantic:
            settings.semantic_scholar_api_key = semantic

        contact = os.environ.get("SCIENTIFIC_RAG_CONTACT_EMAIL", "").strip()
        if contact:
            settings.contact_email = contact

    def _restrict_permissions(self) -> None:
        """Restrict the settings file to the current user on POSIX systems."""
        if os.name == "posix":
            try:
                os.chmod(self._settings_file, 0o600)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Module level singleton
# ---------------------------------------------------------------------------

_manager = SettingsManager()


def get_settings(refresh: bool = False) -> AppSettings:
    """Return the current application settings."""
    return _manager.load(refresh=refresh)


def save_settings(settings: AppSettings) -> None:
    """Persist the given application settings."""
    _manager.save(settings)


def get_provider_settings(provider_type: LLMProviderType) -> ProviderSettings:
    """Return the stored configuration for a single provider."""
    return get_settings().provider(provider_type)
