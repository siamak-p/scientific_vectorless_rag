"""File-based prompt management.

Prompts are stored as plain text files under ``prompts/`` and are never
hardcoded inside Python modules. A prompt file is split into role sections
using ``--- system ---`` / ``--- human ---`` delimiters, which makes it
trivial to edit wording without touching application code.

Placeholders use ``{name}`` syntax. Literal braces are not used inside prompt
bodies, so no escaping is required.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from core.exceptions import PromptNotFoundError
from observability.logger import get_logger, log_event

logger = get_logger("prompts")

PROMPT_ROOT = Path(__file__).resolve().parent
_SECTION_RE = re.compile(r"^---\s*(system|human|assistant)\s*---\s*$", re.IGNORECASE)
_VALID_ROLES = {"system", "human", "assistant"}


class PromptManager:
    """Loads, caches and renders prompt templates from the prompt library."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or PROMPT_ROOT
        self._lock = threading.Lock()
        self._cache: dict[str, list[tuple[str, str]]] = {}

    def sections(self, name: str) -> list[tuple[str, str]]:
        """Return ``[(role, template), ...]`` for a prompt identified by name.

        Args:
            name: Dotted or slash separated prompt id, e.g. ``answer/short_answer``.
        """
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                return cached

            path = self._resolve(name)
            parsed = self._parse(path.read_text(encoding="utf-8"))
            if not parsed:
                raise PromptNotFoundError(name)
            self._cache[name] = parsed
            return parsed

    def chat_prompt(self, name: str) -> ChatPromptTemplate:
        """Return a LangChain :class:`ChatPromptTemplate` for the given prompt."""
        return ChatPromptTemplate.from_messages(self.sections(name))

    def text(self, name: str, role: str = "system") -> str:
        """Return the raw template text of one role section."""
        for section_role, body in self.sections(name):
            if section_role == role:
                return body
        raise PromptNotFoundError(f"{name}#{role}")

    def render(self, name: str, role: str = "system", **values: object) -> str:
        """Return one role section with its placeholders substituted."""
        return self.text(name, role).format(**values)

    def available(self) -> list[str]:
        """Return every prompt id present in the library, sorted."""
        ids = [
            str(path.relative_to(self._root).with_suffix("")).replace("\\", "/")
            for path in self._root.rglob("*.md")
        ]
        return sorted(ids)

    def reload(self) -> None:
        """Clear the cache so edited prompt files are picked up immediately."""
        with self._lock:
            self._cache.clear()
        log_event(logger, "prompts.reload", "Prompt cache cleared")

    # -- internals -------------------------------------------------------

    def _resolve(self, name: str) -> Path:
        relative = name.replace(".", "/")
        path = (self._root / relative).with_suffix(".md")
        if not path.is_file():
            raise PromptNotFoundError(name)
        # Guard against path traversal via crafted prompt ids.
        if self._root.resolve() not in path.resolve().parents:
            raise PromptNotFoundError(name)
        return path

    @staticmethod
    def _parse(raw: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        role: str | None = None
        buffer: list[str] = []

        for line in raw.splitlines():
            match = _SECTION_RE.match(line.strip())
            if match:
                if role and buffer:
                    sections.append((role, "\n".join(buffer).strip()))
                role = match.group(1).lower()
                buffer = []
                continue
            if role is not None:
                buffer.append(line)

        if role and buffer:
            sections.append((role, "\n".join(buffer).strip()))

        return [(r, b) for r, b in sections if r in _VALID_ROLES and b]


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    """Return the shared prompt manager instance."""
    return PromptManager()


def chat_prompt(name: str) -> ChatPromptTemplate:
    """Convenience accessor for a chat prompt template."""
    return get_prompt_manager().chat_prompt(name)
