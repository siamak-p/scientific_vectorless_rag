"""User-defined search sources.

Any academic API that answers a GET request with a JSON list of records can
be wired in from Settings without code: the user names the query parameter,
the path to the record list, and dotted paths from a record to each field.
"""

from __future__ import annotations

from typing import Any

from core.constants import SEARCH_CONCURRENCY, SearchProviderType
from core.models import AppSettings, CustomSearchSource, DocumentMetadata
from scientific_search.base import ScientificSearchProvider, SearchResult

# A worked, real example shown in Settings and loadable with one click.
# Zenodo is key-free, so the user can verify the mechanism immediately.
EXAMPLE_SOURCE = CustomSearchSource(
    id="example-zenodo",
    label="Zenodo",
    endpoint="https://zenodo.org/api/records",
    query_param="q",
    limit_param="size",
    extra_params={"type": "publication", "sort": "bestmatch"},
    results_path="hits.hits",
    field_map={
        "title": "metadata.title",
        "abstract": "metadata.description",
        "authors": "metadata.creators",
        "year": "metadata.publication_date",
        "doi": "doi",
        "url": "links.self_html",
        "pdf_url": "files.0.links.self",
    },
    covers=["all"],
)


class CustomSearchProvider(ScientificSearchProvider):
    """Searches one user-defined JSON endpoint."""

    provider_type = SearchProviderType.CUSTOM

    def __init__(self, settings: AppSettings, source: CustomSearchSource) -> None:
        super().__init__(settings)
        self.source = source

    @property
    def label(self) -> str:
        return self.source.label.strip() or "Custom source"

    def is_available(self) -> bool:
        return self.source.enabled and self.source.is_configured()

    def covers(self, domain: str) -> bool:
        return self.source.covers_domain(domain)

    def max_concurrency(self) -> int:
        # Unknown servers get the conservative treatment a public API needs.
        return min(2, SEARCH_CONCURRENCY)

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Query the endpoint and map each record through the field map."""
        source = self.source
        params: dict[str, Any] = {**source.extra_params, source.query_param or "q": query}
        if source.limit_param:
            params[source.limit_param] = limit
        headers: dict[str, str] = {}
        if source.api_key:
            if source.api_key_header:
                headers[source.api_key_header] = source.api_key
            elif source.api_key_param:
                params[source.api_key_param] = source.api_key

        payload = await self.get_json(source.endpoint.strip(), params=params, headers=headers)
        records = dig(payload, source.results_path)
        if isinstance(records, dict):
            records = list(records.values())
        results = [
            self._to_result(record) for record in (records or []) if isinstance(record, dict)
        ]
        self.log_results(query, results)
        return results

    def _to_result(self, record: dict[str, Any]) -> SearchResult:
        field = self.source.field_map
        get = lambda name: dig(record, field.get(name, ""))  # noqa: E731

        doi = self.clean_doi(_as_str(get("doi")))
        pdf_url = _as_str(get("pdf_url"))
        url = _as_str(get("url")) or (f"https://doi.org/{doi}" if doi else "")

        metadata = DocumentMetadata(
            title=self.clean_text(_as_str(get("title")), 500),
            authors=_authors(get("authors")),
            abstract=self.clean_text(_as_str(get("abstract"))),
            doi=doi,
            url=url,
            publication_year=_year(get("year")),
            provider=self.provider_type.value,
        )
        return SearchResult(
            external_id=doi or url or metadata.title,
            metadata=metadata,
            pdf_url=pdf_url,
            is_open_access=bool(pdf_url),
            provider=self.provider_type,
            provider_label=self.label,
        )


def dig(value: Any, path: str) -> Any:
    """Follow a dotted path through dicts and lists; ``None`` when it breaks."""
    if not path:
        return None
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
        if current is None:
            return None
    return current


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return _as_str(value[0]) if value else ""
    return str(value)


def _authors(value: Any) -> list[str]:
    if isinstance(value, str):
        return [name.strip() for name in value.replace(";", ",").split(",") if name.strip()]
    if isinstance(value, list):
        names: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                names.append(entry.strip())
            elif isinstance(entry, dict):
                name = entry.get("name") or entry.get("display_name") or " ".join(
                    part for part in (entry.get("given"), entry.get("family")) if part
                )
                if name:
                    names.append(str(name).strip())
        return [name for name in names if name]
    return []


def _year(value: Any) -> int | None:
    text = _as_str(value)
    digits = text[:4]
    return int(digits) if digits.isdigit() else None
