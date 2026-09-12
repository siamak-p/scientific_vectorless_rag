"""The page budget must be shared fairly across every selected paper."""

from __future__ import annotations

from core.constants import NodeDecision
from core.models import ChatIndex
from retrieval.pageindex.navigator import NavigationResult, TreeNavigator
from tests.conftest import make_tree


class _AlwaysSelect:
    """A stub LLM client that judges every node worth opening."""

    def __init__(self) -> None:
        self.calls = 0

    async def structured(self, prompt_key, model_cls, temperature=0.0, **kwargs):
        self.calls += 1
        return model_cls(
            evaluations=[
                {
                    "node_id": node_id,
                    "decision": NodeDecision.SELECTED,
                    "reason": "stub",
                    "relevance_score": 0.9,
                    "confidence_score": 0.9,
                }
                for node_id in kwargs["node_ids"]
            ]
        )


class _RejectEverything:
    """A stub LLM client that judges no node worth opening."""

    def __init__(self) -> None:
        self.calls = 0

    async def structured(self, prompt_key, model_cls, temperature=0.0, **kwargs):
        self.calls += 1
        return model_cls(
            evaluations=[
                {
                    "node_id": node_id,
                    "decision": NodeDecision.REJECTED,
                    "reason": "stub",
                    "relevance_score": 0.0,
                    "confidence_score": 0.9,
                }
                for node_id in kwargs["node_ids"]
            ]
        )


class _RateLimitedNavigatorClient:
    async def structured(self, prompt_key, model_cls, temperature=0.0, **kwargs):
        raise RuntimeError("rate_limit_exceeded: tokens per minute quota reached")


async def test_a_rejected_paper_still_contributes_citable_pages() -> None:
    """One short paper used to supply every reference in the answer.

    The short paper is taken whole because it fits its share, which filled the
    result and suppressed the only fallback, so papers whose sections were all
    rejected were dropped and could never be cited.
    """
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-short", "Short Paper", ["Introduction"]))
    index.attach(
        make_tree("doc-long", "Long Paper", ["Intro", "Method", "Results", "Discussion"])
    )

    navigator = TreeNavigator(_RejectEverything(), max_depth=2, page_budget=12)

    result = await navigator.navigate(index, "What did the papers find?")

    assert set(result.pages_by_document) == {"doc-short", "doc-long"}
    assert result.pages_by_document["doc-long"]


async def test_the_rescue_reads_the_section_closest_to_the_question() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(
        make_tree("doc-a", "Paper A", ["Introduction", "Consensus throughput", "Appendix"])
    )

    navigator = TreeNavigator(_RejectEverything(), max_depth=2, page_budget=6)

    result = await navigator.navigate(index, "What is the consensus throughput?")

    # "Consensus throughput" is the second section, pages 3-4.
    assert result.pages_by_document["doc-a"][0] == 3


async def test_page_budget_is_shared_fairly_across_selected_papers() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-a", "Paper A", ["Intro", "Method", "Results", "Discussion"]))
    index.attach(make_tree("doc-b", "Paper B", ["Intro", "Method", "Results", "Discussion"]))

    # A tight global budget that a single document's sections could easily
    # exhaust on their own, if the budget were not split fairly.
    client = _AlwaysSelect()
    navigator = TreeNavigator(client, max_depth=2, page_budget=4)

    result = await navigator.navigate(index, "What did the papers find?")

    assert set(result.pages_by_document) == {"doc-a", "doc-b"}
    assert all(pages for pages in result.pages_by_document.values())
    assert client.calls == 3  # root documents + one section batch per document


async def test_a_single_document_still_gets_the_full_budget() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-a", "Paper A", ["Intro", "Method", "Results", "Discussion"]))

    navigator = TreeNavigator(_AlwaysSelect(), max_depth=2, page_budget=4)

    result = await navigator.navigate(index, "What did the paper find?")

    assert result.total_pages() == 4


async def test_small_ranked_document_is_read_without_navigation_calls() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-a", "Short Paper", ["Introduction"]))
    client = _AlwaysSelect()
    navigator = TreeNavigator(client, max_depth=2, page_budget=8)

    result = await navigator.navigate(index, "What is machine learning?")

    assert result.pages_by_document == {"doc-a": [1, 2, 3, 4]}
    assert client.calls == 0
    assert result.decisions[0].reason == "The complete ranked paper fits within its page budget."


async def test_navigation_records_provider_rate_limit() -> None:
    index = ChatIndex(chat_id="chat-1")
    index.attach(make_tree("doc-a", "Long Paper", ["Intro", "Method", "Results"]))
    navigator = TreeNavigator(_RateLimitedNavigatorClient(), max_depth=2, page_budget=4)

    result = await navigator.navigate(index, "What did the paper find?")

    assert navigator.rate_limit_encountered is True
    assert result.pages_by_document["doc-a"]


def test_merging_navigation_results_returns_the_accumulated_result() -> None:
    first = NavigationResult(pages_by_document={"doc-a": [1, 2]})
    second = NavigationResult(pages_by_document={"doc-a": [2, 3], "doc-b": [4]})

    merged = first.merge(second)

    assert merged is first
    assert merged.pages_by_document == {"doc-a": [1, 2, 3], "doc-b": [4]}
