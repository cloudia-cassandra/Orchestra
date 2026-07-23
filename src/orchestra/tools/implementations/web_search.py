"""Web search tool. No search provider is wired up yet, so this is a pluggable seam:

    from orchestra.tools.implementations.web_search import set_search_provider
    set_search_provider(lambda query, max_results: [...])

until a real provider (Anthropic web search, Brave, SerpAPI, etc.) is chosen.
"""

from typing import Callable

_provider: Callable[[str, int], list[dict]] | None = None


def set_search_provider(fn: Callable[[str, int], list[dict]] | None) -> None:
    global _provider
    _provider = fn


def web_search(query: str, max_results: int = 5) -> dict:
    if _provider is None:
        raise NotImplementedError(
            "No web search provider is configured. Call set_search_provider() with a "
            "function (query, max_results) -> list[{'title', 'url', 'snippet'}]."
        )
    return {"results": _provider(query, max_results)}
