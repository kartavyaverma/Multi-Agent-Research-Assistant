from langchain_core.tools import tool
from tavily import TavilyClient

from app.config import settings

_tavily_client = None


def get_tavily_client() -> TavilyClient:
    """Lazily instantiated so importing this module never requires a
    TAVILY_API_KEY to be set (keeps unit tests independent of external
    services, same reasoning as the lazy LLM getters in graph.py)."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    return _tavily_client


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information on a topic.
    Use this when you need current facts, statistics, or news that you
    are not confident about from memory alone.
    """
    try:
        response = get_tavily_client().search(
            query=query,
            max_results=settings.max_search_results,
            search_depth="basic",
        )
        results = response.get("results", [])
        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, start=1):
            formatted.append(
                f"[{i}] {r.get('title', '')}\n{r.get('content', '')}\nSource: {r.get('url', '')}"
            )
        return "\n\n".join(formatted)
    except Exception as exc:  # noqa: BLE001
        return f"Search failed: {exc}"


AVAILABLE_TOOLS = [web_search]
