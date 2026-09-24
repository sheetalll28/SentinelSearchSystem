"""
Unified web search interface. Agents call `web_search(query)` and get back
a normalized list of {title, url, snippet} regardless of provider.
"""
from __future__ import annotations
from typing import List, TypedDict
from tenacity import retry, stop_after_attempt, wait_exponential

from config import CONFIG


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _search_duckduckgo(query: str, max_results: int) -> List[SearchResult]:
    from duckduckgo_search import DDGS
    out = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return out


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _search_tavily(query: str, max_results: int) -> List[SearchResult]:
    from tavily import TavilyClient
    if not CONFIG.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY not set in .env")
    client = TavilyClient(api_key=CONFIG.tavily_api_key)
    resp = client.search(query=query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in resp.get("results", [])
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _search_serper(query: str, max_results: int) -> List[SearchResult]:
    import requests
    if not CONFIG.serper_api_key:
        raise RuntimeError("SERPER_API_KEY not set in .env")
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": CONFIG.serper_api_key, "Content-Type": "application/json"},
        json={"q": query, "num": max_results},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
        for r in data.get("organic", [])[:max_results]
    ]


def web_search(query: str, max_results: int = 5, provider: str | None = None) -> List[SearchResult]:
    provider = (provider or CONFIG.search_provider).lower()
    if provider == "duckduckgo":
        return _search_duckduckgo(query, max_results)
    if provider == "tavily":
        return _search_tavily(query, max_results)
    if provider == "serper":
        return _search_serper(query, max_results)
    raise ValueError(f"Unknown SEARCH_PROVIDER '{provider}'. Use duckduckgo | tavily | serper.")


def fetch_page_text(url: str, char_limit: int = 4000) -> str:
    """Fetch and lightly clean a page's visible text for the Web Scout agent."""
    import requests
    from bs4 import BeautifulSoup
    from security import is_safe_url, AUDIT_LOG

    safe, reason = is_safe_url(url)
    if not safe:
        AUDIT_LOG.log("blocked_fetch", {"url": url, "reason": reason})
        return f"[fetch blocked: {reason}]"

    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "SentinelSearch/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:char_limit]
    except Exception as e:
        return f"[fetch failed: {e}]"
