"""Web search tool using DuckDuckGo HTML scraping.

Searches DuckDuckGo and returns structured {title, url, snippet} results.
Uses WebFetchTool for the actual HTTP request.
No API key needed.
"""

from __future__ import annotations

import re
import time
from html import unescape as html_unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

from appinfra.log import Logger

from ..base import BaseTool, ToolResult
from .web_fetch import WebFetchTool


# ---------------------------------------------------------------------------
# DuckDuckGo HTML result parsing
# ---------------------------------------------------------------------------

_DDG_URL = "https://html.duckduckgo.com/html/"

_RESULT_BLOCK_RE = re.compile(
    r'<div[^>]+class="result\s[^"]*results_links[^"]*"[^>]*>(.*?)</div>\s*</div>',
    re.DOTALL,
)
_RESULT_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_RESULT_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)


class _TagStripper(HTMLParser):
    """Minimal HTML tag stripper for short strings (titles, snippets)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)


def _strip_tags(html: str) -> str:
    """Remove HTML tags and decode entities for short strings."""
    parser = _TagStripper()
    parser.feed(html)
    return " ".join("".join(parser._parts).split())


def _unwrap_ddg_url(raw_url: str) -> str:
    """Extract the real URL from a DDG redirect link.

    DDG wraps result URLs as //duckduckgo.com/l/?uddg=<encoded_url>&...
    This extracts the actual destination URL.
    """
    decoded = html_unescape(raw_url)
    parsed = urlparse(decoded)
    uddg = parse_qs(parsed.query).get("uddg")
    if uddg:
        return uddg[0]
    # Not a redirect — return cleaned URL
    return decoded


_PARSE_WARN_THRESHOLD = 5_000  # bytes — DDG responses with results are typically >10KB


def _parse_ddg_results(lg: Logger, html: str, max_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML search results into structured data."""
    results: list[dict[str, str]] = []

    for block in _RESULT_BLOCK_RE.findall(html):
        if len(results) >= max_results:
            break

        link_match = _RESULT_LINK_RE.search(block)
        if not link_match:
            continue

        url = _unwrap_ddg_url(link_match.group(1))
        title = _strip_tags(link_match.group(2))

        snippet_match = _RESULT_SNIPPET_RE.search(block)
        snippet = _strip_tags(snippet_match.group(1)) if snippet_match else ""

        if url and title:
            results.append({"title": title, "url": url, "snippet": snippet})

    if not results and len(html) > _PARSE_WARN_THRESHOLD:
        lg.warning(
            "DDG returned large response but 0 results parsed, HTML structure may have changed",
            extra={"html_size": len(html)},
        )

    return results


def _format_results(results: list[dict[str, str]]) -> str:
    """Format search results for LLM consumption."""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   URL: {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo and return structured results.

    Returns a list of {title, url, snippet} results. The agent can then
    use WebFetchTool to read interesting pages — "dumb tools, smart agent".

    Uses DuckDuckGo's HTML interface (no API key needed). Includes simple
    rate limiting to avoid being blocked during long agent loops.

    Example:
        tool = WebSearchTool()
        result = tool.execute(query="Python asyncio tutorial")
        # result.output contains formatted search results
    """

    name = "web_search"
    description = (
        "Search the web and return a list of results with title, URL, and snippet. "
        "Use for: finding information, researching topics, discovering relevant pages. "
        "Follow up with web_fetch to read full page content."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5, max 8)",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        lg: Logger,
        max_queries_per_minute: int = 5,
        web_fetch: WebFetchTool | None = None,
    ) -> None:
        """Initialize web search tool.

        Args:
            lg: Logger instance.
            max_queries_per_minute: Rate limit. Set 0 to disable.
            web_fetch: Optional WebFetchTool instance to reuse. If None,
                creates one with default settings.
        """
        self._lg = lg
        self._rate_limit = max_queries_per_minute
        self._query_timestamps: list[float] = []
        self._web_fetch = web_fetch or WebFetchTool(lg=lg)

    def execute(self, **kwargs: Any) -> ToolResult:
        """Search DuckDuckGo and return structured results.

        Args:
            **kwargs: Must contain 'query'. Optional: 'max_results' (1-8, default 5).

        Returns:
            ToolResult with formatted search results or error.
        """
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(success=False, output="", error="Missing or empty 'query' argument")

        raw = kwargs.get("max_results")
        max_results = max(min(int(raw) if raw is not None else 5, 8), 1)

        if error := self._check_rate_limit():
            return error

        return self._search(query.strip(), max_results)

    def _check_rate_limit(self) -> ToolResult | None:
        """Check and enforce rate limiting. Returns error if limited."""
        if self._rate_limit <= 0:
            return None

        now = time.monotonic()
        cutoff = now - 60.0
        self._query_timestamps = [t for t in self._query_timestamps if t > cutoff]

        if len(self._query_timestamps) >= self._rate_limit:
            wait = 60.0 - (now - self._query_timestamps[0])
            return ToolResult(
                success=False,
                output="",
                error=f"Rate limited: max {self._rate_limit} queries/minute. "
                f"Try again in {wait:.0f}s.",
            )

        self._query_timestamps.append(now)
        return None

    def _search(self, query: str, max_results: int) -> ToolResult:
        """Execute search and parse results."""
        url = f"{_DDG_URL}?q={quote_plus(query)}"

        # Use fetch_raw to get unparsed HTML for our own regex extraction.
        fetch_result = self._web_fetch.fetch_raw(url=url)

        if not fetch_result.success:
            return ToolResult(
                success=False,
                output="",
                error=f"Search request failed: {fetch_result.error}",
            )

        results = _parse_ddg_results(self._lg, fetch_result.output, max_results)
        if not results:
            return ToolResult(success=True, output="No results found.")

        return ToolResult(success=True, output=_format_results(results))
