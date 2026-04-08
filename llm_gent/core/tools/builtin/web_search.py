"""Web search tool with pluggable backend.

Provides ``WebSearchTool`` — an LLM-facing tool that delegates actual search
to a ``WebSearchBackend`` implementation.  The tool handles rate limiting,
input validation, result formatting, and automatic retry on retriable failures.

The search backend is injected at construction time, keeping provider-specific
code (API keys, HTML parsing, etc.) out of this module.
"""

from __future__ import annotations

import time
from typing import Any

from appinfra.log import Logger

from ..base import BaseTool, ToolResult, WebSearchBackend


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
    """Search the web and return structured results.

    Returns a list of {title, url, snippet} results. The agent can then
    use WebFetchTool to read interesting pages — "dumb tools, smart agent".

    Requires a ``WebSearchBackend`` that handles provider-specific search
    logic (HTTP requests, response parsing, etc.).

    Example:
        tool = WebSearchTool(lg, backend=my_backend)
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
        backend: WebSearchBackend,
        max_queries_per_minute: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        """Initialize web search tool.

        Args:
            lg: Logger instance.
            backend: Search backend implementation.
            max_queries_per_minute: Rate limit. Set 0 to disable.
            retry_delay: Seconds to wait before retrying after a retriable failure.
        """
        self._lg = lg
        self._backend = backend
        self._rate_limit = max_queries_per_minute
        self._retry_delay = retry_delay
        self._query_timestamps: list[float] = []

    def execute(self, **kwargs: Any) -> ToolResult:
        """Search the web and return structured results.

        Args:
            **kwargs: Must contain 'query'. Optional: 'max_results' (1-8, default 5).

        Returns:
            ToolResult with formatted search results or error.
        """
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(success=False, output="", error="Missing or empty 'query' argument")

        raw = kwargs.get("max_results")
        try:
            max_results = max(min(int(raw) if raw is not None else 5, 8), 1)
        except (ValueError, TypeError):
            max_results = 5

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
        """Execute search with one automatic retry on retriable failures."""
        results = self._backend.search(query, max_results)
        if results is not None:
            return self._format(results)

        self._lg.info(
            "search backend returned retriable failure, backing off before retry",
            extra={"retry_delay": self._retry_delay},
        )
        time.sleep(self._retry_delay)

        results = self._backend.search(query, max_results)
        if results is not None:
            return self._format(results)

        return ToolResult(
            success=False,
            output="",
            error="Search backend returned a retriable error twice. "
            "Try a different query or wait before searching again.",
        )

    @staticmethod
    def _format(results: list[dict[str, str]]) -> ToolResult:
        """Convert backend results to ToolResult."""
        if not results:
            return ToolResult(success=True, output="No results found.")
        return ToolResult(success=True, output=_format_results(results))
