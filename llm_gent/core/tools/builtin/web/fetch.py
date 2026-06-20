"""Web page fetch tool with HTML-to-text conversion.

Fetches web pages and converts HTML to readable plain text suitable for
LLM consumption. Non-HTML responses (JSON, plain text) pass through as-is.

Uses trafilatura for content extraction (boilerplate removal, main content
detection). Reuses HTTPFetchTool internals (SSRF protection, IP pinning,
domain filtering).
"""

from __future__ import annotations

from typing import Any

from appinfra.log import Logger

from ...base import BaseTool, ToolResult
from ..http import HTTPFetchTool


_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class WebFetchTool(BaseTool):
    """Fetch a web page and return readable text content.

    Unlike HTTPFetchTool (which returns raw response text), this tool
    extracts the main content from HTML pages using trafilatura —
    stripping navigation, ads, footers, and boilerplate. Non-HTML
    responses (JSON, plain text) are returned as-is.

    Reuses HTTPFetchTool's SSRF protection, IP pinning, and domain filtering.

    Example:
        tool = WebFetchTool(lg)
        result = tool.execute(url="https://en.wikipedia.org/wiki/Python")
        # result.output is readable text, not raw HTML
    """

    name = "web_fetch"
    description = (
        "Fetch a web page and return its content as readable text. "
        "HTML is converted to plain text with boilerplate removed. "
        "Non-HTML content (JSON, plain text) is returned as-is. "
        "Use for: reading web pages, articles, documentation."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (must be http:// or https://)",
            },
        },
        "required": ["url"],
    }

    def __init__(
        self,
        lg: Logger,
        timeout: float = 30.0,
        max_response_size: int = 1_000_000,
        max_text_length: int = 50_000,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        block_private_ips: bool = True,
        user_agent: str | None = None,
    ) -> None:
        """Initialize web fetch tool.

        Args:
            lg: Logger instance.
            timeout: Request timeout in seconds.
            max_response_size: Maximum raw response size in bytes (passed to HTTPFetchTool).
            max_text_length: Maximum output text length after HTML conversion.
                Keeps LLM context manageable.
            allowed_domains: If set, only these domains can be accessed.
            blocked_domains: If set, these domains are blocked.
            block_private_ips: Block requests to private/internal IPs.
            user_agent: Custom User-Agent string. Defaults to a Chrome UA.
        """
        self._lg = lg
        self._max_text_length = max_text_length
        self._http = HTTPFetchTool(
            timeout=timeout,
            max_response_size=max_response_size,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            default_headers={
                "User-Agent": user_agent or _DEFAULT_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain,application/json",
                "Accept-Language": "en-US,en;q=0.9",
            },
            block_private_ips=block_private_ips,
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Fetch a web page and return readable text.

        Args:
            **kwargs: Must contain 'url'.

        Returns:
            ToolResult with readable text content or error.
        """
        result = self._http.execute(**kwargs)
        if not result.success:
            return result

        text = self._extract_content(result.output)
        return self._truncate(text)

    def fetch_raw(self, **kwargs: Any) -> ToolResult:
        """Fetch a URL and return the raw response (no HTML extraction).

        Useful when callers need the original response (e.g., to parse
        HTML themselves) while still benefiting from SSRF protection,
        domain filtering, and IP pinning.

        Args:
            **kwargs: Must contain 'url'.

        Returns:
            ToolResult with raw response body or error.
        """
        return self._http.execute(**kwargs)

    def _extract_content(self, content: str) -> str:
        """Extract main content from HTML, pass through non-HTML.

        Detection uses a simple heuristic: content starting with '<' is
        treated as HTML. This may misidentify XML/SVG, but trafilatura
        handles those gracefully (returns content or falls back to raw).
        """
        stripped = content.lstrip()
        if not stripped or stripped[0] != "<":
            return content

        import trafilatura

        extracted = trafilatura.extract(
            content,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
        return extracted if extracted else content

    def _truncate(self, text: str) -> ToolResult:
        """Truncate text to max length, including the suffix."""
        if len(text) <= self._max_text_length:
            return ToolResult(success=True, output=text)
        suffix = f"\n\n(truncated, max {self._max_text_length} chars)"
        truncated = text[: self._max_text_length - len(suffix)]
        return ToolResult(success=True, output=truncated + suffix)
