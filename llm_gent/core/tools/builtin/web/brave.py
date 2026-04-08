"""Brave Search API backend for WebSearchTool.

Implements ``WebSearchBackend`` using the `Brave Web Search API`_.  Requires
a free or paid API key (``BRAVE_SEARCH_API_KEY`` env-var or constructor arg).

.. _Brave Web Search API: https://brave.com/search/api/
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import httpx
from appinfra.log import Logger

from .backend import WebSearchBackendFactory


if TYPE_CHECKING:
    from appinfra import DotDict


# Brave Web Search endpoint
_API_URL = "https://api.search.brave.com/res/v1/web/search"

# HTTP status codes that indicate a retriable failure
_RETRIABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class BraveSearchBackend:
    """Search backend using the Brave Web Search API.

    Satisfies :class:`WebSearchBackend` protocol.  Translates Brave API
    responses into the ``{"title", "url", "snippet"}`` dicts expected by
    ``WebSearchTool``.

    Args:
        lg: Logger instance.
        api_key: Brave Search API key.  Falls back to
            ``BRAVE_SEARCH_API_KEY`` environment variable.
        timeout: HTTP request timeout in seconds.

    Raises:
        ValueError: If no API key is provided or found in the environment.

    Example::

        backend = BraveSearchBackend(lg, api_key="BSA...")
        tool = WebSearchTool(lg, backend=backend)
    """

    def __init__(
        self,
        lg: Logger,
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._lg = lg
        self._api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Brave Search API key required: pass api_key= or set "
                "BRAVE_SEARCH_API_KEY environment variable"
            )
        self._timeout = timeout
        self._client = httpx.Client(timeout=self._timeout)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def search(self, query: str, max_results: int) -> list[dict[str, str]] | None:
        """Query Brave Web Search and return structured results.

        Returns:
            List of result dicts, empty list when the query matched nothing,
            or ``None`` on retriable failures (rate-limit, server error,
            timeout).
        """
        try:
            response = self._request(query, max_results)
        except httpx.TransportError as exc:
            self._lg.warning(
                "brave search request failed",
                extra={"error": str(exc), "query": query},
            )
            return None

        return self._handle_response(response, query)

    def _handle_response(self, response: httpx.Response, query: str) -> list[dict[str, str]] | None:
        """Interpret HTTP response: check status, parse JSON body."""
        if response.status_code in _RETRIABLE_STATUS_CODES:
            self._lg.warning(
                "brave search returned retriable status",
                extra={"status": response.status_code, "query": query},
            )
            return None

        if not response.is_success:
            self._lg.warning(
                "brave search returned error",
                extra={"status": response.status_code, "query": query},
            )
            return []

        try:
            return self._parse(response.json())
        except (ValueError, KeyError):
            self._lg.warning(
                "brave search returned unparseable response",
                extra={"query": query},
            )
            return None

    def _request(self, query: str, max_results: int) -> httpx.Response:
        """Execute the HTTP request to Brave Search API."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params: dict[str, str | int] = {"q": query, "count": max_results}
        return self._client.get(_API_URL, headers=headers, params=params)

    @staticmethod
    def _parse(data: dict[str, Any]) -> list[dict[str, str]]:
        """Extract ``{title, url, snippet}`` dicts from Brave API JSON."""
        results: list[dict[str, str]] = []
        for item in data.get("web", {}).get("results", []):
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("description", "")
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})
        return results


class Factory(WebSearchBackendFactory):
    """Factory for creating :class:`BraveSearchBackend` from configuration.

    Config keys:
        api_key: Brave Search API key (falls back to ``BRAVE_SEARCH_API_KEY``
            env-var if omitted).
        timeout: HTTP request timeout in seconds (default 10.0).
    """

    @classmethod
    def create(cls, lg: Logger, config: DotDict) -> BraveSearchBackend:
        """Create a BraveSearchBackend from config.

        Args:
            lg: Logger instance.
            config: Backend configuration (``api_key``, ``timeout``).

        Returns:
            Configured BraveSearchBackend.
        """
        return BraveSearchBackend(
            lg=lg,
            api_key=config.get("api_key"),
            timeout=config.get("timeout", 10.0),
        )
