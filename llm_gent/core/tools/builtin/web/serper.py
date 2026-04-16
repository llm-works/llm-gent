"""Serper.dev Google Search backend for WebSearchTool.

Implements ``WebSearchBackend`` using the `Serper API`_.  Requires a paid API
key (``SERPER_API_KEY`` env-var or constructor arg).

.. _Serper API: https://serper.dev/
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from appinfra import DotDict
from appinfra.log import Logger
from appinfra.rate_limit import RateLimiter

from .backend import WebSearchBackendFactory


# Serper Google Search endpoint
_API_URL = "https://google.serper.dev/search"

# HTTP status codes that indicate a retriable failure
_RETRIABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class SerperSearchBackend:
    """Search backend using the Serper.dev Google Search API.

    Satisfies :class:`WebSearchBackend` protocol.  Translates Serper API
    responses into the ``{"title", "url", "snippet"}`` dicts expected by
    ``WebSearchTool``.

    Args:
        lg: Logger instance.
        api_key: Serper API key.  Falls back to ``SERPER_API_KEY``
            environment variable.
        timeout: HTTP request timeout in seconds.
        rate_limiter: Optional rate limiter.  If provided, ``next()``
            is called before each request.

    Raises:
        ValueError: If no API key is provided or found in the environment.

    Example::

        from appinfra.rate_limit import RateLimiter

        limiter = RateLimiter(lg, per_minute=60.0, initial=True)
        backend = SerperSearchBackend(lg, api_key="...", rate_limiter=limiter)
    """

    def __init__(
        self,
        lg: Logger,
        api_key: str | None = None,
        timeout: float = 10.0,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._lg = lg
        self._api_key = api_key or os.environ.get("SERPER_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Serper API key required: pass api_key= or set SERPER_API_KEY environment variable"
            )
        self._timeout = timeout
        self._rate_limiter = rate_limiter
        self._client = httpx.Client(timeout=self._timeout)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def search(self, query: str, max_results: int, offset: int = 0) -> list[dict[str, str]] | None:
        """Query Serper Google Search and return structured results.

        Note:
            The Serper API does not support pagination via offset.  If
            *offset* > 0 a trace-level log is emitted and the parameter is
            ignored.

        Returns:
            List of result dicts, empty list when the query matched nothing,
            or ``None`` on retriable failures (server error, timeout).
        """
        if self._rate_limiter:
            self._rate_limiter.next()

        if offset > 0:
            self._lg.trace(
                "serper search ignoring offset (unsupported)",
                extra={"query": query, "offset": offset},
            )

        try:
            response = self._request(query, max_results)
        except httpx.TransportError as exc:
            self._lg.warning(
                "serper search request failed",
                extra={"error": str(exc), "query": query},
            )
            return None

        return self._handle_response(response, query)

    def _handle_response(self, response: httpx.Response, query: str) -> list[dict[str, str]] | None:
        """Interpret HTTP response: check status, parse JSON body."""
        if response.status_code in _RETRIABLE_STATUS_CODES:
            self._lg.warning(
                "serper search returned retriable status",
                extra={"status": response.status_code, "query": query},
            )
            return None

        if not response.is_success:
            self._lg.warning(
                "serper search returned error",
                extra={"status": response.status_code, "query": query},
            )
            return []

        try:
            return self._parse(response.json())
        except (ValueError, KeyError):
            self._lg.warning(
                "serper search returned unparseable response",
                extra={"query": query},
            )
            return None

    def _request(self, query: str, max_results: int) -> httpx.Response:
        """Execute the HTTP request to Serper API."""
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, str | int] = {"q": query, "num": max_results}
        return self._client.post(_API_URL, headers=headers, json=payload)

    @staticmethod
    def _parse(data: dict[str, Any]) -> list[dict[str, str]]:
        """Extract ``{title, url, snippet}`` dicts from Serper API JSON."""
        results: list[dict[str, str]] = []
        for item in data.get("organic") or []:
            title = item.get("title", "")
            url = item.get("link", "")
            snippet = item.get("snippet", "")
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})
        return results


class Factory(WebSearchBackendFactory):
    """Factory for creating :class:`SerperSearchBackend` from configuration.

    Config keys:
        api_key: Serper API key (falls back to ``SERPER_API_KEY`` env-var
            if omitted).
        timeout: HTTP request timeout in seconds (default 10.0).
        per_minute: Maximum queries per minute (default 60.0).  Set to 0 or
            omit to disable rate limiting.
    """

    @classmethod
    def create(cls, lg: Logger, config: DotDict) -> SerperSearchBackend:
        """Create a SerperSearchBackend from config.

        Args:
            lg: Logger instance.
            config: Backend configuration (``api_key``, ``timeout``,
                ``per_minute``).

        Returns:
            Configured SerperSearchBackend.
        """
        per_minute = config.get("per_minute", 60.0)
        rate_limiter = RateLimiter(lg, per_minute=per_minute, initial=True) if per_minute else None
        return SerperSearchBackend(
            lg=lg,
            api_key=config.get("api_key"),
            timeout=config.get("timeout", 10.0),
            rate_limiter=rate_limiter,
        )
