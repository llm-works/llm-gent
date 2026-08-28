# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""WebSearchBackend protocol and factory ABC.

Defines the contract for search backend implementations and the factory
pattern for creating backends from configuration.  Follows the
``llm-infer`` ``Backend.from_config()`` pattern.

External backend authors implement both:

1. A class satisfying :class:`WebSearchBackend` (the runtime contract).
2. A :class:`WebSearchBackendFactory` subclass whose ``create()`` classmethod
   constructs the backend from a logger and config dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from appinfra import DotDict
    from appinfra.log import Logger


@runtime_checkable
class WebSearchBackend(Protocol):
    """Backend for web search providers.

    Implementations handle the actual HTTP requests and result parsing for a
    specific search provider (e.g. Brave Search API, SerpAPI, custom scraper).

    Return conventions:
        - ``list[dict]``: successful results (may be empty for genuine no-results)
        - ``None``: retriable failure (e.g. rate limit, challenge page) — the
          caller will back off and retry once before giving up.
    """

    def search(self, query: str, max_results: int, offset: int = 0) -> list[dict[str, str]] | None:
        """Execute a search query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            offset: Number of results to skip (for pagination).

        Returns:
            List of ``{"title": ..., "url": ..., "snippet": ...}`` dicts,
            empty list for no results, or ``None`` to signal a retriable failure.
        """
        ...


class WebSearchBackendFactory(ABC):
    """ABC for creating :class:`WebSearchBackend` instances from configuration.

    Each backend implementation provides a ``Factory`` subclass that knows how
    to construct the backend from a logger and a config :class:`DotDict`.

    Example::

        from appinfra import DotDict
        from appinfra.log import Logger
        from llm_gent.core.tools.builtin.web.backend import (
            WebSearchBackendFactory,
        )

        class Factory(WebSearchBackendFactory):
            @classmethod
            def create(cls, lg: Logger, config: DotDict) -> MySearchBackend:
                return MySearchBackend(
                    lg=lg,
                    rate_limit=config.get("per_minute", 6.0),
                )
    """

    @classmethod
    @abstractmethod
    def create(cls, lg: Logger, config: DotDict) -> WebSearchBackend:
        """Create a backend instance from configuration.

        Args:
            lg: Logger instance.
            config: Backend-specific configuration from YAML.

        Returns:
            Configured backend instance satisfying :class:`WebSearchBackend`.
        """
        ...


def validated_factory(cls: Any, label: str) -> type[WebSearchBackendFactory]:
    """Verify *cls* is a :class:`WebSearchBackendFactory` subclass.

    Args:
        cls: Candidate class to validate.
        label: Human-readable label for error messages (e.g. dotted path
            or built-in type name).

    Raises:
        TypeError: If *cls* is not a ``WebSearchBackendFactory`` subclass.
    """
    if not (isinstance(cls, type) and issubclass(cls, WebSearchBackendFactory)):
        raise TypeError(f"{label} resolved to {cls!r}, not a WebSearchBackendFactory subclass")
    return cls
