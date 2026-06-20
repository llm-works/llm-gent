"""Web tools: search, fetch, and pluggable search backends."""

from .backend import WebSearchBackend, WebSearchBackendFactory
from .fetch import WebFetchTool
from .search import WebSearchTool


__all__ = [
    "WebFetchTool",
    "WebSearchBackend",
    "WebSearchBackendFactory",
    "WebSearchTool",
]
