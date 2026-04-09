"""Web tools: search, fetch, and pluggable search backends."""

from .backend import WebSearchBackend, WebSearchBackendFactory
from .brave import BraveSearchBackend
from .fetch import WebFetchTool
from .search import WebSearchTool


__all__ = [
    "BraveSearchBackend",
    "WebFetchTool",
    "WebSearchBackend",
    "WebSearchBackendFactory",
    "WebSearchTool",
]
