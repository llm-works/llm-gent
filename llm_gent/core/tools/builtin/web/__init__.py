# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

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
