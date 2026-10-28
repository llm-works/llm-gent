# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tool use infrastructure for agents."""

from .base import BaseTool, Tool, ToolCall, ToolCallResult, ToolResult
from .builtin import (
    CompleteTaskTool,
    FetchResult,
    FileReadTool,
    FileWriteTool,
    HTTPFetchTool,
    RecallTool,
    RememberTool,
    ShellTool,
    WebFetchTool,
    WebSearchBackend,
    WebSearchBackendFactory,
    WebSearchTool,
)
from .executor import SimpleToolExecutor, ToolExecutionResult
from .factory import ToolFactory
from .names import ALL_TOOLS, ToolName
from .registry import Registry


__all__ = [
    # Base types
    "BaseTool",
    "Tool",
    "ToolCall",
    "ToolCallResult",
    "ToolResult",
    # Names & Catalogs
    "ToolName",
    "ALL_TOOLS",
    # Factory & Registry
    "ToolFactory",
    "Registry",
    # Executor
    "SimpleToolExecutor",
    "ToolExecutionResult",
    # Built-in tools
    "CompleteTaskTool",  # cq: keep re-exports .builtin's list; llm_gent repeats it
    "FetchResult",
    "FileReadTool",
    "FileWriteTool",
    "HTTPFetchTool",
    "RecallTool",
    "RememberTool",
    "ShellTool",
    "WebFetchTool",
    "WebSearchBackend",
    "WebSearchBackendFactory",
    "WebSearchTool",
]
