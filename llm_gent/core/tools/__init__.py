# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tool use infrastructure for agents."""

from enum import StrEnum

from .base import BaseTool, Tool, ToolCall, ToolCallResult, ToolResult
from .builtin import (
    CompleteTaskTool,
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
from .registry import Registry


class ToolName(StrEnum):
    """Tool name identifiers.

    Using str Enum so values work in YAML configs and as strings.
    Matches ToolFactory constants for consistency.
    """

    SHELL = "shell"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    HTTP_FETCH = "http_fetch"
    WEB_FETCH = "web_fetch"
    WEB_SEARCH = "web_search"
    COMPLETE_TASK = "complete_task"
    REMEMBER = "remember"
    RECALL = "recall"


# All tool types available in the platform
ALL_TOOLS: list[ToolName] = [
    ToolName.SHELL,
    ToolName.READ_FILE,
    ToolName.WRITE_FILE,
    ToolName.HTTP_FETCH,
    ToolName.WEB_FETCH,
    ToolName.WEB_SEARCH,
    ToolName.COMPLETE_TASK,
    # Note: REMEMBER/RECALL require LearnTrait, added dynamically when needed
]


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
    "CompleteTaskTool",
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
