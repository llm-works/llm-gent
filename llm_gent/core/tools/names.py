# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Names of the built-in tool types."""

from enum import StrEnum


class ToolName(StrEnum):
    """Tool name identifiers.

    Using str Enum so values work in YAML configs and as strings.
    ToolFactory's tool type constants are these members' values.
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
    # Note: REMEMBER/RECALL require MemoryTrait, added dynamically when needed
]
