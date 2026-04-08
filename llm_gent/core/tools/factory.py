"""Tool factory for creating Tool instances from configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger


if TYPE_CHECKING:
    from llm_gent.core.tools.base import Tool
    from llm_gent.core.traits.builtin.learn import LearnTrait


class ToolFactory:
    """Factory for creating Tool instances from configuration.

    Supports built-in tool types and custom tool registration.

    Example:
        factory = ToolFactory(lg)

        # Create built-in tools
        shell = factory.create(ToolFactory.SHELL, {"allowed_commands": ["ls", "grep"]})
        reader = factory.create(ToolFactory.READ_FILE, {"allowed_paths": ["/home"]})

        # Register custom tool type
        factory.register("my_tool", lambda config: MyTool(**config))
        custom = factory.create("my_tool", {"option": "value"})
    """

    # Canonical tool type constants
    SHELL = "shell"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    HTTP_FETCH = "http_fetch"
    WEB_FETCH = "web_fetch"
    WEB_SEARCH = "web_search"
    COMPLETE_TASK = "complete_task"
    REMEMBER = "remember"
    RECALL = "recall"

    # Built-in tool type aliases
    _ALIASES: dict[str, str] = {
        "file_read": READ_FILE,
        "file_write": WRITE_FILE,
        "fetch": HTTP_FETCH,
    }

    def __init__(self, lg: Logger) -> None:
        """Initialize factory with built-in tool creators.

        Args:
            lg: Logger instance, passed through to tools that need it.
        """
        self._lg = lg
        self._creators: dict[str, Callable[[dict[str, Any]], Tool]] = {}
        self._learn_trait: LearnTrait | None = None
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in tool creators."""
        from llm_gent.core.tools.builtin import (
            CompleteTaskTool,
            FileReadTool,
            FileWriteTool,
            HTTPFetchTool,
            ShellTool,
            WebFetchTool,
            WebSearchTool,
        )

        self._creators[self.SHELL] = lambda c: ShellTool(**c)
        self._creators[self.READ_FILE] = lambda c: FileReadTool(**c)
        self._creators[self.WRITE_FILE] = lambda c: FileWriteTool(**c)
        self._creators[self.HTTP_FETCH] = lambda c: HTTPFetchTool(**c)
        self._creators[self.WEB_FETCH] = lambda c: WebFetchTool(lg=self._lg, **c)
        self._creators[self.WEB_SEARCH] = lambda c: WebSearchTool(lg=self._lg, **c)
        self._creators[self.COMPLETE_TASK] = lambda _: CompleteTaskTool()

    def set_learn_trait(self, learn_trait: LearnTrait | None) -> None:
        """Set LearnTrait for memory tools (remember/recall).

        Args:
            learn_trait: LearnTrait instance or None.
        """
        self._learn_trait = learn_trait

    def register(self, tool_type: str, creator: Callable[[dict[str, Any]], Tool]) -> None:
        """Register a custom tool type.

        Args:
            tool_type: Tool type identifier.
            creator: Callable that takes config dict and returns Tool.

        Example:
            factory.register("my_tool", lambda c: MyTool(**c))
        """
        self._creators[tool_type] = creator

    def create(self, tool_type: str, config: dict[str, Any] | None = None) -> Tool | None:
        """Create a tool from type and configuration.

        Args:
            tool_type: Tool type (e.g., "shell", "read_file").
            config: Tool-specific configuration.

        Returns:
            Configured Tool instance, or None if tool cannot be created
            (e.g., memory tools without LearnTrait).

        Raises:
            ValueError: If tool type is unknown.
        """
        config = config or {}
        canonical_type = self._ALIASES.get(tool_type, tool_type)

        # Handle memory tools specially (need LearnTrait)
        if canonical_type in (self.REMEMBER, self.RECALL):
            return self._create_memory_tool(canonical_type)

        creator = self._creators.get(canonical_type)
        if creator is None:
            raise ValueError(f"Unknown tool type: {tool_type}")

        return creator(config)

    def _create_memory_tool(self, tool_type: str) -> Tool | None:
        """Create remember or recall tool.

        Returns None if LearnTrait not available (tool will be skipped).
        """
        from llm_gent.core.tools.builtin import RecallTool, RememberTool

        if self._learn_trait is None:
            return None  # Caller should skip this tool

        if tool_type == self.REMEMBER:
            return RememberTool(self._learn_trait)
        return RecallTool(self._learn_trait)
